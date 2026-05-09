import math
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..modules.screening import GatedScreening, unit_length_norm
from ..modules.common import SwiGLU
from ..flow import image_pred_to_velocity_pred
from ..image import tensor_to_pil


class BottleneckPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        bottleneck_dim: int,
        patch_size: int,
    ):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.out_proj = nn.Conv2d(
            bottleneck_dim,
            hidden_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, in_channels, height, width]
    ) -> torch.Tensor:
        patches = self.in_proj(
            images
        )  # [batch_size, bottleneck_dim, height // patch_size, width // patch_size]
        patches = self.out_proj(
            patches
        )  # [batch_size, hidden_dim, height // patch_size, width // patch_size]
        patches = patches.flatten(2).transpose(
            1, 2
        )  # [batch_size, num_patches, hidden_dim]

        return unit_length_norm(patches)


class LabelEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        num_repeats: int = 8,
    ):
        super().__init__()

        self.num_repeats = num_repeats
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_classes + 1,
            embedding_dim,
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(num_repeats, embedding_dim),
            requires_grad=True,
        )

        self.uncond_id = (
            num_classes  # The last ID is reserved for unconditional embedding
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.1 / math.sqrt(self.embedding_dim),
        )
        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.1 / math.sqrt(self.embedding_dim),
        )

    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        return (
            unit_length_norm(self.embedding(label_ids))
            .unsqueeze(1)
            .expand(-1, self.num_repeats, -1)
        ) + self.positional_embedding.unsqueeze(
            0
        )  # [batch_size, num_repeats, embedding_dim]


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        time_embedding_dim: int,
        hidden_dim: int,
        num_repeats: int = 8,
        max_period: float = 10_000.0,
    ):
        super().__init__()

        self.num_repeats = num_repeats
        self.time_embedding_dim = time_embedding_dim
        self.max_period = max_period

        assert hidden_dim % 2 == 0, (
            "Hidden dimension must be even for Fourier features."
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(num_repeats, hidden_dim),
            requires_grad=True,
        )
        self.mlp = SwiGLU(time_embedding_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.1 / math.sqrt(self.time_embedding_dim),
        )

    def fourier_features(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: shape [batch]
        returns: shape [batch, embedding_dim]
        """
        if tau.ndim != 1:
            raise ValueError("tau must have shape [batch].")

        half_dim = self.time_embedding_dim // 2
        device = tau.device
        dtype = tau.dtype

        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=dtype)
            / half_dim
        )

        args = tau[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_embed = self.fourier_features(timestep)  # [batch_size, embedding_dim]
        time_embed = self.mlp(time_embed)  # [batch_size, hidden_dim]
        time_embed = time_embed.unsqueeze(1).expand(
            -1, self.num_repeats, -1
        )  # [batch_size, num_repeats, hidden_dim]

        return time_embed + self.positional_embedding.unsqueeze(
            0
        )  # [batch_size, num_repeats, hidden_dim]


class MultiScreenForClassFlowMatching(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # phi^2
        num_heads: int,  # phi
        num_blocks: int,  # phi
        num_classes: int,
        num_repeats: int = 8,
        bottleneck_dim: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        window_threshold: float = 256.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_blocks = num_blocks
        self.window_threshold = window_threshold
        self.bottleneck_dim = bottleneck_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_labels = num_classes
        self.num_repeats = num_repeats

        # embeddings
        self.patch_embedding = BottleneckPatchEmbedding(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            patch_size=patch_size,
        )
        self.label_embedding = LabelEmbedding(
            num_classes=num_classes,
            embedding_dim=hidden_dim,
            num_repeats=num_repeats,
        )
        self.time_embedding = TimeEmbedding(
            time_embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_repeats=num_repeats,
        )

        self.layers = nn.ModuleList(
            [
                GatedScreening(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    window_threshold=window_threshold,
                    num_layers=num_blocks,
                    is_causal=False,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = nn.Linear(
            hidden_dim,
            patch_size * patch_size * in_channels,
            bias=False,
        )

        self.gradient_checkpointing = False

        self.init_weights()

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "to_gate" in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                elif "to_q" in name or "to_k" in name or "to_v" in name:
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.1 / math.sqrt(self.head_dim),
                    )
                elif "to_out" in name:
                    nn.init.normal_(
                        module.weight, mean=0.0, std=0.1 / math.sqrt(self.hidden_dim)
                    )
                else:
                    nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=0.1 / math.sqrt(module.in_features),
                    )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.1 / math.sqrt(module.embedding_dim)
                )

    def set_gradient_checkpointing(self, value: bool):
        self.gradient_checkpointing = value

    def set_trace_screening(self, value: bool):
        for layer in self.layers:
            layer.set_trace_screening(value)  # type: ignore

    def prepare_patch_position_ids(
        self,
        height: int,
        width: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        num_patches = h_patch * w_patch

        position_ids = torch.stack(
            [
                torch.arange(w_patch).unsqueeze(0).expand(h_patch, -1)
                - w_patch / 2,  # x position
                torch.arange(h_patch).unsqueeze(1).expand(-1, w_patch)
                - h_patch / 2,  # y position
            ],
            dim=-1,
        ).view(num_patches, 2)  # [num_patches, 2]
        modal_ids = torch.full((num_patches, 1), fill_value=2)

        position_ids = torch.cat([position_ids, modal_ids], dim=-1)  # [num_patches, 3]

        return position_ids.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_patches, 3]

    def prepare_label_position_ids(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        position_ids = (
            torch.arange(self.num_repeats).unsqueeze(1).repeat(1, 2)
        ) - self.num_repeats / 2  # [num_repeats, 2]
        modal_ids = torch.full((self.num_repeats, 1), fill_value=1)

        position_ids = torch.cat([position_ids, modal_ids], dim=-1)  # [num_repeats, 3]

        return position_ids.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_repeats, 3]

    def prepare_time_position_ids(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        position_ids = (
            torch.arange(self.num_repeats).unsqueeze(1).repeat(1, 2)
        ) - self.num_repeats / 2  # [num_repeats, 2]
        modal_ids = torch.full((self.num_repeats, 1), fill_value=0)

        position_ids = torch.cat([position_ids, modal_ids], dim=-1)  # [num_repeats, 3]

        return position_ids.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_repeats, 3]

    def prepare_noise_image(
        self, height: int, width: int, batch_size: int
    ) -> torch.Tensor:
        noise = torch.randn(batch_size, self.in_channels, height, width)

        return noise

    def pixel_shuffle(
        self,
        patches: torch.Tensor,  # [batch_size, num_patches, patch_size * patch_size * in_channels]
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = patches.size(0)
        height_p = height // self.patch_size
        width_p = width // self.patch_size

        patches = patches.transpose(1, 2).view(
            batch_size,
            self.patch_size**2 * self.in_channels,
            height_p,
            width_p,
        )
        images = F.pixel_shuffle(patches, upscale_factor=self.patch_size)

        return images

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, in_channels, height, width]
        label_ids: torch.Tensor,  # [batch_size]
        timestep: torch.Tensor,  # [batch_size]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _in_channels, height, width = images.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            "Height and width must be divisible by patch size."
        )

        # prepare context
        patches = self.patch_embedding(images)
        num_patches = patches.size(1)

        label_embeddings = self.label_embedding(label_ids)
        time_embeddings = self.time_embedding(timestep)
        hidden_states = torch.cat(
            [
                patches,
                label_embeddings,
                time_embeddings,
            ],
            dim=1,
        )  # [batch_size, num_patches + num_repeats + num_repeats, hidden_dim]

        # prepare position ids
        patch_ids = self.prepare_patch_position_ids(height, width, batch_size).to(
            images.device
        )
        label_ids = self.prepare_label_position_ids(batch_size).to(images.device)
        time_ids = self.prepare_time_position_ids(batch_size).to(images.device)
        position_ids = torch.cat(
            [
                patch_ids,
                label_ids,
                time_ids,
            ],
            dim=1,
        )

        if attention_mask is not None:
            assert attention_mask.size(1) == hidden_states.size(1), (
                "Attention mask size must match the number of tokens."
            )

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = hidden_states + checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )  # type: ignore
            else:
                hidden_states = hidden_states + layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

        # select image patches
        patches = hidden_states[
            :, :num_patches, :
        ]  # [batch_size, num_patches, hidden_dim]
        output = self.final_layer(
            patches
        )  # [batch_size, num_patches, patch_size * patch_size * in_channels]
        images = self.pixel_shuffle(
            output, height, width
        )  # [batch_size, in_channels, height, width]

        return images

    @torch.inference_mode()
    def generate(
        self,
        label_ids: torch.Tensor,
        width: int = 128,
        height: int = 128,
        num_steps: int = 20,
        cfg_scale: float = 3.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> list[Image.Image]:
        assert width % self.patch_size == 0 and height % self.patch_size == 0, (
            "Width and height must be divisible by patch size."
        )
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        device = device if device is not None else next(self.parameters()).device

        batch_size = label_ids.size(0)
        do_cfg = cfg_scale > 1.0

        noisy_image = self.prepare_noise_image(height, width, batch_size).to(
            device,
            dtype,
        )
        timesteps = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,  # last is 1.0
            device=device,
            dtype=dtype,
        )

        for i in tqdm(range(num_steps), total=num_steps):
            timestep = timesteps[i]
            next_timestep = timesteps[i + 1]

            batch_timestep = timestep.expand(batch_size)  # [batch_size]
            batch_noisy_image = noisy_image
            batch_label_ids = label_ids

            if do_cfg:
                batch_noisy_image = batch_noisy_image.repeat(
                    2, 1, 1, 1
                )  # [batch_size * 2, in_channels, height, width]
                batch_label_ids = torch.cat(
                    [
                        batch_label_ids,
                        torch.full_like(
                            batch_label_ids,
                            self.label_embedding.uncond_id,
                        ),
                    ],
                    dim=0,
                )  # [batch_size * 2]
                batch_timestep = batch_timestep.repeat(2)  # [batch_size * 2]

            pred_image = self.forward(
                images=batch_noisy_image,
                label_ids=batch_label_ids,
                timestep=batch_timestep,
            )

            velocity_pred = image_pred_to_velocity_pred(
                pred_image=pred_image,
                noisy_image=batch_noisy_image,
                timestep=batch_timestep,
            )

            if do_cfg:
                velocity_cond, velocity_uncond = velocity_pred.chunk(2, dim=0)
                velocity_pred = velocity_uncond + cfg_scale * (
                    velocity_cond - velocity_uncond
                )

            noisy_image = noisy_image + velocity_pred * (next_timestep - timestep)

        images = tensor_to_pil(noisy_image)

        return images
