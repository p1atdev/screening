import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..modules.screening import GatedScreening, unit_length_norm


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
    ):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, in_channels, height, width]
    ) -> torch.Tensor:
        patches = self.in_proj(
            images
        )  # [batch_size, hidden_dim, height // patch_size, width // patch_size]
        patches = patches.flatten(2).transpose(
            1, 2
        )  # [batch_size, num_patches, hidden_dim]

        return unit_length_norm(patches)


class MultiScreenForImageClassification(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # phi^2
        num_heads: int,  # phi
        num_blocks: int,  # phi
        num_classes: int = 64,
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

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
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

        self.head = nn.Linear(hidden_dim, num_classes, bias=False)

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
            layer.set_trace_screening(value)

    def forward(
        self,
        image_feature: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedding(image_feature)

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

        logits = self.head(hidden_states)

        return logits
