import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..modules.screening import GatedScreening, unit_length_norm


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        # s_E
        self._scale = nn.Parameter(
            torch.zeros(1),
            requires_grad=True,
        )

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self._scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.scale * unit_length_norm(self.embedding(input_ids))


class LMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

        # s_F
        self._scale = nn.Parameter(
            torch.ones(1) * math.log(hidden_dim**0.5),
            requires_grad=True,
        )

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self._scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.scale * F.linear(
            hidden_states,
            weight=unit_length_norm(self.linear.weight),
            bias=self.linear.bias,
        )

        return output


class MultiScreenForCausalLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # phi^2
        num_heads: int,  # phi
        num_blocks: int,  # phi
        vocab_size: int = 64,
        window_threshold: float = 256.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_blocks = num_blocks
        self.window_threshold = window_threshold

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
        )

        self.layers = nn.ModuleList(
            [
                GatedScreening(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    window_threshold=window_threshold,
                    num_layers=num_blocks,
                    is_causal=True,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = LMHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = hidden_states + checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = hidden_states + layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

        logits = self.head(hidden_states)

        return logits
