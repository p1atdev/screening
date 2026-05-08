import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def unit_length_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clip(min=eps)


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


# MiPE is a RoPE-like rotation [18] applied only to the first two dimensions,
# with a rotation angle modulated by the learned screening window w.
def mipe_rotation(
    position_ids: torch.Tensor,  # [batch_size, seq_len]
    window: torch.Tensor,  # [num_heads]
    window_threshold: float = 256.0,
) -> torch.Tensor:
    batch_size, seq_len = position_ids.size()
    num_heads = window.size(0)
    window = window[None, :, None].repeat(
        batch_size, 1, seq_len
    )  # [batch_size, num_heads, seq_len]

    # gamma(w)
    gamma = torch.where(
        window < window_threshold,
        (torch.cos(torch.pi * window / window_threshold) + 1) / 2,
        torch.zeros_like(window),
    )

    position_ids = position_ids[:, None, :].repeat(
        1, num_heads, 1
    )  # [batch_size, num_heads, seq_len]

    rotation = torch.pi * position_ids * gamma / window

    return rotation


def apply_mipe(
    sequence: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    position_ids: torch.Tensor,  # [batch_size, seq_len]
    window: torch.Tensor,
    window_threshold: float = 256.0,
) -> torch.Tensor:

    rotation = mipe_rotation(
        position_ids=position_ids,
        window=window,
        window_threshold=window_threshold,
    )  # [batch_size, num_heads, seq_len]

    cos = torch.cos(rotation)
    sin = torch.sin(rotation)

    x1 = sequence[..., 0] * cos - sequence[..., 1] * sin
    x2 = sequence[..., 0] * sin + sequence[..., 1] * cos

    return torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1), sequence[..., 2:]], dim=-1)


def trim_similarity(
    similarity: torch.Tensor,  # [batch_size, num_heads, seq_len, seq_len]
    acceptance: torch.Tensor,  # [num_heads]
) -> torch.Tensor:

    relevance = (
        torch.max(
            1 - (1 - similarity) / acceptance[None, :, None, None],
            torch.zeros_like(similarity),
        )
        ** 2
    )

    return relevance


def tanh_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = x.norm(p=2, dim=-1, keepdim=True)
    scale = torch.where(
        norm > eps,
        torch.tanh(norm) / norm.clamp_min(eps),
        torch.ones_like(norm),
    )
    return scale * x


def causal_softmask(
    position_ids: torch.Tensor,  # [batch_size, seq_len]
    window: torch.Tensor,  # [num_heads]
) -> torch.Tensor:
    position_diff = position_ids[:, None, :] - position_ids[:, :, None]
    # [batch_size, seq_len, seq_len]
    position_diff = position_diff[:, None, :, :].repeat(
        1, window.size(0), 1, 1
    )  # [batch_size, num_heads, seq_len, seq_len]

    window = window[None, :, None, None]

    mask = torch.where(
        (-window < position_diff) & (position_diff <= 0),
        (torch.cos(torch.pi * position_diff / window) + 1) / 2,
        torch.zeros_like(position_diff),
    )

    return mask


def screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.LongTensor,
    #
    window: torch.Tensor,
    window_threshold: float,  # distance threshold for MiPE
    acceptance: torch.Tensor,  # similarity acceptance
    attention_mask: torch.Tensor | None = None,  # [batch_size, seq_len] mask
) -> torch.Tensor:
    query = unit_length_norm(query)
    key = unit_length_norm(key)
    value = unit_length_norm(value)

    # MiPE
    query = apply_mipe(
        sequence=query,
        position_ids=position_ids,
        window=window,
        window_threshold=window_threshold,
    )
    key = apply_mipe(
        sequence=key,
        position_ids=position_ids,
        window=window,
        window_threshold=window_threshold,
    )

    similarity = query @ key.transpose(-2, -1)

    # Trim
    relevance = trim_similarity(similarity, acceptance)

    # Softmask
    softmask = causal_softmask(
        position_ids=position_ids,
        window=window,
    )  # [batch_size, num_heads, seq_len, seq_len]
    relevance = relevance * softmask

    # Optional attention mask (e.g., for padding tokens)
    if attention_mask is not None:
        relevance = relevance * attention_mask[:, None, None, :]

    # @
    screened = relevance @ value

    # TanhNorm
    screened = tanh_norm(screened)

    return screened


class GatedScreening(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # phi
        num_heads: int,  # phi
        window_threshold: float = 256.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, (
            "hidden_dim must be divisible by num_heads"
        )

        # s_w
        self._window_exponent = nn.Parameter(
            torch.linspace(0, math.log(window_threshold), num_heads),
            requires_grad=True,
        )

        # s_r
        self._acceptance = nn.Parameter(
            torch.zeros(num_heads),
            requires_grad=True,
        )

        self.window_threshold = window_threshold

        # qkvgo
        self.to_q = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.to_k = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.to_v = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.to_gate = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.to_out = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        # s_O
        self._scale = nn.Parameter(
            torch.ones(1) * math.log(hidden_dim**-0.5),
            requires_grad=True,
        )

    @property
    def window(self) -> torch.Tensor:
        return torch.exp(self._window_exponent) + 1

    @property
    def acceptance(self) -> torch.Tensor:
        return torch.sigmoid(self._acceptance)

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self._scale)

    def pre_screening_reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def post_screening_reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _num_heads, seq_len, _head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q = self.pre_screening_reshape(self.to_q(hidden_states))
        k = self.pre_screening_reshape(self.to_k(hidden_states))
        v = self.pre_screening_reshape(self.to_v(hidden_states))
        gate = self.pre_screening_reshape(self.to_gate(hidden_states))

        screened = screening(
            query=q,
            key=k,
            value=v,
            position_ids=position_ids,
            window=self.window,
            window_threshold=self.window_threshold,
            acceptance=self.acceptance,
            attention_mask=attention_mask,
        )
        # print(f"screened: {screened.shape}, gate: {gate.shape}")
        screened = screened * torch.tanh(F.silu(gate))
        screened = self.post_screening_reshape(screened)

        return self.to_out(screened) * self.scale


class MultiScreen(nn.Module):
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
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = LMHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )

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

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)

        for layer in self.layers:
            hidden_states = hidden_states + layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        logits = self.head(hidden_states)

        return logits
