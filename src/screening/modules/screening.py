import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def unit_length_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clip(min=eps)


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
    position_ids: torch.Tensor,
    #
    window: torch.Tensor,
    window_threshold: float,  # distance threshold for MiPE
    acceptance: torch.Tensor,  # similarity acceptance
    attention_mask: torch.Tensor | None = None,  # [batch_size, seq_len] mask
    is_causal: bool = True,
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
    if is_causal:
        softmask = causal_softmask(
            position_ids=position_ids,
            window=window,
        )  # [batch_size, num_heads, seq_len, seq_len]
        relevance = relevance * softmask

    # Optional attention mask (e.g., for padding tokens)
    if attention_mask is not None:
        mask = attention_mask.to(device=relevance.device, dtype=relevance.dtype)
        relevance = relevance * mask[:, None, None, :]

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
        num_layers: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, (
            "hidden_dim must be divisible by num_heads"
        )
        self.is_causal = is_causal

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
            torch.ones(1) * math.log(1 / math.sqrt(num_heads * num_layers)),
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
        position_ids: torch.Tensor,
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
            is_causal=self.is_causal,
        )
        # print(f"screened: {screened.shape}, gate: {gate.shape}")
        screened = screened * torch.tanh(F.silu(gate))
        screened = self.post_screening_reshape(screened)

        return self.to_out(screened) * self.scale
