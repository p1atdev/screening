import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import flash_screening as FS

from .common import SwiGLU


def unit_length_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clip(min=eps)


# MiPE is a RoPE-like rotation [18] applied to one feature pair per position
# axis, with a rotation angle modulated by the learned screening window w.
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


def compute_freqs_cis(
    position_ids: torch.Tensor,  # [batch_size, seq_len, num_axes]
    window: torch.Tensor,  # [num_heads]
    window_threshold: float = 256.0,
) -> torch.Tensor:
    freqs_cis = []  # [batch_size, num_heads, seq_len, 2*num_axes]

    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(-1)  # [batch_size, seq_len, 1]

    for axis in range(position_ids.size(-1)):
        rotation = mipe_rotation(
            position_ids=position_ids[..., axis],
            window=window,
            window_threshold=window_threshold,
        )  # [batch_size, num_heads, seq_len]

        freqs_cis.append(
            torch.stack([torch.cos(rotation), torch.sin(rotation)], dim=-1)
        )
        # [batch_size, num_heads, seq_len, 2]

    return torch.cat(freqs_cis, dim=-1)


def apply_mipe(
    sequence: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    freqs_cis: torch.Tensor,  # [batch_size, num_heads, seq_len, 2 (cos, sin) * num_axes]
) -> torch.Tensor:
    _, _, _, encoded_dim = freqs_cis.size()

    assert encoded_dim % 2 == 0, "encoded_dim must be even"
    assert sequence.size(-1) >= encoded_dim, "head_dim must be >= encoded_dim"

    x_even = sequence[..., :encoded_dim:2]
    x_odd = sequence[..., 1:encoded_dim:2]

    cos = freqs_cis[..., :encoded_dim:2]
    sin = freqs_cis[..., 1:encoded_dim:2]

    x1 = x_even * cos - x_odd * sin
    x2 = x_even * sin + x_odd * cos

    rotated = torch.stack((x1, x2), dim=-1).flatten(-2)

    return torch.cat(
        [rotated, sequence[..., encoded_dim:]],
        dim=-1,
    )


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
    position_ids: torch.Tensor,  # [batch_size, seq_len] or [batch_size, seq_len, 1]
    window: torch.Tensor,  # [num_heads]
) -> torch.Tensor:
    if position_ids.ndim == 3 and position_ids.size(-1) == 1:
        position_ids = position_ids.squeeze(-1)

    assert position_ids.ndim == 2, (
        "Causal softmask only supports position ids shaped [batch_size, seq_len] "
        "or [batch_size, seq_len, 1]"
    )

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
    #
    acceptance: torch.Tensor,  # similarity acceptance
    window: torch.Tensor,
    #
    position_ids: torch.Tensor | None = None,  # [batch_size, seq_len, num_axes]
    attention_mask: torch.Tensor | None = None,  # [batch_size, seq_len] mask
    is_causal: bool = True,
    return_score: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    similarity = query @ key.transpose(-2, -1)

    # Trim
    relevance = trim_similarity(similarity, acceptance)

    # Softmask
    if is_causal:
        softmask_position_ids = position_ids
        if softmask_position_ids is None:
            batch_size, _, seq_len, _ = query.size()
            softmask_position_ids = torch.arange(
                seq_len,
                device=query.device,
            )[None, :].expand(batch_size, -1)

        softmask = causal_softmask(
            position_ids=softmask_position_ids,
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

    if return_score:
        return screened, relevance

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

        self.screening_trace: dict[str, torch.Tensor] = {}
        self.trace_screening = False

    @property
    def window(self) -> torch.Tensor:
        return torch.exp(self._window_exponent) + 1

    @property
    def acceptance(self) -> torch.Tensor:
        return torch.sigmoid(self._acceptance)

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self._scale)

    def set_trace_screening(self, value: bool):
        self.trace_screening = value
        self.screening_trace.clear()

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
        **kwargs,
    ) -> torch.Tensor:

        q = self.pre_screening_reshape(self.to_q(hidden_states))
        k = self.pre_screening_reshape(self.to_k(hidden_states))
        v = self.pre_screening_reshape(self.to_v(hidden_states))
        gate = self.pre_screening_reshape(self.to_gate(hidden_states))

        # RSS
        q = unit_length_norm(q)
        k = unit_length_norm(k)
        v = unit_length_norm(v)

        # MiPE
        if position_ids is not None:
            freqs_cis = compute_freqs_cis(
                position_ids=position_ids,
                window=self.window,
                window_threshold=self.window_threshold,
            )
            q = apply_mipe(q, freqs_cis)
            k = apply_mipe(k, freqs_cis)

        screening_output = FS.flash_screening(
            query=q,
            key=k,
            value=v,
            position_ids=position_ids,
            window=self.window,
            acceptance=self.acceptance,
            attention_mask=attention_mask,
            is_causal=self.is_causal,
            return_score=self.trace_screening,
        )
        if self.trace_screening:
            screened, screening_score = screening_output
            with torch.no_grad():
                self.screening_trace = {
                    "score": screening_score.detach().cpu(),
                    "before_gate": screened.detach().cpu(),
                }
        else:
            screened = screening_output
            self.screening_trace.clear()

        # print(f"screened: {screened.shape}, gate: {gate.shape}")
        screened = screened * torch.tanh(F.silu(gate))

        if self.trace_screening:
            with torch.no_grad():
                self.screening_trace["after_gate"] = screened.detach().cpu()

        screened = self.post_screening_reshape(screened)

        return self.to_out(screened) * self.scale


class GatedCrossScreening(GatedScreening):
    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        position_ids: torch.Tensor,
        context_position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        q = self.pre_screening_reshape(self.to_q(hidden_states))
        k = self.pre_screening_reshape(self.to_k(context_states))
        v = self.pre_screening_reshape(self.to_v(context_states))
        gate = self.pre_screening_reshape(self.to_gate(hidden_states))

        # RSS
        q = unit_length_norm(q)
        k = unit_length_norm(k)
        v = unit_length_norm(v)

        # MiPE
        if position_ids is not None and context_position_ids is not None:
            freqs_cis = compute_freqs_cis(
                position_ids=position_ids,
                window=self.window,
                window_threshold=self.window_threshold,
            )
            q = apply_mipe(q, freqs_cis)

            context_freqs_cis = compute_freqs_cis(
                position_ids=context_position_ids,
                window=self.window,
                window_threshold=self.window_threshold,
            )
            k = apply_mipe(k, context_freqs_cis)

        screening_output = FS.flash_screening(
            query=q,
            key=k,
            value=v,
            position_ids=position_ids,
            window=self.window,
            acceptance=self.acceptance,
            attention_mask=attention_mask,
            is_causal=False,  # Cross-attention is typically non-causal
            return_score=self.trace_screening,
        )
        if self.trace_screening:
            screened, screening_score = screening_output
            with torch.no_grad():
                self.screening_trace = {
                    "score": screening_score.detach().cpu(),
                    "before_gate": screened.detach().cpu(),
                }
        else:
            screened = screening_output
            self.screening_trace.clear()

        screened = screened * torch.tanh(F.silu(gate))

        if self.trace_screening:
            with torch.no_grad():
                self.screening_trace["after_gate"] = screened.detach().cpu()

        screened = self.post_screening_reshape(screened)

        return self.to_out(screened) * self.scale


class MultiScreenBlock(nn.Module):
    # GatedScreening + SwiGLU
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_threshold: float = 256.0,
        num_layers: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()

        self.screening = GatedScreening(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_threshold=window_threshold,
            num_layers=num_layers,
            is_causal=is_causal,
        )

        self.mlp = SwiGLU(
            hidden_dim,
            hidden_dim,
        )

    def set_trace_screening(self, value: bool) -> None:
        self.screening.set_trace_screening(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.screening(
            hidden_states=unit_length_norm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(unit_length_norm(hidden_states))

        return hidden_states
