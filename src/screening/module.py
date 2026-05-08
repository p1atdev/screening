import torch
import torch.nn as nn
import torch.nn.functional as F


def unit_length_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.scale = nn.Parameter(
            torch.ones(1) * embedding_dim**0.5,
            requires_grad=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.scale * unit_length_norm(self.embedding(input_ids))


class LMHead(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.scale = nn.Parameter(
            torch.ones(1) * embedding_dim**-0.5,
            requires_grad=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.scale * self.linear(hidden_states)
        return output


# MiPE is a RoPE-like rotation [18] applied only to the first two dimensions,
# with a rotation angle modulated by the learned screening window w.
def mipe_rotation(
    position_ids: torch.Tensor,
    window: torch.Tensor,
    window_threshold: float = 10.0,
) -> torch.Tensor:

    angle = (torch.cos(torch.pi * window / window_threshold) + 1) / 2

    rotation = torch.pi * position_ids * angle / window

    return rotation


def apply_mipe(
    sequence: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    position_ids: torch.Tensor,  # [batch_size, seq_len]
    window: torch.Tensor,
    window_threshold: float = 10.0,
) -> torch.Tensor:

    rotation = mipe_rotation(
        position_ids=position_ids,
        window=window,
        window_threshold=window_threshold,
    )  # [batch_size, seq_len]

    cos = torch.cos(rotation)
    sin = torch.sin(rotation)

    cos = cos[:, None, :]  # [batch_size, 1, seq_len]
    sin = sin[:, None, :]  # [batch_size, 1, seq_len]

    x1 = sequence[..., 0] * cos - sequence[..., 1] * sin
    x2 = sequence[..., 0] * sin + sequence[..., 1] * cos

    return torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1), sequence[..., 2:]], dim=-1)


def trim_similarity(
    similarity: torch.Tensor,
    acceptance: torch.Tensor,
) -> torch.Tensor:

    relevance = (
        torch.max(
            1 - (1 - similarity) / acceptance,
            torch.zeros_like(similarity),
        )
        ** 2
    )

    return relevance


def tanh_norm(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(p=2, dim=-1, keepdim=True)
    return torch.tanh(norm) * x / norm


def causal_softmask(
    position_ids: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    position_diff = position_ids[:, :, None] - position_ids[:, None, :]

    mask = torch.where(
        (-window < position_diff) & (position_diff <= 0),
        (torch.cos(torch.pi * position_diff / window) + 1) / 2,
        torch.ones_like(position_diff),
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

    similarity = query @ key.transpose(-2, -1)

    # Trim
    relevance = trim_similarity(similarity, acceptance)

    # Softmask
    softmask = causal_softmask(
        position_ids=position_ids,
        window=window,
    )  # [batch_size, seq_len, seq_len]
    softmask = softmask[:, None, :, :]  # [batch_size, 1, seq_len, seq_len]
    relevance = relevance * softmask

    # @
    screened = relevance @ value

    # TanhNorm
    screened = tanh_norm(screened)

    return screened


class GatedScreening(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_threshold: float = 10.0,
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
            torch.ones(1),
            requires_grad=True,
        )

        # s_r
        self._acceptance = nn.Parameter(
            torch.ones(1),
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

        self.scale = nn.Parameter(
            torch.ones(1) * hidden_dim**-0.5,
            requires_grad=True,
        )

    @property
    def window(self) -> torch.Tensor:
        return torch.exp(self._window_exponent) + 1

    @property
    def acceptance(self) -> torch.Tensor:
        return torch.sigmoid(self._acceptance)

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
        )
        screened = screened * torch.tanh(F.silu(gate))
        screened = self.post_screening_reshape(screened)

        return self.to_out(screened) * self.scale


class MultiScreen(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        vocab_size: int = 64,
        window_threshold: float = 10.0,
    ):
        super().__init__()

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
            embedding_dim=hidden_dim,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
            )

        logits = self.head(hidden_states)

        return logits
