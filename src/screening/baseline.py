import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def pre_attn_reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        return x

    def post_attn_reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = (
            x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )  # (batch_size, seq_len, hidden_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.pre_attn_reshape(self.to_q(x))
        k = self.pre_attn_reshape(self.to_k(x))
        v = self.pre_attn_reshape(self.to_v(x))
        gate = self.pre_attn_reshape(self.to_gate(x))

        # TODO: RoPE

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )  # (batch_size, num_heads, seq_len, head_dim)
        gate = torch.sigmoid(gate)
        attn = attn * gate

        out = self.post_attn_reshape(attn)
        out = self.to_out(out)

        return out


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, multiple_of: int = 64):
        super().__init__()

        intermediate_dim = 4 * hidden_dim
        intermediate_dim = int(2 * intermediate_dim / 3)
        intermediate_dim = multiple_of * (
            (intermediate_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, multiple_of: int = 64):
        super().__init__()

        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads)
        self.mlp_norm = nn.RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, multiple_of)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        multiple_of: int = 64,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, multiple_of)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.gradient_checkpointing = False

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        logits = self.head(x)

        return logits
