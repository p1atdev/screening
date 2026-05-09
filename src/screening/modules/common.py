import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        multiple_of: int = 64,
    ):
        super().__init__()

        intermediate_dim = 4 * out_features
        intermediate_dim = int(2 * intermediate_dim / 3)
        intermediate_dim = multiple_of * (
            (intermediate_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(in_features, intermediate_dim, bias=False)
        self.w2 = nn.Linear(in_features, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
