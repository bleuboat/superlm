import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['SwiGLU']

class SwiGLU(nn.Module):
    def __init__(self, n_in: int, n_inner: int, n_out: int, bias: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(n_in, n_inner, bias)
        self.up   = nn.Linear(n_in, n_inner, bias)
        self.down = nn.Linear(n_inner, n_out, bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))