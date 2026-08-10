import torch.nn as nn
from torch import Tensor
from ..config import ModelConfig


__all__ = ["Bigram"]


class Bigram(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.logits = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)

    def forward(self, idx: Tensor) -> Tensor:
        return self.logits(idx)
