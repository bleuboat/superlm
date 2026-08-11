import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from superlm.activations import ACTIVATIONS
from superlm.config import ModelConfig
from .utils import register_model


__all__ = ["BoW"]


class CausalBoW(nn.Module):
    bias: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(n, n)).view(1, n, n))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.size()
        att = torch.zeros((b, t, t), device=x.device)
        att = att.masked_fill(self.bias[:, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ x
        return self.dropout(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up   = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.act  = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.cbow = CausalBoW(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.ffnf = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.cbow(self.ln_1(x))
        x = x + self.ffnf(self.ln_2(x))
        return x


@register_model
class BoW(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.eps)

    def forward(self, idx: Tensor) -> Tensor:
        pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x
