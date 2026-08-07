import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import SwiGLU
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig
from ..generation import GenerationModule

__all__ = ['BoW']

class CausalBoW(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n = config.block_size
        self.register_buffer('bias', torch.tril(torch.ones(n, n)).view(1, n, n))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x
        return self.dropout(y)

class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.cbow = CausalBoW(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.ffnf = SwiGLU(config.n_embd, config.n_inner, config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.cbow(self.ln_1(x))
        x = x + self.ffnf(self.ln_2(x))
        return x

class BoW(GenerationModule):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(tokenizer, config, generation_config)
        config = self.config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.eps)
        if not self.config.tie_word_embeddings:
            self._lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def lm_head(self, input: Tensor) -> Tensor:
        if not self.config.tie_word_embeddings:
            return self._lm_head(input)
        return F.linear(input, self.wte.weight)

    def forward(self, idx: Tensor) -> Tensor:
        pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x