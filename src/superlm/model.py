import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from .tokenizer import Tokenizer
from .config import ModelConfig, GenerationConfig
from .generation import GenerationModule

__all__ = ['Transformer'] 

class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].to(position_ids.device).expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up   = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.ffnf = SwiGLU(config)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(self.ln_1(x), pos_emb)
        x = x + self.ffnf(self.ln_2(x))
        return x

class Transformer(GenerationModule):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(tokenizer, config, generation_config)
        config = self.config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = RotaryEmbedding(config)
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
        x = tok_emb
        for block in self.h:
            x = block(x, pos_emb)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x