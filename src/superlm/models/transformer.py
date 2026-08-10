import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..activations import ACTIVATIONS
from ..config import ModelConfig


__all__ = ["SuperLM_Model"]


class SuperLM_RotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
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


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    n_batch, n_kv_head, n_ctx, n_head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(n_batch, n_kv_head, n_rep, n_ctx, n_head_dim)
    return hidden_states.reshape(n_batch, n_kv_head * n_rep, n_ctx, n_head_dim)


class SuperLM_Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_head_dim = config.n_embd // config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_kv_group = config.n_head // config.n_kv_head
        self.dropout = config.dropout
        self.q_attn = nn.Linear(self.n_embd, self.n_head    * self.n_head_dim, bias=True)
        self.k_attn = nn.Linear(self.n_embd, self.n_kv_head * self.n_head_dim, bias=True)
        self.v_attn = nn.Linear(self.n_embd, self.n_kv_head * self.n_head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_head * self.n_head_dim, self.n_embd, bias=False)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> Tensor:
        B, T, C = x.size()

        q = self.q_attn(x).view(B, T, self.n_head,    self.n_head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_kv_head, self.n_head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_kv_head, self.n_head_dim).transpose(1, 2)

        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_kv_group), repeat_kv(v, self.n_kv_group)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.o_proj(y)
        return y


class SuperLM_MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up   = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.act  = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class SuperLM_Layer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.attn = SuperLM_Attention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.ffnf = SuperLM_MLP(config)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(self.ln_1(x), pos_emb)
        x = x + self.ffnf(self.ln_2(x))
        return x


class SuperLM_Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)
        self.wpe = SuperLM_RotaryEmbedding(config)
        self.layers = nn.ModuleList([SuperLM_Layer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.eps)

    def forward(self, idx: Tensor) -> Tensor:
        pos = torch.arange(0, idx.size()[1], device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(idx, pos)
        x = tok_emb
        for layer in self.layers:
            x = layer(x, pos_emb)
        x = self.ln_f(x)
        return x
