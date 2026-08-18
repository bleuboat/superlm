import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections.abc import Callable
from superlm.activations import ACTIVATIONS
from superlm.config import ModelConfig
from superlm.utils import ModelOutput, Model, register_model


__all__ = ["SuperlmModel"]


class SuperlmRotaryEmbedding(nn.Module):
    inv_freq: Tensor
    __call__: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.n_embd // config.n_head
        base = config.rope_theta
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
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
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(n_batch, n_kv_head, n_rep, n_ctx, n_head_dim)
    return hidden_states.reshape(n_batch, n_kv_head * n_rep, n_ctx, n_head_dim)


class SuperlmAttention(nn.Module):
    __call__: Callable[[Tensor, tuple[Tensor, Tensor]], Tensor]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_head_dim = config.n_embd // config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_kv_group = config.n_head // config.n_kv_head
        self.dropout = config.dropout
        self.q_attn = nn.Linear(self.n_embd, self.n_head * self.n_head_dim, bias=config.bias)
        self.k_attn = nn.Linear(self.n_embd, self.n_kv_head * self.n_head_dim, bias=config.bias)
        self.v_attn = nn.Linear(self.n_embd, self.n_kv_head * self.n_head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.n_head_dim, self.n_embd, bias=config.bias)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> Tensor:
        b, t, c = x.size()

        q = self.q_attn(x).view(b, t, self.n_head, self.n_head_dim).transpose(1, 2)
        k = self.k_attn(x).view(b, t, self.n_kv_head, self.n_head_dim).transpose(1, 2)
        v = self.v_attn(x).view(b, t, self.n_kv_head, self.n_head_dim).transpose(1, 2)

        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_kv_group), repeat_kv(v, self.n_kv_group)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.o_proj(y)


class SuperlmMLP(nn.Module):
    __call__: Callable[[Tensor], Tensor]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.act = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class SuperlmLayer(nn.Module):
    __call__: Callable[[Tensor, tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.attn = SuperlmAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.ffnf = SuperlmMLP(config)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        attn = self.attn(self.ln_1(x), pos_emb)
        x = x + attn
        x = x + self.ffnf(self.ln_2(x))
        return x, attn


@register_model
class SuperlmModel(Model):
    wte: nn.Embedding
    wpe: SuperlmRotaryEmbedding
    layers: nn.ModuleList[SuperlmLayer]
    ln_f: nn.RMSNorm

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)
        self.wpe = SuperlmRotaryEmbedding(config)
        self.layers = nn.ModuleList([SuperlmLayer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.eps)

    def forward(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        pos = torch.arange(input_ids.size()[1], device=input_ids.device).unsqueeze(0)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(input_ids, pos)
        hidden_state = tok_emb

        all_hidden_states: list[Tensor] = []
        all_attentions: list[Tensor] = []
        hidden_states = None
        attentions = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_state)
            hidden_state, attention = layer(hidden_state, pos_emb)
            if output_attentions:
                all_attentions.append(attention)

        if output_hidden_states:
            all_hidden_states.append(hidden_state)
            hidden_states = tuple(all_hidden_states)

        if output_attentions:
            attentions = tuple(all_attentions)

        logits = self.ln_f(hidden_state)
        return ModelOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
