import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from .utils import Model
from ..activations import ACTIVATIONS
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig
from ..generation import GenerationModel

class SuperLM_RotaryEmbedding(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().to(position_ids.device).expand(position_ids.shape[0], -1, 1)
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

class SuperLM_Attention(Module):
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

        q = self.q_attn(x).view(B, T, self.n_head,    self.n_head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_attn(x).view(B, T, self.n_kv_head, self.n_head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_attn(x).view(B, T, self.n_kv_head, self.n_head_dim).transpose(1, 2) # (B, nh, T, hs)

        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_kv_group), repeat_kv(v, self.n_kv_group)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.o_proj(y)
        return y

class SuperLM_MLP(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up   = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.act  = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))

class SuperLM_Layer(Module):
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

class SuperLM_Model(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)
        self.wpe = SuperLM_RotaryEmbedding(config)
        self.h = nn.ModuleList([SuperLM_Layer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.eps)

    def forward(self, idx: Tensor) -> Tensor:
        pos = torch.arange(0, idx.size()[1], device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(idx, pos)
        x = tok_emb
        for SuperLM_Layer in self.h:
            x = SuperLM_Layer(x, pos_emb)
        x = self.ln_f(x)
        return x

@Model.register_model('CausalLM')
class SuperLM_ForCausalLM(GenerationModel):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(tokenizer, config, generation_config)
        self.model = SuperLM_Model(self.config)
        if not self.config.tie_word_embeddings:
            self._lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()

    def lm_head(self, input: Tensor) -> Tensor:
        if not self.config.tie_word_embeddings:
            return self._lm_head(input)
        return F.linear(input, self.model.wte.weight)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self.model(input_ids)
        x = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.loss_function(x.view(-1, self.config.vocab_size), labels.view(-1))
        return x, loss

@Model.register_model('SequenceClassification')
class SuperLM_ForSequenceClassification(Model):
    pass

@Model.register_model('QuestionAnswering')
class SuperLM_ForQuestionAnswering(Model):
    pass

@Model.register_model('TokenClassification')
class SuperLM_ForTokenClassification(Model):
    pass

__all__ = [
    'SuperLM_Model',
    'SuperLM_ForCausalLM',
    'SuperLM_ForSequenceClassification',
    'SuperLM_ForQuestionAnswering',
    'SuperLM_ForTokenClassification',
]