import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections.abc import Callable
from superlm.activations import ACTIVATIONS
from superlm.config import ModelConfig
from .utils import *


class CausalBoW(nn.Module):
    bias: Tensor
    __call__: Callable[[Tensor], Tensor]

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
    __call__: Callable[[Tensor], Tensor]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.up = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.down = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.act = ACTIVATIONS[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class Layer(nn.Module):
    __call__: Callable[[Tensor], Tensor]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, config.rms_norm_eps)
        self.cbow = CausalBoW(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, config.rms_norm_eps)
        self.ffnf = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.cbow(self.ln_1(x))
        return x + self.ffnf(self.ln_2(x))


@register_model
class BoW(Model):
    wte: Embedding
    wpe: nn.Embedding
    layers: nn.ModuleList[Layer]
    ln_f: nn.RMSNorm

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = Embedding(config)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        pos = torch.arange(input_ids.size()[1], device=input_ids.device).unsqueeze(0)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        hidden_state = tok_emb + pos_emb

        all_hidden_states: list[Tensor] = []
        hidden_states = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_state)
            hidden_state = layer(hidden_state)

        if output_hidden_states:
            all_hidden_states.append(hidden_state)
            hidden_states = tuple(all_hidden_states)

        logits = self.ln_f(hidden_state)
        return ModelOutput(
            logits=logits,
            hidden_states=hidden_states,
        )
