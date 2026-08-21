import torch.nn as nn
from torch import Tensor
from superlm.config import ModelConfig
from .utils import ModelOutput, Model, register_model


__all__ = ["Bigram"]


@register_model
class Bigram(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_ix)

    def forward(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        logits = self.wte(input_ids)
        return ModelOutput(logits=logits)
