from torch import Tensor
from superlm.config import ModelConfig
from .utils import *


__all__ = ["Bigram"]


@register_model
class Bigram(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.wte = Embedding(config)

    def forward(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        logits = self.wte(input_ids)
        return ModelOutput(logits=logits)
