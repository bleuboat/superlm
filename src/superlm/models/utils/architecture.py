import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import cast

from superlm.generation import GenerationMixin
from superlm.tokenizer import Tokenizer
from superlm.config import ModelConfig, GenerationConfig

from .model import Model, register_architecture
from .loss import (
    CausalLMLoss,
    SequenceClassificationLoss,
    TokenClassificationLoss,
)


__all__ = [
    "CausalLM",
    "SequenceClassification",
    "TokenClassification",
]


@register_architecture
class CausalLM(Model, GenerationMixin):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        super().__init__(tokenizer, config, generation_config)
        if not self.config.tie_word_embeddings:
            self._lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.loss_function = CausalLMLoss(self.config)

    def lm_head(self, input: Tensor) -> Tensor:
        if not self.config.tie_word_embeddings:
            return self._lm_head(input)
        wte = cast(nn.Embedding, self.model.wte)
        return F.linear(input, wte.weight)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return logits, loss


@register_architecture
class SequenceClassification(Model):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        super().__init__(tokenizer, config, generation_config)
        self.score = nn.Linear(self.config.n_embd, self.config.num_labels, bias=False)
        self.loss_function = SequenceClassificationLoss(self.config)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.model(input_ids)
        logits = self.score(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return logits, loss


@register_architecture
class TokenClassification(Model):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        super().__init__(tokenizer, config, generation_config)
        self.score = nn.Linear(self.config.n_embd, self.config.num_labels, bias=False)
        self.loss_function = TokenClassificationLoss(self.config)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.model(input_ids)
        logits = self.score(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return logits, loss
