import torch.nn as nn
from torch import Tensor
from typing import cast

from superlm.generation import GenerationArchitecture
from superlm.tokenizer import Tokenizer
from superlm.config import ModelConfig, GenerationConfig

from .model import Architecture, ModelOutput, register_architecture
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
class CausalLM(GenerationArchitecture):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        super().__init__(tokenizer, config, generation_config)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.loss_function = CausalLMLoss(self.config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.wte.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = cast(Tensor, outputs.logits)
        if labels is None:
            loss = None
            logits = self.lm_head(logits)
        else:
            loss = self.loss_function(logits, self.lm_head.weight, labels)
            logits = None
        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_architecture
class SequenceClassification(Architecture):
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
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = cast(Tensor, outputs.logits)
        logits = self.score(logits)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_architecture
class TokenClassification(Architecture):
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
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = cast(Tensor, outputs.logits)
        logits = self.score(logits)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
