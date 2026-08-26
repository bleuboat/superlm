from typing import cast

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from superlm.config import GenerationConfig, ModelConfig
from superlm.generation import GenerationMixin
from superlm.models import ModelOutput

from .utils import Architecture


class CausalLMLoss:
    def __init__(self, config: ModelConfig) -> None:
        self.n_embd = config.n_embd
        self.ignore_index = config.pad_token_ix if config.pad_token_ix is not None else -100

    def __call__(self, logits: Tensor, weight: Tensor, labels: Tensor) -> Tensor:
        logits = logits.view(-1, self.n_embd)
        labels = labels.view(-1)
        return F.linear_cross_entropy(logits, weight, labels, ignore_index=self.ignore_index)


class CausalLM(Architecture, GenerationMixin):
    def __init__(self, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(config, generation_config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.loss_function = CausalLMLoss(config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.wte.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        logits_to_keep: int = 0,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if logits_to_keep == 1:
            slice_indices = -1
        elif logits_to_keep == 0:
            slice_indices = slice(None)
        else:
            slice_indices = slice(-logits_to_keep, None)

        output_logits = cast(Tensor, outputs.logits)[:, slice_indices, :]

        if labels is None:
            loss = None
            logits = self.lm_head(output_logits)
        else:
            loss = self.loss_function(output_logits, self.lm_head.weight, labels)
            logits = None

        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
