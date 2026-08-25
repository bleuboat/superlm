import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import cast
from superlm.config import ModelConfig, GenerationConfig
from superlm.models import ModelOutput
from .utils import Architecture


class TokenClassificationLoss:
    def __init__(self, config: ModelConfig) -> None:
        self.num_labels = config.num_labels
        self.ignore_index = config.pad_token_ix if config.pad_token_ix is not None else -100

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)


class TokenClassification(Architecture):
    def __init__(self, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(config, generation_config)
        self.score = nn.Linear(config.n_embd, config.num_labels, bias=False)
        self.loss_function = TokenClassificationLoss(config)

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
        output_logits = cast(Tensor, outputs.logits)
        logits = self.score(output_logits)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        return ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
