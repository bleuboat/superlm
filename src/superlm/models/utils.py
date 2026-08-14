import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import cast, overload
from superlm.generation import GenerationMixin
from superlm.tokenizer import Tokenizer
from superlm.config import ModelConfig, GenerationConfig


__all__ = [
    "auto_model",
    "register_model",
    "Model",
    "CausalLM",
    "SequenceClassification",
    "QuestionAnswering",
    "TokenClassification",
]

MODEL_CLASSES: dict[str, type[nn.Module]] = {}


def auto_model(config: ModelConfig) -> nn.Module:
    model_class = MODEL_CLASSES[config.model_type]
    return model_class(config)


def register_model(model_class: type[nn.Module]) -> type[nn.Module]:
    MODEL_CLASSES[model_class.__module__.split(".")[-1]] = model_class
    return model_class


class Model(nn.Module):
    @overload
    def __call__(self, input_ids: Tensor) -> tuple[Tensor, None]:
        ...

    @overload
    def __call__(self, input_ids: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def __call__(self, input_ids: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        return self._wrapped_call_impl(input_ids, labels) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(p.device for p in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(p.dtype for p in self.parameters() if p.is_floating_point())


class CausalLM(Model, GenerationMixin):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(tokenizer, config, generation_config)
        self.model = auto_model(self.config)
        if not self.config.tie_word_embeddings:
            self._lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()

    def lm_head(self, input: Tensor) -> Tensor:
        if not self.config.tie_word_embeddings:
            return self._lm_head(input)
        wte = cast(nn.Embedding, self.model.wte)
        return F.linear(input, wte.weight)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self.model(input_ids)
        x = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.loss_function(x.view(-1, self.config.vocab_size), labels.view(-1))
        return x, loss


# TODO: Unsupported models
class SequenceClassification(Model):
    pass


class QuestionAnswering(Model):
    pass


class TokenClassification(Model):
    pass
