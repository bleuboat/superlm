import torch
import torch.nn as nn

from torch import Tensor
from typing import overload

from superlm.tokenizer import Tokenizer
from superlm.config import ModelConfig, GenerationConfig


__all__ = [
    "Model",
    "auto_model",
    "register_model",
    "register_architecture",
]

MODEL_CLASSES: dict[str, type[nn.Module]] = {}
ARCHITECTURE_CLASSES: dict[str, type[Model]] = {}


def auto_model(
    tokenizer: Tokenizer,
    config: ModelConfig,
    generation_config: GenerationConfig,
) -> Model:
    return ARCHITECTURE_CLASSES[config.architecture](tokenizer, config, generation_config)


def register_model[T: type[nn.Module]](model_class: T) -> T:
    MODEL_CLASSES[model_class.__module__.split(".")[-1]] = model_class
    return model_class


def register_architecture[T: type[Model]](architecture_class: T) -> T:
    ARCHITECTURE_CLASSES[architecture_class.__name__.lower()] = architecture_class
    return architecture_class


class Model(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        super().__init__()

        self.config = config.copy()
        self.generation_config = generation_config.copy()

        self.config.vocab_size = tokenizer.vocab_size
        self.config.bos_token_ix = tokenizer.bos_token_ix
        self.config.eos_token_ix = tokenizer.eos_token_ix
        self.config.pad_token_ix = tokenizer.pad_token_ix
        self.generation_config.bos_token_ix = tokenizer.bos_token_ix
        self.generation_config.eos_token_ix = tokenizer.eos_token_ix
        self.generation_config.pad_token_ix = tokenizer.pad_token_ix

        self.model = MODEL_CLASSES[self.config.model_type](self.config)

    @overload
    def __call__(self, input_ids: Tensor) -> tuple[Tensor, None]:
        ...

    @overload
    def __call__(self, input_ids: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def __call__(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return self._wrapped_call_impl(input_ids, labels)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(p.device for p in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(p.dtype for p in self.parameters() if p.is_floating_point())
