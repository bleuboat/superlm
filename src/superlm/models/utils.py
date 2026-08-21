import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch import Tensor
from typing import Any
from dataclasses import dataclass
from functools import partial

from superlm.config import ModelConfig


__all__ = [
    "register_model",
    "auto_model",
    "GradientCheckpointingLayer",
    "ModelOutput",
    "Model",
]

MODEL_CLASSES: dict[str, type[Model]] = {}


def register_model[T: type[Model]](model_class: T) -> T:
    MODEL_CLASSES[model_class.__module__.split(".")[-1]] = model_class
    return model_class


def auto_model(config: ModelConfig) -> Model:
    return MODEL_CLASSES[config.model_type](config)


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                partial(self._wrapped_call_impl, **kwargs),
                *args,
                use_reentrant=False,
            )
        return self._wrapped_call_impl(*args, **kwargs)


@dataclass
class ModelOutput:
    logits: Tensor | None = None
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class Model(nn.Module):
    wte: nn.Embedding

    def __call__(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> ModelOutput:
        return self._wrapped_call_impl(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )  # pyright: ignore
