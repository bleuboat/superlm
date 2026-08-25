import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

from torch import Tensor
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from superlm.config import ModelConfig


__all__ = ["GradientCheckpointingLayer", "Embedding", "ModelOutput", "Model"]

MODEL_CLASSES: dict[str, type[Model]] = {}


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                partial(self._wrapped_call_impl, **kwargs),
                *args,
                use_reentrant=False,
            )  # pyright: ignore[reportUnknownVariableType]
        return self._wrapped_call_impl(*args, **kwargs)


class Embedding(nn.Module):
    __call__: Callable[[Tensor], Tensor]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.pad_token_ix = config.pad_token_ix
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.n_embd)))

        with torch.no_grad():
            self.weight.normal_()
            self.weight[self.pad_token_ix].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        weight = torch.cat((self.weight, self.weight[:-1].mean(0, keepdim=True)))
        return F.embedding(input, weight, self.pad_token_ix)


@dataclass
class ModelOutput:
    logits: Tensor | None = None
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


class Model(nn.Module):
    wte: Embedding

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
        )

    @staticmethod
    def auto(config: ModelConfig) -> Model:
        return MODEL_CLASSES[config.model_type](config)

    def __init_subclass__(cls) -> None:
        MODEL_CLASSES[cls.__module__.split(".")[-1]] = cls
