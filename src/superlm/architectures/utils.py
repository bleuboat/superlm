import torch
import torch.nn as nn
from torch import Tensor

from superlm.tokenizer import Tokenizer
from superlm.config import ModelConfig, GenerationConfig
from superlm.models import ModelOutput, auto_model


__all__ = [
    "register_architecture",
    "auto_architecture",
    "Architecture",
]


ARCHITECTURE_CLASSES: dict[str, type[Architecture]] = {}


def register_architecture[T: type[Architecture]](architecture_class: T) -> T:
    ARCHITECTURE_CLASSES[architecture_class.__name__.lower()] = architecture_class
    return architecture_class


def auto_architecture(
    tokenizer: Tokenizer,
    config: ModelConfig,
    generation_config: GenerationConfig,
) -> Architecture:
    return ARCHITECTURE_CLASSES[config.architecture](tokenizer, config, generation_config)


class Architecture(nn.Module):
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

        self.model = auto_model(self.config)

        if self.config.gradient_checkpointing:
            for module in self.model.modules():
                module.gradient_checkpointing = True

    def __call__(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        return self._wrapped_call_impl(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )  # pyright: ignore

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(p.device for p in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(p.dtype for p in self.parameters() if p.is_floating_point())
