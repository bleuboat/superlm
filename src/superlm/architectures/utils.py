import torch
import torch.nn as nn
from torch import Tensor

from superlm.config import GenerationConfig, ModelConfig
from superlm.models import GradientCheckpointingLayer, Model, ModelOutput

__all__ = ["Architecture"]

ARCHITECTURE_CLASSES: dict[str, type[Architecture]] = {}


class Architecture(nn.Module):
    def __init__(self, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__()

        self.config = config
        self.generation_config = generation_config
        self.model = Model.auto(self.config)

        if self.config.gradient_checkpointing:
            for module in self.model.modules():
                if isinstance(module, GradientCheckpointingLayer):
                    module.gradient_checkpointing = True

    def __call__(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        logits_to_keep: int = 0,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ModelOutput:
        return self._wrapped_call_impl(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(p.device for p in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return next(p.dtype for p in self.parameters() if p.is_floating_point())

    @staticmethod
    def auto(config: ModelConfig, generation_config: GenerationConfig) -> Architecture:
        return ARCHITECTURE_CLASSES[config.architecture](config, generation_config)

    def __init_subclass__(cls) -> None:
        ARCHITECTURE_CLASSES[cls.__name__.lower()] = cls
