from typing import overload
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig
from ..generation import GenerationModule

from .transformer import Transformer

__all__ = ['Model']

MODELS = {
    'transformer': Transformer,
}

class Model:
    def __new__(tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> GenerationModule:
        return MODELS[config.model_type](tokenizer, config, generation_config)

    @overload
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        ...