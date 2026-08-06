from typing import overload
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig
from ..generation import GenerationModule

from .transformer import Transformer
from .bigram import Bigram
from .bow import BoW

__all__ = ['Model']

class Model:
    models = {
        'transformer': Transformer,
        'bigram': Bigram,
        'bow': BoW,
    }
    
    def __new__(cls, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> GenerationModule:
        model_class = cls.models.get(config.model_type, None)
        if model_class is None:
            raise RuntimeError('Model type not defined')
        return model_class(tokenizer, config, generation_config)

    @overload
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        ...