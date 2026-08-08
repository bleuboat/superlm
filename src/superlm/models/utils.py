import torch.nn as nn
from typing import Callable, TypeVar
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig

__all__ = ['Model']

class Model(nn.Module):
    '''
    Base class for all neural network models.

    All models should subclass this class.
    '''
    _model_classes: dict[str, type[Model]] = {}

    def __new__(cls, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> Model:
        if cls is not Model:
            return super().__new__(cls)
        model_class = cls._model_classes.get(config.model_type, None)
        if model_class is None:
            raise RuntimeError('Model type not defined')
        return model_class(tokenizer, config, generation_config)

    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
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

    @classmethod
    def register_model(cls, name: str) -> Callable[[T], T]:
        def decorator(model_class: T) -> T:
            cls._model_classes[name] = model_class
            return model_class
        return decorator

T = TypeVar('T', bound=type[Model])