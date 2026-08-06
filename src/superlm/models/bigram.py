import torch.nn as nn
from torch import Tensor
from ..tokenizer import Tokenizer
from ..config import ModelConfig, GenerationConfig
from ..generation import GenerationModule

__all__ = ['Bigram'] 

class Bigram(GenerationModule):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__(tokenizer, config, generation_config)
        n = self.config.vocab_size
        self.logits = nn.Embedding(n, n)

    def forward(self, idx: Tensor) -> Tensor:
        return self.logits(idx)