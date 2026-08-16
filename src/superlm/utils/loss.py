import torch
import torch.nn.functional as F
from torch import Tensor
from superlm.config import ModelConfig


class CausalLMLoss:
    def __init__(self, config: ModelConfig) -> None:
        self.vocab_size = config.vocab_size
        self.ignore_index = config.pad_token_ix if config.pad_token_ix is not None else -100

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)


class SequenceClassificationLoss:
    def __init__(self, config: ModelConfig) -> None:
        self.num_labels = config.num_labels

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.num_labels == 1:
            return F.mse_loss(logits.squeeze(), labels.squeeze())
        if labels.dtype in {torch.long, torch.int}:
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return F.binary_cross_entropy_with_logits(logits, labels)


class TokenClassificationLoss:
    def __init__(self, config: ModelConfig) -> None:
        self.num_labels = config.num_labels
        self.ignore_index = config.pad_token_ix if config.pad_token_ix is not None else -100

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)
