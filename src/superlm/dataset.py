import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Container, Iterable
from .tokenizer import Tokenizer

__all__ = ['TextDataset']

class TextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, data: str | Iterable[str], block_size: int, special_tokens: Container[str] = ()) -> None:
        self.dataset = tokenizer([data] if isinstance(data, str) else data, special_tokens=special_tokens, device='cpu')
        self.block_size = block_size
        self.length = 0
        self.indexes = []
        for i in range(len(self.dataset)):
            pads = torch.nonzero(self.dataset[i] == tokenizer.pad_token_ix)
            length = pads[0].item() if pads.shape[0] else len(self.dataset[i])
            self.length += length
            for j in range(max(length - self.block_size, 1)):
                self.indexes.append((i, j))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        i, j = self.indexes[index]
        inputs  = self.dataset[i, j     : j + self.block_size    ]
        targets = self.dataset[i, j + 1 : j + self.block_size + 1]
        return inputs, targets

    def __len__(self) -> int:
        return len(self.indexes)