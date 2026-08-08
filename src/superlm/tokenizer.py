from __future__ import annotations
import re
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typing import Container, Iterable

__all__ = ['TokenNotFoundError', 'Tokenizer']

PATTERN = r'[A-Za-z]+|[^A-Za-z]'

class TokenNotFoundError(LookupError):
    def __init__(self, token: str, *, text: str) -> None:
        super().__init__(f'Unknown token "{token}" (at text "{text}")')

class Tokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.token_to_ix = { ch:i for i,ch in enumerate(tokens) }
        self.bos_token_ix = self.token_to_ix['<BOS>']
        self.eos_token_ix = self.token_to_ix['<EOS>']
        self.pad_token_ix = self.token_to_ix['<PAD>']

    def encode(self, contents: Iterable[str], *, special_tokens: Container[str], device: DeviceLikeType) -> Tensor:
        tokens_list = []
        tokens_length = []
        for content in contents:
            tokens = re.findall(PATTERN, content)
            tokens_list.append(tokens)
            tokens_length.append(len(tokens))
        max_length = max(tokens_length)

        def _encode(tokens: list[str]) -> list[int]:
            out = []
            if 'bos' in special_tokens:
                out.append(self.bos_token_ix)
            for token in tokens:
                if token in self.token_to_ix:
                    out.append(self.token_to_ix[token])
                else:
                    raise TokenNotFoundError(token, text=''.join(tokens))
            if 'eos' in special_tokens:
                out.append(self.eos_token_ix)
            if 'pad' in special_tokens:
                for _ in range(max_length - len(tokens)):
                    out.append(self.pad_token_ix)
            return out

        return torch.tensor([_encode(tokens) for tokens in tokens_list], dtype=torch.long, device=torch.device(device))

    def decode(self, contents: Tensor | Iterable[Tensor]) -> list[str]:
        def _decode(content: Tensor) -> str:
            out = []
            for token in content:
                ix = token.item()
                if ix not in (self.bos_token_ix, self.eos_token_ix, self.pad_token_ix):
                    out.append(self.tokens[ix])
            return ''.join(out)
        return [_decode(content) for content in contents]

    __call__ = encode

    @classmethod
    def from_data(cls, data: str | Iterable[str]) -> Tokenizer:
        data = data if isinstance(data, str) else ''.join(data)
        tokens = set(re.findall(PATTERN, data))
        tokens.update(('<BOS>', '<EOS>', '<PAD>'))
        return cls(sorted(tokens))