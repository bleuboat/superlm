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
        print('正在准备 Tokenizer……')
        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.token_to_ix = { ch:i for i,ch in enumerate(tokens) }
        self.bos_token_ix = self.eos_token_ix = self.pad_token_ix = None
        if '<BOS>' in self.token_to_ix:
            self.bos_token_ix = self.token_to_ix['<BOS>']
        if '<EOS>' in self.token_to_ix:
            self.eos_token_ix = self.token_to_ix['<EOS>']
        if '<PAD>' in self.token_to_ix:
            self.pad_token_ix = self.token_to_ix['<PAD>']

    def encode(self, contents: Iterable[str], *, special_tokens: Container[str] | None = None, device: DeviceLikeType) -> Tensor:
        tokens_list = []
        tokens_length = []
        for content in contents:
            tokens = re.findall(PATTERN, content)
            tokens_list.append(tokens)
            tokens_length.append(len(tokens))
        max_length = max(tokens_length)

        def _encode(tokens: list[str]) -> list[int]:
            out = []
            if (special_tokens is None or 'bos' in special_tokens) and self.bos_token_ix is not None:
                out.append(self.bos_token_ix)
            for token in tokens:
                if token in self.token_to_ix:
                    out.append(self.token_to_ix[token])
                else:
                    raise TokenNotFoundError(token, text=''.join(tokens))
            if (special_tokens is None or 'eos' in special_tokens) and self.eos_token_ix is not None:
                out.append(self.eos_token_ix)
            if (special_tokens is None or 'pad' in special_tokens) and self.pad_token_ix is not None:
                for _ in range(max_length - len(tokens)):
                    out.append(self.pad_token_ix)
            return out
        
        return torch.tensor([_encode(tokens) for tokens in tokens_list], dtype=torch.long, device=torch.device(device))
    
    def decode(self, contents: Tensor | Iterable[Tensor], skip_special_tokens: bool = True) -> list[str]:
        def _decode(content: Tensor) -> str:
            out = []
            for token in content:
                ix = token.item()
                if not (skip_special_tokens and ix in (self.bos_token_ix, self.eos_token_ix, self.pad_token_ix)):
                    out.append(self.tokens[ix])
            return ''.join(out)
        return [_decode(content) for content in contents]
    
    __call__ = encode
    
    @classmethod
    def from_data(cls, data: str | Iterable[str], special_tokens: Iterable[str] = ()) -> Tokenizer:
        data = data if isinstance(data, str) else ''.join(data)
        tokens = set(re.findall(PATTERN, data))
        tokens.update(f'<{token.upper()}>' for token in special_tokens)
        return cls(sorted(tokens))