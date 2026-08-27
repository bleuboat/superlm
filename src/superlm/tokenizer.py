import re
from collections.abc import Container, Iterable
from typing import cast

import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType

from .config import ConfigGroup

PATTERN = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?[A-Za-z]+|[\r\n]+|\s+|.")


class Tokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.vocab_size = len(tokens) - 1
        self.token_to_ix = {ch: i for i, ch in enumerate(tokens)}
        self.bos_token_ix = self.token_to_ix["<BOS>"]
        self.eos_token_ix = self.token_to_ix["<EOS>"]
        self.pad_token_ix = self.token_to_ix["<PAD>"]
        self.unk_token_ix = self.token_to_ix["<UNK>"]
        self.special_tokens = {
            self.bos_token_ix,
            self.eos_token_ix,
            self.pad_token_ix,
            self.unk_token_ix,
        }

    def encode(
        self,
        contents: Iterable[str],
        *,
        special_tokens: Container[str] = (),
        dtype: torch.dtype = torch.long,
        device: DeviceLikeType,
    ) -> Tensor:
        tokens_list: list[list[str]] = []
        tokens_length: list[int] = []
        for content in contents:
            tokens = PATTERN.findall(content)
            tokens_list.append(tokens)
            tokens_length.append(len(tokens))
        max_length = max(tokens_length)

        def _encode(tokens: list[str]) -> list[int]:
            out: list[int] = []
            if "bos" in special_tokens:
                out.append(self.bos_token_ix)
            out.extend(self.token_to_ix.get(token, self.unk_token_ix) for token in tokens)
            if "eos" in special_tokens:
                out.append(self.eos_token_ix)
            if "pad" in special_tokens:
                out.extend([self.pad_token_ix] * (max_length - len(tokens)))
            return out

        return torch.tensor(
            [_encode(tokens) for tokens in tokens_list],
            dtype=dtype,
            device=device,
        )

    def decode(self, contents: Tensor | Iterable[Tensor]) -> list[str]:
        def _decode(content: Tensor) -> str:
            out: list[str] = []
            for token in content:
                ix = cast(int, token.item())
                if ix not in self.special_tokens:
                    out.append(self.tokens[ix])
            return "".join(out)

        return [_decode(content) for content in contents]

    __call__ = encode

    @classmethod
    def from_data(cls, data: Iterable[str]) -> Tokenizer:
        tokens = sorted({token for text in data for token in PATTERN.findall(text)})
        tokens.extend(("<BOS>", "<EOS>", "<PAD>", "<UNK>"))
        return cls(tokens)

    def update_config(self, config: ConfigGroup) -> None:
        config.model.vocab_size = self.vocab_size
        config.model.bos_token_ix = self.bos_token_ix
        config.model.eos_token_ix = self.eos_token_ix
        config.model.pad_token_ix = self.pad_token_ix
        config.generation.bos_token_ix = self.bos_token_ix
        config.generation.eos_token_ix = self.eos_token_ix
        config.generation.pad_token_ix = self.pad_token_ix
