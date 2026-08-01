from torch import Tensor
from .tokenizer import Tokenizer

__all__ = ['Streamer']

class Streamer:
    def __init__(self, tokenizer: Tokenizer, skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.next_tokens_are_prompt = True

    def put(self, value: Tensor) -> None:
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        if value.dim() == 1:
            value = [value]
        print(self.tokenizer.decode(value, **self.decode_kwargs)[0], flush=True, end='')

    def end(self) -> None:
        self.next_tokens_are_prompt = True
        print(flush=True)