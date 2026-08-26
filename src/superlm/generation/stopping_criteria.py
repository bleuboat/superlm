import time
from abc import ABC, abstractmethod

from torch import Tensor


class StoppingCriteria(ABC):
    @abstractmethod
    def __call__(self, input_ids: Tensor) -> bool:
        raise NotImplementedError


class StoppingCriteriaList(list[StoppingCriteria], StoppingCriteria):
    def __call__(self, input_ids: Tensor) -> bool:
        return any(criteria(input_ids) for criteria in self)


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, input_ids: Tensor) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time: float, initial_timestamp: float | None = None) -> None:
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: Tensor) -> bool:
        return time.time() - self.initial_timestamp > self.max_time


class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_ix: int) -> None:
        self.eos_token_ix = eos_token_ix

    def __call__(self, input_ids: Tensor) -> bool:
        return input_ids[:, -1].item() == self.eos_token_ix
