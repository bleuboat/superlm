import torch
from torch import Tensor
from abc import ABC, abstractmethod


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        raise NotImplementedError


class LogitsProcessorList(list[LogitsProcessor], LogitsProcessor):
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class TemperatureLogitsWarper(LogitsProcessor):
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        return scores / self.temperature


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return scores.scatter(1, input_ids, score)


class TopPLogitsWarper(LogitsProcessor):
    def __init__(self, top_p: float, min_tokens_to_keep: int = 1) -> None:
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        logits, indices = torch.sort(scores, descending=False)
        cumulative_probs = logits.softmax(dim=-1).cumsum(dim=-1)
        indices_to_remove = cumulative_probs <= (1 - self.top_p)
        indices_to_remove[..., -self.min_tokens_to_keep:] = 0
        indices_to_remove = indices_to_remove.scatter(1, indices, indices_to_remove)
        return scores.masked_fill(indices_to_remove, -float("inf"))


class TopKLogitsWarper(LogitsProcessor):
    def __init__(self, top_k: int, min_tokens_to_keep: int = 1) -> None:
        self.top_k = max(top_k, min_tokens_to_keep)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        top_k = min(self.top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        return scores.masked_fill(indices_to_remove, -float("inf"))
