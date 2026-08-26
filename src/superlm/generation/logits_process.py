from abc import ABC, abstractmethod

import torch
from torch import Tensor


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        raise NotImplementedError


class LogitsProcessorList(list[LogitsProcessor], LogitsProcessor):
    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_ix: int) -> None:
        self.min_length = min_length
        self.eos_token_ix = eos_token_ix

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if input_ids.shape[-1] < self.min_length:
            vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
            scores = torch.where(vocab_tensor == self.eos_token_ix, -float("inf"), scores)
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
    def __init__(self, top_p: float) -> None:
        self.top_p = top_p

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        logits, indices = torch.sort(scores, descending=False)
        cumulative_probs = logits.softmax(dim=-1).cumsum(dim=-1)
        indices_to_remove = cumulative_probs <= (1 - self.top_p)
        indices_to_remove[..., -1:] = 0
        indices_to_remove = indices_to_remove.scatter(1, indices, indices_to_remove)
        return scores.masked_fill(indices_to_remove, -float("inf"))


class TopKLogitsWarper(LogitsProcessor):
    def __init__(self, top_k: int) -> None:
        self.top_k = max(top_k, 1)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        top_k = min(self.top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        return scores.masked_fill(indices_to_remove, -float("inf"))


class TopHLogitsWarper(LogitsProcessor):
    def __init__(self, top_h: float) -> None:
        self.top_h = top_h

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        batch_size, vocab_size = scores.shape
        device = scores.device
        keep_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
        top_n = min(100, vocab_size)

        top_logits, top_idx = torch.topk(scores, top_n, dim=-1, largest=True, sorted=True)

        dist = torch.distributions.Categorical(logits=top_logits)
        probs = dist.probs
        log_probs = torch.log(probs)

        tau = (dist.entropy() * self.top_h).unsqueeze(-1)

        entropy_terms = -probs * log_probs
        cumulative_entropy = torch.cumsum(entropy_terms, dim=-1)

        selection_mask = cumulative_entropy <= tau
        selection_mask[:, 0] = True

        keep_mask.scatter_(dim=1, index=top_idx, src=selection_mask)

        scores_processed = scores.clone()
        scores_processed[~keep_mask] = -float("inf")
        return scores_processed


class MinPLogitsWarper(LogitsProcessor):
    def __init__(self, min_p: float) -> None:
        self.min_p = min_p

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        probs = torch.softmax(scores, dim=-1)
        top_probs = probs.amax(dim=-1, keepdim=True)
        scaled_min_p = self.min_p * top_probs
        tokens_to_remove = probs < scaled_min_p

        k = min(1, probs.shape[-1])
        sorted_indices = torch.topk(probs, k, dim=-1).indices
        tokens_to_remove.scatter_(-1, sorted_indices, value=False)

        return scores.masked_fill(tokens_to_remove, -float("inf"))
