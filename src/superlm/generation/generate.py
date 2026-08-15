import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from collections.abc import Generator

from superlm.streamer import Streamer
from superlm.config import ModelConfig, GenerationConfig

from .logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from .stopping_criteria import (
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxTimeCriteria,
    EosTokenCriteria,
)


__all__ = ["GenerationMixin"]


class GenerationMixin(nn.Module):
    config: ModelConfig
    generation_config: GenerationConfig

    def _get_generation_config(self, inputs: Tensor, **kwargs: Any) -> GenerationConfig:
        generation_config = self.generation_config.copy()
        generation_config.update(**kwargs)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + inputs.shape[-1]
        return generation_config

    def _get_processors(self, generation_config: GenerationConfig) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if generation_config.repetition_penalty != 1:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.do_sample:
            if generation_config.temperature != 1:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            if generation_config.top_k != 0:
                processors.append(TopKLogitsWarper(generation_config.top_k))
            if generation_config.top_p < 1:
                processors.append(TopPLogitsWarper(generation_config.top_p))
        return processors

    def _get_criteria(self, generation_config: GenerationConfig) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.eos_token_ix is not None:
            criteria.append(EosTokenCriteria(eos_token_ix=generation_config.eos_token_ix))
        if generation_config.max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        return criteria

    @torch.no_grad()
    def generate(self, inputs: Tensor, streamer: Streamer | None = None, **kwargs: Any) -> Tensor:
        generation_config = self._get_generation_config(inputs, **kwargs)
        processors = self._get_processors(generation_config)
        criteria = self._get_criteria(generation_config)

        if streamer is not None:
            streamer.put(inputs.cpu())

        do_sample = generation_config.do_sample
        unfinished = True
        while unfinished:
            outputs, _ = self(inputs)
            logits = outputs[:, -1, :]
            score = processors(inputs, logits)
            if do_sample:
                probs = F.softmax(score, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(score, dim=-1).unsqueeze(1)
            inputs = torch.cat((inputs, next_token), dim=-1)
            if streamer is not None:
                streamer.put(next_token.cpu())
            unfinished = not criteria(inputs)
        if streamer is not None:
            streamer.end()
        return inputs

    @torch.no_grad()
    def api(self, inputs: Tensor, **kwargs: Any) -> Generator[Tensor]:
        generation_config = self._get_generation_config(inputs, **kwargs)
        processors = self._get_processors(generation_config)
        criteria = self._get_criteria(generation_config)

        do_sample = generation_config.do_sample
        unfinished = True
        while unfinished:
            outputs = self(inputs)
            logits = outputs[:, -1, :]
            score = processors(inputs, logits)
            if do_sample:
                probs = F.softmax(score, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(score, dim=-1).unsqueeze(1)
            yield next_token
            inputs = torch.cat((inputs, next_token), dim=-1)
            unfinished = not criteria(inputs)
