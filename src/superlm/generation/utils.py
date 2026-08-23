import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Unpack, cast
from collections.abc import Generator

from superlm.streamer import Streamer
from superlm.config import GenerationConfig, GenerationConfigTypedDict
from superlm.architectures import Architecture

from .logits_process import *
from .stopping_criteria import *


__all__ = ["GenerationArchitecture"]


class GenerationArchitecture(Architecture):
    def _get_generation_config(
        self,
        inputs: Tensor,
        **kwargs: Unpack[GenerationConfigTypedDict],
    ) -> GenerationConfig:
        generation_config = self.generation_config.copy()
        generation_config.update(**kwargs)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + inputs.shape[-1]
        if generation_config.min_new_tokens is not None:
            generation_config.min_length = generation_config.min_new_tokens + inputs.shape[-1]
        return generation_config

    @staticmethod
    def _get_processors(generation_config: GenerationConfig) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if generation_config.repetition_penalty is not None:
            processors.append(RepetitionPenaltyLogitsProcessor(
                generation_config.repetition_penalty,
            ))
        if generation_config.min_length is not None and generation_config.eos_token_ix is not None:
            processors.append(MinLengthLogitsProcessor(
                generation_config.min_length, generation_config.eos_token_ix,
            ))
        if generation_config.do_sample:
            if generation_config.temperature is not None:
                processors.append(TemperatureLogitsWarper(
                    generation_config.temperature,
                ))
            if generation_config.top_k is not None:
                processors.append(TopKLogitsWarper(
                    generation_config.top_k,
                ))
            if generation_config.top_p is not None:
                processors.append(TopPLogitsWarper(
                    generation_config.top_p,
                ))
            if generation_config.top_h is not None:
                processors.append(TopHLogitsWarper(
                    generation_config.top_h,
                ))
            if generation_config.min_p is not None:
                processors.append(MinPLogitsWarper(
                    generation_config.min_p,
                ))
        return processors

    @staticmethod
    def _get_criteria(generation_config: GenerationConfig) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.eos_token_ix is not None:
            criteria.append(EosTokenCriteria(eos_token_ix=generation_config.eos_token_ix))
        if generation_config.max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        return criteria

    @torch.no_grad()
    def generate(
        self,
        inputs: Tensor,
        *,
        streamer: Streamer | None = None,
        **kwargs: Unpack[GenerationConfigTypedDict],
    ) -> Tensor:
        generation_config = self._get_generation_config(inputs, **kwargs)
        processors = self._get_processors(generation_config)
        criteria = self._get_criteria(generation_config)

        if streamer:
            streamer.put(inputs.cpu())

        do_sample = generation_config.do_sample
        unfinished = True
        while unfinished:
            outputs = self(inputs)
            logits = cast(Tensor, outputs.logits)
            score = processors(inputs, logits[:, -1, :])
            if do_sample:
                probs = F.softmax(score, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(score, dim=-1).unsqueeze(1)
            inputs = torch.cat((inputs, next_token), dim=-1)
            unfinished = not criteria(inputs)
            if streamer:
                streamer.put(next_token.cpu())

        if streamer:
            streamer.end()

        return inputs

    @torch.no_grad()
    def api(self, inputs: Tensor, **kwargs: Unpack[GenerationConfigTypedDict]) -> Generator[Tensor]:
        generation_config = self._get_generation_config(inputs, **kwargs)
        processors = self._get_processors(generation_config)
        criteria = self._get_criteria(generation_config)

        do_sample = generation_config.do_sample
        unfinished = True
        while unfinished:
            outputs = self(inputs)
            logits = cast(Tensor, outputs.logits)
            score = processors(inputs, logits[:, -1, :])
            if do_sample:
                probs = F.softmax(score, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(score, dim=-1).unsqueeze(1)
            yield next_token
            inputs = torch.cat((inputs, next_token), dim=-1)
            unfinished = not criteria(inputs)
