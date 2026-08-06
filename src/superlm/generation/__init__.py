import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType
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
    EosTokenCriteria
)
from ..tokenizer import Tokenizer
from ..streamer import Streamer
from ..config import ModelConfig, GenerationConfig

__all__ = ['GenerationModule']

class GenerationModule(nn.Module):
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig, generation_config: GenerationConfig) -> None:
        super().__init__()

        self.config = config.copy()
        self.generation_config = generation_config.copy()
        
        self.config.vocab_size = tokenizer.vocab_size
        self.generation_config.bos_token_ix = tokenizer.bos_token_ix
        self.generation_config.eos_token_ix = tokenizer.eos_token_ix
        self.generation_config.pad_token_ix = tokenizer.pad_token_ix

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        streamer: Streamer | None = None,
        device: DeviceLikeType | None = None,
        **kwargs,
    ) -> torch.Tensor:
        generation_config = self.generation_config.copy()
        generation_config.update(**kwargs)

        if inputs is None:
            inputs = torch.tensor([[generation_config.bos_token_ix]], dtype=torch.long, device=device)

        if streamer is not None:
            streamer.put(inputs.cpu())

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + inputs.shape[-1]

        processors = LogitsProcessorList()
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.do_sample:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            if generation_config.top_k is not None and generation_config.top_k != 0:
                processors.append(TopKLogitsWarper(generation_config.top_k))
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                processors.append(TopPLogitsWarper(generation_config.top_p))

        criteria = StoppingCriteriaList()
        criteria.append(EosTokenCriteria(eos_token_ix=generation_config.eos_token_ix))
        if generation_config.max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))

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
            inputs = torch.cat((inputs, next_token), dim=-1)
            if streamer is not None:
                streamer.put(next_token.cpu())
            unfinished = not criteria(inputs)
        if streamer is not None:
            streamer.end()
        return inputs