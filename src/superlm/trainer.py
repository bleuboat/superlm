import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import time
from typing import Any, cast
from zlib import crc32

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from .architectures import CausalLM
from .config import AdamConfig, TrainingConfig
from .dataset import TextDataset
from .paths import CHECKPOINTS_PATH
from .tokenizer import Tokenizer

__all__ = ["Trainer"]

IS_CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    torch_random_state: Tensor
    cuda_random_state: Tensor
    step: int
    loss: float
    losses: list[tuple[int, float]]


class Trainer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: CausalLM,
        data: Iterable[str],
        training_config: TrainingConfig,
        adam_config: AdamConfig,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

        self.smooth_loss = math.log(self.tokenizer.vocab_size)
        self.best_loss = 100.0
        self.epochs = training_config.epochs
        self.batch_size = training_config.batch_size
        self.accumulation_steps = training_config.accumulation_steps
        self.num_steps = 0
        self.step = 0
        self.start_time = 0.0
        self.losses: list[tuple[int, float]] = []

        self.optimizer = optim.AdamW(model.parameters(), **adam_config)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.optimizer.zero_grad()

        self.dataset = TextDataset(self.tokenizer, data, self.model.config.block_size)
        self.dataloader = cast(
            Iterable[tuple[Tensor, Tensor]],
            DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=IS_CUDA_AVAILABLE,
            ),
        )
        self.dataloader_iter = iter(self.dataloader)
        self.num_steps = (
            (self.dataset.length * self.epochs) // (self.model.config.block_size * self.batch_size)
        )

        hash = crc32(
            json.dumps(dict(self.model.config), sort_keys=True).encode()
            + self.dataset.dataset.numpy().tobytes(),
        )
        self.checkpoint_path = CHECKPOINTS_PATH / f"{hash:x}.pth"

    def train_step(self) -> int:
        self.step += 1
        try:
            inputs, targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            inputs, targets = next(self.dataloader_iter)
        inputs = inputs.to(self.model.device, non_blocking=True)
        targets = targets.to(self.model.device, non_blocking=True)
        outputs = self.model(inputs, targets)
        loss = cast(Tensor, outputs.loss)
        (loss / self.accumulation_steps).backward()
        self.smooth_loss = self.smooth_loss * 0.999 + loss.item() * 0.001
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return self.step

    def train(self, steps: Sequence[int] | None = None) -> None:
        if self.checkpoint_path.exists():
            self.restart()
        else:
            self.checkpoint()
        if steps is None:
            steps = (50, 100, 200)
        self.start_time = time()
        while True:
            step = self.train_step()
            if step % steps[0] == 0:
                self.log()
            if step % steps[1] == 0:
                self.sample()
            if step % steps[2] == 0:
                self.checkpoint()
            if step >= self.num_steps:
                break
        self.log()
        self.sample()
        self.checkpoint_path.unlink()

    def log(self) -> None:
        self.losses.append((self.step, self.smooth_loss))
        time_used = time() - self.start_time
        print("----")
        print(
            f"iter [{self.step}/{self.num_steps}]", "|",
            f"loss: {self.smooth_loss:.6f}", "|",
            f"time: {time_used:.6f}s",
        )

    def sample(self) -> None:
        sample_ix = self.model.generate(
            inputs=self.tokenizer(
                contents=("",),
                special_tokens=("bos",),
                device=self.model.device,
            ),
            max_new_tokens=100,
        )
        txt = self.tokenizer.decode(sample_ix)[0]
        print("----")
        print(txt)

    def checkpoint(self) -> None:
        if self.smooth_loss > self.best_loss:
            return
        self.best_loss = self.smooth_loss
        checkpoint = Checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            torch_random_state=torch.get_rng_state(),
            cuda_random_state=(
                torch.cuda.get_rng_state() if IS_CUDA_AVAILABLE else torch.get_rng_state()
            ),
            step=self.step,
            loss=self.smooth_loss,
            losses=self.losses,
        )
        torch.save(checkpoint, self.checkpoint_path)

    def restart(self) -> None:
        checkpoint: Checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        torch.set_rng_state(checkpoint.torch_random_state)
        if IS_CUDA_AVAILABLE:
            torch.cuda.set_rng_state(checkpoint.cuda_random_state)
        self.step = checkpoint.step
        self.smooth_loss = checkpoint.loss
        self.best_loss = checkpoint.loss
        self.losses = checkpoint.losses

    def _lr_lambda(self, step: int) -> float:
        if not self.num_steps:
            return 0.0
        warmup_steps = int(self.num_steps * 0.05)
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (self.num_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
