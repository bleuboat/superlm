from __future__ import annotations

import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Iterable, Sequence

from .config import TrainingConfig, AdamConfig
from .tokenizer import Tokenizer
from .model import Transformer
from .dataset import TextDataset

__all__ = ['Trainer']

class Trainer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: Transformer,
        device: torch.device,
        checkpoint_path: str,
        data: Iterable[str],
        training_config: TrainingConfig,
        adam_config: AdamConfig,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.smooth_loss = math.log(self.tokenizer.vocab_size)
        self.best_loss = 100.0
        self.epochs = training_config.epochs
        self.block_size = training_config.block_size
        self.batch_size = training_config.batch_size
        self.accumulation_steps = training_config.accumulation_steps
        self.num_steps = 0
        self.step = 0
        self.start_time = None
        self.losses = []

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_ix if tokenizer.pad_token_ix is not None else -100)
        self.optimizer = optim.AdamW(model.parameters(), **adam_config)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.optimizer.zero_grad()

        self.dataset = TextDataset(self.tokenizer, data, self.block_size)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        self.dataloader_iter = iter(self.dataloader)
        self.num_steps = (self.dataset.length * self.epochs) // (self.block_size * self.batch_size)
    
    def train_step(self) -> int:
        self.step += 1
        try:
            inputs, targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            inputs, targets = next(self.dataloader_iter)
        inputs  = inputs .to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1, self.model.config.vocab_size), targets.view(-1))
        (loss / self.accumulation_steps).backward()
        self.smooth_loss = self.smooth_loss * 0.999 + loss.item() * 0.001
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        return self.step

    def train(self, prompt: str = '', length: int | None = None, steps: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if os.path.exists(self.checkpoint_path):
            self.restart()
        if steps is None:
            steps = (50, 100, 200)
        self.start_time = time.time()
        while True:
            step = self.train_step()
            if step % steps[0] == 0:
                self.log()
            if step % steps[1] == 0:
                self.sample(prompt, length)
            if step % steps[2] == 0:
                self.checkpoint()
            if step >= self.num_steps:
                break
        return self.model.state_dict()

    def log(self) -> None:
        self.losses.append((self.step, self.smooth_loss))
        print('----')
        print(f'iter [{self.step}/{self.num_steps}] | loss: {self.smooth_loss:.6f} | time: {time.time() - self.start_time:.6f}s')

    def sample(self, prompt: str = '', length: int | None = None) -> None:
        sample_ix = self.model.generate(
            inputs=self.tokenizer(
                contents=[prompt],
                special_tokens=(('bos',) if self.tokenizer.bos_token_ix >= 0 else ()),
                device=self.device,
            ),
            max_new_tokens=length if length else self.block_size,
        )
        txt = self.tokenizer.decode(sample_ix)[0]
        print('----')
        print(txt)
        
    def checkpoint(self) -> None:
        if self.smooth_loss >= self.best_loss:
            return
        self.best_loss = self.smooth_loss
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'smooth_loss': self.smooth_loss,
            'losses': self.losses,
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state(),
        }
        torch.save(checkpoint, self.checkpoint_path)

    def restart(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.smooth_loss = checkpoint['smooth_loss']
        self.best_loss = self.smooth_loss
        self.losses = checkpoint['losses']
        torch.set_rng_state(checkpoint['torch_random_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_random_state'])
    
    def _lr_lambda(self, step: int) -> float:
        if not self.num_steps:
            return 0.0
        warmup_steps = int(self.num_steps * 0.05)
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (self.num_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))