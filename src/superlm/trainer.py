from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
from .config import TrainingConfig, AdamConfig
from .tokenizer import Tokenizer
from .model import Transformer
from .dataset import TextDataset

__all__ = ['Trainer']

class Trainer:
    def __init__(self, tokenizer: Tokenizer, model: Transformer, device: torch.device, training_config: TrainingConfig, adam_config: AdamConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

        self.smooth_loss = math.log(self.tokenizer.vocab_size)
        self.best_loss = None
        self.epochs = training_config.epochs
        self.block_size = training_config.block_size
        self.batch_size = training_config.batch_size
        self.accumulation_steps = training_config.accumulation_steps
        self.num_steps = 0
        self.step = 0
        self.losses = []

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_ix if tokenizer.pad_token_ix is not None else -100)
        self.optimizer = optim.AdamW(model.parameters(), **adam_config)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.optimizer.zero_grad()

        self.dataset = TextDataset(self.tokenizer, self.workspace.data, self.block_size)
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

    def train(self, **steps: int) -> None:
        log_step = steps.get('log', 50)
        sample_step = steps.get('sample', 100)
        checkpoint_step = steps.get('checkpoint', 200)
        while True:
            step = self.train_step()
            if step % log_step == 0:
                self.log()
            if step % sample_step == 0:
                self.sample()
            if step % checkpoint_step == 0:
                self.checkpoint()
            if step >= self.num_steps:
                break

    def log(self) -> None:
        self.losses.append((self.step, self.smooth_loss))
        print('----')
        print(f'iter [{self.step}/{self.num_steps}] | loss: {self.smooth_loss:.6f} | time: {time.time() - self.start_time:.6f}s')

    def sample(self, prompt: str, length: int | None = None) -> None:
        sample_ix = self.model.generate(
            inputs=self.tokenizer([prompt], ('bos',) if self.tokenizer.bos_token_ix >= 0 else ()),
            max_new_tokens=length if length else self.block_size,
        )
        txt = self.tokenizer.decode(sample_ix)[0]
        print('----')
        print(txt)

    def show(self) -> None:
        import matplotlib.pyplot as plt
        x, y = zip(*self.losses)
        plt.plot(x, y)
        plt.xlabel('num_steps')
        plt.ylabel('loss')
        plt.grid(True)
        plt.savefig(self.workspace.fig_path, dpi=300)
        plt.show()
        
    def checkpoint(self) -> None:
        if self.best_loss is not None and self.smooth_loss >= self.best_loss:
            return
        self.best_loss = self.smooth_loss
        
        checkpoint = {
            # model
            'model_config': self.model.config,
            'model_generation_config': self.model.generation_config,
            'model_state_dict': self.model.state_dict(),

            # tokenizer
            'tokens': self.tokenizer.tokens,

            # trainer
            'epochs': self.epochs,
            'block_size': self.block_size,
            'batch_size': self.batch_size,
            'accumulation_steps': self.accumulation_steps,
            'optimizer_defaults': self.optimizer.defaults,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'smooth_loss': self.smooth_loss,
            'losses': self.losses,

            # random
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state(),
        }
        torch.save(checkpoint, self.workspace.model_path)

    @staticmethod
    def load(path, *, train: bool = True) -> tuple[Transformer, Tokenizer, Trainer] | tuple[Transformer, Tokenizer]:
        checkpoint = torch.load(path, weights_only=False)
        model_config = checkpoint['model_config']
        model_generation_config = checkpoint['model_generation_config']
        model_state_dict = checkpoint['model_state_dict']
        tokens = checkpoint['tokens']
        epochs = checkpoint['epochs']
        block_size = checkpoint['block_size']
        batch_size = checkpoint['batch_size']
        accumulation_steps = checkpoint['accumulation_steps']
        optimizer_defaults = checkpoint['optimizer_defaults']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict']
        step = checkpoint['step']
        smooth_loss = checkpoint['smooth_loss']
        losses = checkpoint['losses']
        torch_random_state = checkpoint['torch_random_state']
        cuda_random_state = checkpoint['cuda_random_state']

        tokenizer = Tokenizer(tokens)
        model = Transformer(tokenizer, **model_config, **model_generation_config)
        model.load_state_dict(model_state_dict)
        torch.set_rng_state(torch_random_state)
        torch.cuda.set_rng_state(cuda_random_state)

        if not train:
            model.eval()
            return model, tokenizer
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            epochs=epochs,
            block_size=block_size,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            **optimizer_defaults,
        )
        trainer.optimizer.load_state_dict(optimizer_state_dict)
        trainer.scheduler.load_state_dict(scheduler_state_dict)
        trainer.step = step
        trainer.smooth_loss = smooth_loss
        trainer.best_loss = smooth_loss
        trainer.losses = losses
        return model, tokenizer, trainer
    
    def _lr_lambda(self, step: int) -> float:
        if not self.num_steps:
            return 0.0
        warmup_steps = int(self.num_steps * 0.05)
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (self.num_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))