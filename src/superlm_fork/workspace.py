from __future__ import annotations

import os
import time
import json
import torch
import dotenv

from torch._prims_common import DeviceLikeType
from typing import Sequence

from .config import *
from .tokenizer import Tokenizer
from .streamer import Streamer
from .model import Transformer
from .trainer import Trainer

try:
    import matplotlib.pyplot as plt # type: ignore
except ModuleNotFoundError:
    plt = None

__all__ = ['WorkSpace', 'command']

def _get_workspace() -> str:
    dotenv.load_dotenv()
    return os.getenv('WORKSPACE_PATH')

WORKSPACE_PATH = _get_workspace()

if WORKSPACE_PATH is None:
    dotenv.set_key('.env', 'WORKSPACE_PATH', 'workspace')
    WORKSPACE_PATH = _get_workspace()

class WorkSpace:
    name: str
    path: str
    device: torch.device
    inputs: dict[str, str]
    tokenizer: Tokenizer
    model: Transformer
    trainer: Trainer
    special_tokens: list[str]
    model_config: ModelConfig
    generation_config: GenerationConfig
    training_config: TrainingConfig
    adam_config: AdamConfig
    _train_losses: list[float]

    def __init__(self, name: str, *, seed: int | None = None, device: DeviceLikeType | None = None) -> None:
        self.name = name
        self.path = f'{WORKSPACE_PATH}/{name}'

        if seed is not None:
            torch.manual_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.special_tokens = []
        self.model_config = ModelConfig()
        self.generation_config = GenerationConfig()
        self.training_config = TrainingConfig()
        self.adam_config = AdamConfig()

        self._inputs = None
        self._tokenizer = None
        self._model = None
        self._trainer = None

        self._train_losses = []
        
        try:
            os.listdir(self.path)
            self.load()
        except FileNotFoundError:
            os.makedirs(self.path)

    @property
    def inputs(self) -> dict[str, str]:
        if self._inputs is None:
            self.get_inputs()
        return self._inputs

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self.setup_tokenizer()
        return self._tokenizer

    @property
    def model(self) -> Transformer:
        if self._model is None:
            self.setup_model()
        return self._model

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self.setup_trainer()
        return self._trainer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer) -> None:
        self._tokenizer = value

    @model.setter
    def model(self, value: Transformer) -> None:
        self._model = value

    @trainer.setter
    def trainer(self, value: Trainer) -> None:
        self._trainer = value

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_device(self, device: DeviceLikeType) -> None:
        self.device = torch.device(device)
        if self._model is not None:
            self._model.to(self.device)

    def copy(self, other: WorkSpace | str) -> WorkSpace:
        if isinstance(other, str):
            other = WorkSpace(other)
        files = os.listdir(other.path)
        for file in files:
            with open(f'{self.path}/{file}', 'w', encoding='utf-8') as f:
                f.write(open(f'{other.path}/{file}', 'r', encoding='utf-8').read())
        self.load()
        return self

    def clone(self, other: WorkSpace | str) -> WorkSpace:
        if isinstance(other, str):
            other = WorkSpace(other)
        other.copy(self)
        return other

    def config(self, **config) -> None:
        for k, v in config.items():
            if k == 'special_tokens':
                self.special_tokens = [f'<{token.upper()}>' for token in v]
            elif k in ModelConfig.params:
                self.model_config[k] = v
            elif k in GenerationConfig.params:
                self.generation_config[k] = v
            elif k in TrainingConfig.params:
                self.training_config[k] = v
            elif k in AdamConfig.params:
                self.adam_config[k] = v

    def check(self) -> None:
        for config in (
            self.model_config,
            self.generation_config,
            self.training_config,
            self.adam_config,
        ):
            config.check()

    def train(self, **steps: int) -> None:
        # start
        self.check()
        self.model.train()
        start_time = time.time()

        # loop
        pass

        # end
        self.save()
        if plt is not None:
            x, y = zip(*self._losses)
            plt.plot(x, y)
            plt.xlabel('num_steps')
            plt.ylabel('loss')
            plt.grid(True)
            plt.savefig(f'{self.path}/loss.png', dpi=300)
            plt.show()
        self.model.eval()

    def generate(self, inputs: str, *, stream: bool = False, **kwargs) -> str:
        self.check()
        if self.tokenizer.bos_token_ix is not None:
            inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=('bos'))
        else:
            inputs_tensor = self.tokenizer([inputs], device=self.device)
        if stream:
            streamer = Streamer(self.tokenizer)
        else:
            streamer = None
        res = self.model.generate(inputs_tensor, streamer=streamer, device=self.device, **kwargs)
        return self.tokenizer.decode(res)[0]

    def get_inputs(self) -> None:
        try:
            self._inputs = {'input': open(f'{self.path}/input.txt', 'r', encoding='utf-8').read()}
        except FileNotFoundError:
            try:
                self._inputs = {}
                files = os.listdir(f'{self.path}/inputs')
                for file in files:
                    self._inputs[file.split('.')[0]] = open(f'{self.path}/inputs/{file}', 'r', encoding='utf-8').read()
            except FileNotFoundError:
                raise FileNotFoundError(f'Input not found. (at {self.path})')

    def setup_tokenizer(self) -> None:
        self.tokenizer = Tokenizer.from_data(self.inputs.values(), self.special_tokens)

    def setup_model(self) -> None:
        self.model = Transformer(self.tokenizer, self.model_config, self.generation_config)
        self.model.to(self.device)

    def setup_trainer(self) -> None:
        self.trainer = Trainer(self.tokenizer, self.model, self.training_config, self.adam_config)

    def save(self) -> None:
        with open(f'{self.path}/vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.tokens, f)
        with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        with open(f'{self.path}/generation_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.model.generation_config.to_dict(), f, indent=2)
        torch.save(self.model.state_dict(), f'{self.path}/model.pth')

    def load(self) -> None:
        try:
            self.tokenizer = Tokenizer(json.loads(open(f'{self.path}/tokenizer.json', encoding='utf-8').read()))
            self.model_config = ModelConfig(**json.loads(open(f'{self.path}/config.json', encoding='utf-8').read()))
            self.generation_config = GenerationConfig(**json.loads(open(f'{self.path}/generation_config.json', encoding='utf-8').read()))
            self.model.load_state_dict(torch.load(f'{self.path}/model.pth'))
        except FileNotFoundError:
            pass

    def __repr__(self) -> str:
        return f'WorkSpace({self.name})'

def command() -> None:
    workspace = None
    while True:
        command = input('>>> ').lower().split()
        if not command:
            continue
        args, kwargs = _get_args(command)

        match command[0]:
            case 'set' | 'train' | 'generate' if not workspace:
                print('Error: Workspace not set')
                
            case 'workspace':
                workspace = WorkSpace(command[1])

            case 'set':
                workspace.config(**kwargs)

            case 'train':
                workspace.train(*args, **kwargs)

            case 'generate':
                out = workspace.generate(*args, **kwargs)
                if not kwargs.get('stream', False):
                    print(out)

            case 'q' | 'quit' | 'exit':
                if workspace:
                    del workspace
                break

def _get_args(command: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    args = []
    kwargs = {}
    for arg in command[1:]:
        if '=' not in arg:
            args.append(arg)
        else:
            kwarg = arg.split('=')
            kwargs[kwarg[0]] = kwarg[1]
    return args, kwargs