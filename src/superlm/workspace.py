from __future__ import annotations

import os
import ast
import json
import torch
import dotenv

from torch._prims_common import DeviceLikeType
from typing import Any, Iterable, Sequence

from .config import *
from .tokenizer import Tokenizer
from .streamer import Streamer
from .models import Model
from .trainer import Trainer
from .generation import GenerationModule

try:
    import matplotlib.pyplot as plt # type: ignore
except ModuleNotFoundError:
    plt = None

__all__ = ['WorkSpace', 'command']

def get_workspace_path() -> str:
    dotenv.load_dotenv()
    path = os.getenv('WORKSPACE_PATH')
    if path is None:
        dotenv.set_key('.env', 'WORKSPACE_PATH', 'workspace')
        return get_workspace_path()
    return os.getenv('WORKSPACE_PATH')

WORKSPACE_PATH = get_workspace_path()

class WorkSpace:
    name: str
    path: str
    paths: dict[str, str]
    device: torch.device
    inputs: dict[str, str]
    tokenizer: Tokenizer
    streamer: Streamer
    model: GenerationModule
    trainer: Trainer
    model_config: ModelConfig
    generation_config: GenerationConfig
    training_config: TrainingConfig
    adam_config: AdamConfig

    def __init__(self, name: str, *, seed: int | None = None, device: DeviceLikeType | None = None) -> None:
        self.name = name
        self.path = f'{WORKSPACE_PATH}/{name}'
        self.paths = {
                        'input': f'{self.path}/input.txt',
                       'inputs': f'{self.path}/inputs',
                   'checkpoint': f'{self.path}/checkpoint.pth',
                       'config': f'{self.path}/config.json',
            'generation_config': f'{self.path}/generation_config.json',
                         'loss': f'{self.path}/loss.png',
                        'model': f'{self.path}/model.pth',
                        'vocab': f'{self.path}/vocab.json',
        }

        if seed is not None:
            torch.manual_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_config = ModelConfig()
        self.generation_config = GenerationConfig()
        self.training_config = TrainingConfig()
        self.adam_config = AdamConfig()

        self._inputs = None
        self._tokenizer = None
        self._streamer = None
        self._model = None
        self._trainer = None
        
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
    def streamer(self) -> Streamer:
        if self._streamer is None:
            self.setup_streamer()
        return self._streamer

    @property
    def model(self) -> GenerationModule:
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

    @streamer.setter
    def streamer(self, value: Streamer) -> None:
        self._streamer = value

    @model.setter
    def model(self, value: GenerationModule) -> None:
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

    def config(self, **configs) -> None:
        for k, v in configs.items():
            for config in (
                self.model_config,
                self.generation_config,
                self.training_config,
                self.adam_config,
            ):
                if k in config.keys():
                    if config is self.model_config:
                        self.model = None
                        self.del_checkpoint()
                    config[k] = v
                    break

    def check(self) -> None:
        for config in (
            self.model_config,
            self.generation_config,
            self.training_config,
            self.adam_config,
        ):
            config.check()
    
    def train(self, length: int | None = None, steps: Sequence[int] | None = None) -> None:
        self.check()
        self.save()
        self.model.train()
        trained = self.trainer.train(length, steps)
        self.model.load_state_dict(trained)
        self.model.eval()
        self.save()

        if plt is not None:
            x, y = zip(*self.trainer.losses)
            plt.plot(x, y)
            plt.xlabel('num_steps')
            plt.ylabel('loss')
            plt.grid(True)
            plt.savefig(self.paths['loss'], dpi=300)
            plt.show()

    def generate(self, inputs: str, *, stream: bool = False, **kwargs) -> str:
        self.check()
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=('bos'))
        if stream:
            streamer = self.streamer
        else:
            streamer = None
        res = self.model.generate(inputs_tensor, streamer=streamer, device=self.device, **kwargs)
        return self.tokenizer.decode(res)[0]

    def get_inputs(self) -> None:
        try:
            self._inputs = {'input': open(self.paths['input'], 'r', encoding='utf-8').read()}
        except FileNotFoundError:
            try:
                self._inputs = {}
                files = os.listdir(self.paths['inputs'])
                for file in files:
                    self._inputs[file.split('.')[0]] = open(f'{self.paths['inputs']}/{file}', 'r', encoding='utf-8').read()
            except FileNotFoundError:
                raise FileNotFoundError(f'Input not found. (at {self.path})')

    def setup_tokenizer(self) -> None:
        self.tokenizer = Tokenizer.from_data(self.inputs.values())

    def setup_streamer(self) -> None:
        self.streamer = Streamer(self.tokenizer)

    def setup_model(self) -> None:
        self.model = Model(self.tokenizer, self.model_config, self.generation_config)
        self.model.to(self.device)

    def setup_trainer(self) -> None:
        self.trainer = Trainer(self.tokenizer, self.model, self.device, self.paths['checkpoint'], self.inputs.values(), self.training_config, self.adam_config)

    def save(self) -> None:
        with open(self.paths['vocab'], 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.tokens, f)
        with open(self.paths['config'], 'w', encoding='utf-8') as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        with open(self.paths['generation_config'], 'w', encoding='utf-8') as f:
            json.dump(self.model.generation_config.to_dict(), f, indent=2)
        torch.save(self.model.state_dict(), self.paths['model'])

    def load(self) -> None:
        try:
            self.tokenizer = Tokenizer(json.loads(open(self.paths['vocab'], encoding='utf-8').read()))
            self.model_config = ModelConfig(**json.loads(open(self.paths['config'], encoding='utf-8').read()))
            self.generation_config = GenerationConfig(**json.loads(open(self.paths['generation_config'], encoding='utf-8').read()))
            try:
                self.model.load_state_dict(torch.load(self.paths['model']))
            except RuntimeError:
                self.setup_model()
        except FileNotFoundError:
            pass

    def info(self) -> None:
        print(f'----')
        print(f'模型大小：{sum(p.numel() for p in self.model.parameters())/1e6:.2f}M')
        print(f'数据大小：{self.trainer.dataset.length}')
        print(f'词表大小：{self.tokenizer.vocab_size}')

    def del_checkpoint(self) -> None:
        if os.path.exists(self.paths['checkpoint']):
            os.remove(self.paths['checkpoint'])

    def __repr__(self) -> str:
        return f'WorkSpace({self.name})'

def command() -> None:
    workspace = None
    while True:
        command = input('>>> ').lower().split()
        if not command:
            continue
        args, kwargs = _get_args(command[1:])

        match command[0]:
            case 'set' | 'config' | 'info' | 'train' | 'eval' | 'inference' | 'generate' if not workspace:
                print('Error: Workspace not set')

            case 'ws' | 'work' | 'space' | 'workspace':
                workspace = WorkSpace(*args, **kwargs)

            case 'set' | 'config':
                workspace.config(*args, **kwargs)

            case 'info':
                workspace.info(*args, **kwargs)

            case 'train':
                workspace.train(*args, **kwargs)

            case 'eval' | 'inference' | 'generate':
                kwargs['stream'] = True
                workspace.generate(*args, **kwargs)

            case 'q' | 'quit' | 'exit':
                if workspace:
                    del workspace
                break

def _get_args(params: Iterable[str]) -> tuple[list[Any], dict[str, Any]]:
    args = []
    kwargs = {}
    for arg in params:
        if '=' not in arg:
            args.append(arg)
        else:
            kwarg = arg.split('=')
            kwargs[kwarg[0]] = kwarg[1]
    for i in range(len(args)):
        args[i] = ast.literal_eval(args[i])
    for key in kwargs:
        kwargs[key] = ast.literal_eval(kwargs[key])
    return args, kwargs