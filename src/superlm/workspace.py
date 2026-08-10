import os
import json
import torch
import dotenv
import shutil
import safetensors.torch

from torch._prims_common import DeviceLikeType
from typing import Callable, Generator, Sequence

from .config    import *
from .models    import *
from .tokenizer import *
from .streamer  import *
from .trainer   import *

try:
    import matplotlib.pyplot as plt # type: ignore
except ModuleNotFoundError:
    plt = None

__all__ = ['WorkSpace']

def get_workspace_path() -> str:
    dotenv.load_dotenv()
    path = os.getenv('WORKSPACE_PATH')
    if path is None:
        path = 'workspace'
        dotenv.set_key('.env', 'WORKSPACE_PATH', 'workspace')
    return path

WORKSPACE_PATH = get_workspace_path()

class WorkSpace:
    name: str
    path: str
    paths: dict[str, str]
    dtype: torch.dtype
    device: torch.device
    model_config: ModelConfig
    generation_config: GenerationConfig
    training_config: TrainingConfig
    adam_config: AdamConfig

    def __init__(
        self,
        name: str,
        *,
        seed: int | None = None,
        dtype: torch.dtype | None = None,
        device: DeviceLikeType | None = None,
    ) -> None:
        self.name = name
        self.path = f'{WORKSPACE_PATH}/{name}'
        self.paths = {
                        'input': f'{self.path}/input.txt',
                       'inputs': f'{self.path}/inputs',
                   'checkpoint': f'{self.path}/checkpoint.pth',
                       'config': f'{self.path}/config.json',
            'generation_config': f'{self.path}/generation_config.json',
                         'loss': f'{self.path}/loss.png',
                        'model': f'{self.path}/model.safetensors',
                        'vocab': f'{self.path}/vocab.json',
        }

        if seed is not None:
            torch.manual_seed(seed)

        if dtype is not None:
            torch.set_default_dtype(dtype)
        else:
            torch.set_default_dtype(torch.bfloat16)

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
        except FileNotFoundError:
            os.makedirs(self.path)

    @property
    def inputs(self) -> dict[str, str]:
        if self._inputs is None:
            self.get_inputs()
        return self._inputs # type: ignore

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self.setup_tokenizer()
        return self._tokenizer # type: ignore

    @property
    def streamer(self) -> Streamer:
        if self._streamer is None:
            self.setup_streamer()
        return self._streamer # type: ignore

    @property
    def model(self) -> CausalLM:
        if self._model is None:
            self.setup_model()
        return self._model # type: ignore

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self.setup_trainer()
        return self._trainer # type: ignore

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer | None) -> None:
        self._tokenizer = value

    @streamer.setter
    def streamer(self, value: Streamer | None) -> None:
        self._streamer = value

    @model.setter
    def model(self, value: CausalLM | None) -> None:
        self._model = value

    @trainer.setter
    def trainer(self, value: Trainer | None) -> None:
        self._trainer = value

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_dtype(self, dtype: torch.dtype) -> None:
        torch.set_default_dtype(dtype)

    def set_device(self, device: DeviceLikeType) -> None:
        self.device = torch.device(device)
        if self._model is not None:
            self._model.to(self.device)

    def copy(self, other: WorkSpace | str) -> WorkSpace:
        if isinstance(other, str):
            other = WorkSpace(other)
        shutil.copytree(other.path, self.path, dirs_exist_ok=True)
        self.load()
        return self

    def clone(self, other: WorkSpace | str) -> WorkSpace:
        if isinstance(other, str):
            other = WorkSpace(other)
        other.copy(self)
        return other

    def config(self, **configs) -> None:
        for k, v in configs.items():
            for config in (self.model_config, self.generation_config, self.training_config, self.adam_config):
                if k in config.keys():
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
        self.show_losses(self.trainer.losses)

    def generate(self, inputs: str, *, stream: bool = False, **kwargs) -> str:
        self.check()
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=('bos',))
        if stream:
            streamer = self.streamer
        else:
            streamer = None
        res = self.model.generate(inputs_tensor, streamer=streamer, **kwargs)
        return self.tokenizer.decode(res)[0]

    def api(self, inputs: str, map: Callable[[str], str] | None = None, **kwargs) -> Generator[str]:
        self.check()
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=('bos',))
        for token in self.model.api(inputs_tensor, **kwargs):
            out = self.tokenizer.decode(token)[0]
            if map is None:
                yield out
            else:
                yield map(out)

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
        self.model = CausalLM(self.tokenizer, self.model_config, self.generation_config)
        self.model.to(self.device)

    def setup_trainer(self) -> None:
        self.trainer = Trainer(self.tokenizer, self.model, self.paths['checkpoint'], self.inputs.values(), self.training_config, self.adam_config)

    def save(self) -> None:
        with open(self.paths['vocab'], 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.tokens, f)
        with open(self.paths['config'], 'w', encoding='utf-8') as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        with open(self.paths['generation_config'], 'w', encoding='utf-8') as f:
            json.dump(self.model.generation_config.to_dict(), f, indent=2)
        safetensors.torch.save_model(self.model, self.paths['model'])

    def load(self) -> None:
        try:
            self.tokenizer = Tokenizer(json.loads(open(self.paths['vocab'], encoding='utf-8').read()))
            self.model_config = ModelConfig(**json.loads(open(self.paths['config'], encoding='utf-8').read()))
            self.generation_config = GenerationConfig(**json.loads(open(self.paths['generation_config'], encoding='utf-8').read()))
            try:
                safetensors.torch.load_model(self.model, self.paths['model'])
            except RuntimeError:
                self.setup_model()
        except FileNotFoundError:
            pass

    def info(self) -> None:
        print(f'----')
        print(f'模型大小：{self.model.num_parameters/1e6:.2f}M')
        print(f'数据大小：{self.trainer.dataset.length}')
        print(f'词表大小：{self.tokenizer.vocab_size}')

    def show_losses(self, losses: list[tuple[int, int]]) -> None:
        print('----')
        if plt is not None:
            x, y = zip(*losses)
            plt.plot(x, y)
            plt.xlabel('num_steps')
            plt.ylabel('loss')
            plt.grid(True)
            plt.savefig(self.paths['loss'], dpi=300)
            plt.show()

    def __repr__(self) -> str:
        return f'WorkSpace({self.name})'