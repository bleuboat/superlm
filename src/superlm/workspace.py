import json
import torch
from safetensors.torch import save_model, load_model  # pyright: ignore[reportUnknownVariableType]

from typing import Any, cast
from collections.abc import Callable, Generator, Sequence

from .paths import *
from .config import *
from .dtypes import *
from .architectures import *
from .tokenizer import *
from .streamer import *
from .trainer import *

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


__all__ = ["WorkSpace"]

DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WorkSpace:
    def __init__(
        self,
        path: str,
        *,
        seed: int | None = None,
        dtype: str = DEFAULT_DTYPE,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.paths = Paths(*path.split("/"))

        if seed is not None:
            torch.manual_seed(seed)

        self.set_dtype(dtype)
        self.set_device(device)

        self.model_config = ModelConfig()
        self.generation_config = GenerationConfig()
        self.training_config = TrainingConfig()
        self.adam_config = AdamConfig()

        self._inputs = None
        self._tokenizer = None
        self._streamer = None
        self._model = None
        self._trainer = None

        self.paths.root.mkdir(exist_ok=True)

    @property
    def inputs(self) -> dict[str, str]:
        if self._inputs is None:
            self.get_inputs()
        return cast(dict[str, str], self._inputs)

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self.setup_tokenizer()
        return cast(Tokenizer, self._tokenizer)

    @property
    def streamer(self) -> Streamer:
        if self._streamer is None:
            self.setup_streamer()
        return cast(Streamer, self._streamer)

    @property
    def model(self) -> CausalLM:
        if self._model is None:
            self.setup_model()
        return cast(CausalLM, self._model)

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self.setup_trainer()
        return cast(Trainer, self._trainer)

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

    def set_dtype(self, dtype: str) -> None:
        self.dtype = DTYPES[dtype]
        torch.set_default_dtype(self.dtype)

    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        if getattr(self, "_model", None) is not None:
            self.model.to(self.device)

    def set_model_name(self, model_name: str) -> None:
        self.paths.model_name = model_name

    def copy(self, other: WorkSpace | str) -> WorkSpace:
        import shutil  # ruff: ignore[import-outside-top-level]
        if isinstance(other, str):
            other = WorkSpace(other)
        shutil.copytree(other.paths.root, self.paths.root, dirs_exist_ok=True)
        self.load()
        return self

    def clone(self, other: WorkSpace | str) -> WorkSpace:
        if isinstance(other, str):
            other = WorkSpace(other)
        other.copy(self)
        return other

    def config(self, **configs: Any) -> None:
        for k, v in configs.items():
            for config in (
                self.model_config, self.generation_config, self.training_config, self.adam_config,
            ):
                if k in config:
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

    def train(self, steps: Sequence[int] | None = None) -> None:
        self.check()
        self.save()
        self.model.train()
        trained = self.trainer.train(steps)
        self.model.load_state_dict(trained)
        self.model.eval()
        self.save()
        self.show_losses(self.trainer.losses)

    def generate(self, inputs: str, *, stream: bool = False, **kwargs: Any) -> str:
        self.check()
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=("bos",))
        streamer = self.streamer if stream else None
        res = self.model.generate(inputs_tensor, streamer=streamer, **kwargs)
        return self.tokenizer.decode(res)[0]

    def api(
        self,
        inputs: str,
        *,
        map: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> Generator[str]:
        self.check()
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=("bos",))
        for token in self.model.api(inputs_tensor, **kwargs):
            out = self.tokenizer.decode(token)[0]
            if map is None:
                yield out
            else:
                yield map(out)

    def get_inputs(self) -> None:
        if not self.paths.inputs.exists():
            raise FileNotFoundError(2, "input not found", self.paths.inputs)

        inputs: dict[str, str] = {}
        files = self.paths.inputs.glob("*.txt")
        for file in files:
            content = file.read_text(encoding="utf-8")
            inputs[file.stem] = content

        self._inputs = inputs

    def setup_tokenizer(self) -> None:
        self.tokenizer = Tokenizer.from_data(self.inputs.values())

    def setup_streamer(self) -> None:
        self.streamer = Streamer(self.tokenizer)

    def setup_model(self) -> None:
        self.model = CausalLM(self.tokenizer, self.model_config, self.generation_config)
        self.model.to(self.device)

    def setup_trainer(self) -> None:
        self.trainer = Trainer(
            self.tokenizer,
            self.model,
            self.paths.checkpoint,
            self.inputs.values(),
            self.training_config,
            self.adam_config,
        )

    def save(self) -> None:
        self.paths.model_root.mkdir(exist_ok=True)
        self.paths.vocab.write_text(
            json.dumps(self.tokenizer.tokens), encoding="utf-8",
        )
        self.paths.config.write_text(
            json.dumps(dict(self.model.config), indent=2),
        )
        self.paths.generation_config.write_text(
            json.dumps(dict(self.model.generation_config), indent=2),
        )
        save_model(self.model, str(self.paths.model))

    def load(self) -> None:
        if self.paths.vocab.exists():
            self.tokenizer = Tokenizer(
                json.loads(self.paths.vocab.read_text(encoding="utf-8")),
            )
        if self.paths.config.exists():
            self.model_config = ModelConfig(
                **json.loads(self.paths.config.read_text()),
            )
        if self.paths.generation_config.exists():
            self.generation_config = GenerationConfig(
                **json.loads(self.paths.generation_config.read_text()),
            )
        if self.paths.model.exists():
            load_model(self.model, self.paths.model)

    def info(self) -> None:
        print("----")
        print(f"model parameters: {self.model.num_parameters / 1e6:.2f}M")
        print(f"data size: {self.trainer.dataset.length}")
        print(f"vocab size: {self.tokenizer.vocab_size}")

    def show_losses(self, losses: list[tuple[int, float]]) -> None:
        print("----")
        if plt is not None:
            x, y = zip(*losses, strict=True)
            plt.plot(x, y)
            plt.xlabel("num_steps")
            plt.ylabel("loss")
            plt.grid(visible=True)
            plt.savefig(self.paths.loss, dpi=300)
            plt.show()

    def __repr__(self) -> str:
        return f"WorkSpace({self.paths.space_name}/{self.paths.model_name})"
