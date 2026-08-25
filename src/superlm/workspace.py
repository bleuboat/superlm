import json
import torch
from safetensors.torch import save_model, load_model  # pyright: ignore[reportUnknownVariableType]

from typing import Unpack, cast
from collections.abc import Callable, Generator, Sequence

from .architectures import *
from .config import *
from .dtypes import DTYPES
from .paths import Paths
from .tokenizer import Tokenizer
from .streamer import Streamer
from .trainer import Trainer

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


__all__ = ["WorkSpace"]

DEFAULT_DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WorkSpace:
    def __init__(
        self,
        path: str,
        *,
        seed: str | int | None = None,
        dtype: str = DEFAULT_DTYPE,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.set_seed(seed)
        self.set_dtype(dtype)
        self.set_device(device)

        self.paths = Paths(*path.split("/"))
        self.paths.root.mkdir(exist_ok=True)
        self.config = ConfigGroup()

        self._inputs = None
        self._tokenizer = None
        self._streamer = None
        self._model = None
        self._trainer = None

    @property
    def inputs(self) -> dict[str, str]:
        if self._inputs is None:
            self._get_inputs()
        return cast(dict[str, str], self._inputs)

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._setup_tokenizer()
        return cast(Tokenizer, self._tokenizer)

    @property
    def streamer(self) -> Streamer:
        if self._streamer is None:
            self._setup_streamer()
        return cast(Streamer, self._streamer)

    @property
    def model(self) -> CausalLM:
        if self._model is None:
            self._setup_model()
        return cast(CausalLM, self._model)

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self._setup_trainer()
        return cast(Trainer, self._trainer)

    @inputs.setter
    def inputs(self, value: dict[str, str] | None) -> None:
        self._inputs = value

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

    def _get_inputs(self) -> None:
        if not self.paths.inputs.exists():
            raise FileNotFoundError(2, "input not found", self.paths.inputs)

        inputs: dict[str, str] = {}
        files = self.paths.inputs.glob("*.txt")
        for file in files:
            content = file.read_text(encoding="utf-8")
            inputs[file.stem] = content

        self.inputs = inputs

    def _setup_tokenizer(self) -> None:
        self.tokenizer = Tokenizer.from_data(self.inputs.values())

    def _setup_streamer(self) -> None:
        self.streamer = Streamer(self.tokenizer)

    def _setup_model(self) -> None:
        self.model = CausalLM(
            self.tokenizer,
            self.config.model,
            self.config.generation,
        ).to(self.device)

    def _setup_trainer(self) -> None:
        self.trainer = Trainer(
            self.tokenizer,
            self.model,
            self.paths.checkpoint,
            self.inputs.values(),
            self.config.training,
            self.config.adam,
        )

    @staticmethod
    def set_seed(seed: str | int | None) -> None:
        if seed is not None:
            torch.manual_seed(seed)

    def set_dtype(self, dtype: str) -> None:
        self.dtype = DTYPES[dtype]
        torch.set_default_dtype(self.dtype)

    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        if getattr(self, "_model", None) is not None:
            self.model.to(self.device)

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

    def train(self, steps: Sequence[int] | None = None) -> None:
        self.save()
        self.model.train()
        trained = self.trainer.train(steps)
        self.model.load_state_dict(trained)
        self.model.eval()
        self.save()
        self.show_losses(self.trainer.losses)

    def generate(
        self,
        inputs: str,
        *,
        stream: bool = False,
        **kwargs: Unpack[GenerationConfigTypedDict],
    ) -> str:
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=("bos",))
        streamer = self.streamer if stream else None
        out = self.model.generate(inputs_tensor, streamer=streamer, **kwargs)
        return self.tokenizer.decode(out)[0]

    def api(
        self,
        inputs: str,
        *,
        map: Callable[[str], str] | None = None,
        **kwargs: Unpack[GenerationConfigTypedDict],
    ) -> Generator[str]:
        inputs_tensor = self.tokenizer([inputs], device=self.device, special_tokens=("bos",))
        for token in self.model.api(inputs_tensor, **kwargs):
            out = self.tokenizer.decode(token)[0]
            if map is None:
                yield out
            else:
                yield map(out)

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
            self.config.model = ModelConfig(
                **json.loads(self.paths.config.read_text()),
            )
        if self.paths.generation_config.exists():
            self.config.generation = GenerationConfig(
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
