import json
import torch
from safetensors.torch import save_model, load_model  # pyright: ignore[reportUnknownVariableType]

from typing import Unpack
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
        /, *,
        seed: int | None = None,
        dtype: str | None = None,
        device: str | None = None,
    ) -> None:
        self.set_seed(seed)
        self.set_dtype(dtype)
        self.set_device(device)

        self.paths = Paths(*path.split("/"))
        self.config = ConfigGroup()

        self._inputs = None
        self._tokenizer = None
        self._streamer = None
        self._model = None
        self._trainer = None

    @property
    def inputs(self) -> dict[str, str]:
        if self._inputs is None:
            self._inputs = self._get_inputs()
        return self._inputs

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = self._get_tokenizer()
        return self._tokenizer

    @property
    def streamer(self) -> Streamer:
        if self._streamer is None:
            self._streamer = self._get_streamer()
        return self._streamer

    @property
    def model(self) -> CausalLM:
        if self._model is None:
            self._model = self._get_model()
        return self._model

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self._trainer = self._get_trainer()
        return self._trainer

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

    def _get_inputs(self) -> dict[str, str]:
        if not self.paths.inputs.exists():
            raise FileNotFoundError(2, "input not found", self.paths.inputs)
        inputs: dict[str, str] = {}
        files = self.paths.inputs.glob("*.txt")
        for file in files:
            content = file.read_text(encoding="utf-8")
            inputs[file.stem] = content
        return inputs

    def _get_tokenizer(self) -> Tokenizer:
        return Tokenizer.from_data(self.inputs.values())

    def _get_streamer(self) -> Streamer:
        return Streamer(self.tokenizer)

    def _get_model(self) -> CausalLM:
        return CausalLM(
            self.tokenizer,
            self.config.model,
            self.config.generation,
        ).to(self.device)

    def _get_trainer(self) -> Trainer:
        return Trainer(
            self.tokenizer,
            self.model,
            self.paths.checkpoint,
            self.inputs.values(),
            self.config.training,
            self.config.adam,
        )

    @staticmethod
    def set_seed(seed: int | None) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)

    def set_dtype(self, dtype: str | None) -> None:
        if dtype is None:
            dtype = DEFAULT_DTYPE
        self.dtype = DTYPES[dtype]
        torch.set_default_dtype(self.dtype)

    def set_device(self, device: str | None) -> None:
        if device is None:
            device = DEFAULT_DEVICE
        self.device = torch.device(device)
        if getattr(self, "_model", None) is not None:
            self.model.to(self.device)

    def train(self, steps: Sequence[int] | None = None) -> None:
        self.save()
        self.model.train()
        self.trainer.train(steps)
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
        if not all((
            self.paths.vocab.exists(),
            self.paths.config.exists(),
            self.paths.generation_config.exists(),
            self.paths.model.exists(),
        )):
            return
        self.tokenizer = Tokenizer(
            json.loads(self.paths.vocab.read_text(encoding="utf-8")),
        )
        self.config.model = ModelConfig(
            **json.loads(self.paths.config.read_text()),
        )
        self.config.generation = GenerationConfig(
            **json.loads(self.paths.generation_config.read_text()),
        )
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
