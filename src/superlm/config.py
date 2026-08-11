from .activations import ACTIVATIONS
from typing import Any, Self
from collections.abc import Iterator, KeysView, ValuesView, ItemsView


__all__ = ["ModelConfig", "GenerationConfig", "TrainingConfig", "AdamConfig"]


class Parameter:
    __slots__ = ("default", "value")

    def __init__(self, value: Any, default: Any = None) -> None:
        self.value = value
        if callable(default):
            self.default = default
        else:
            self.default = lambda: default

    def get(self) -> Any:
        if self.value is not None:
            return self.value
        return self.default()

    def set(self, value: Any) -> None:
        if isinstance(value, Parameter):
            self.value = value.value
            self.default = value.default
        else:
            self.value = value

    def __repr__(self) -> str:
        return f"Parameter({self.get()})"


class Kwargs:
    __slots__ = ("kwargs",)

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs.copy()

    def parameter(self, name: str, default: Any = None) -> Any:
        return Parameter(self.kwargs.pop(name, None), default)


class Config(dict[str, Parameter]):
    def __init__(self, **kwargs: Any) -> None:
        new_kwargs = Kwargs(kwargs)
        self.init(new_kwargs)
        if len(new_kwargs.kwargs) > 0:
            raise KeyError(*new_kwargs.kwargs.keys())

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            super().__getitem__(key).set(value)
        elif isinstance(value, Parameter):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, Parameter(value))

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key).get()

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  {",\n  ".join(f"{k}={self[k]}" for k in self)},\n)"

    def init(self, kwargs: Kwargs) -> None:
        return

    def check(self) -> None:
        return

    def to_dict(self) -> dict[str, Any]:
        return {k: self[k] for k in self}

    def copy(self: Self) -> Self:
        return type(self)(**self)

    def keys(self) -> KeysView[str]:
        return self.to_dict().keys()

    def values(self) -> ValuesView[Any]:
        return self.to_dict().values()

    def items(self) -> ItemsView[str, Any]:
        return self.to_dict().items()

    def update(self, /, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self[k] = v


class ModelConfig(Config):
    def check(self) -> None:
        if self.model_type == "transformer":
            assert self.n_embd % self.n_head == 0, \
                f"embed_dim must be divisible by num_heads, got {self.n_embd} and {self.n_head}"
        assert self.hidden_act in ACTIVATIONS, f"unknown hidden activation '{self.hidden_act}'"

    def init(self, kwargs: Kwargs) -> None:
        self.model_type         : str                 = kwargs.parameter("model_type")
        self.vocab_size         : int                 = kwargs.parameter("vocab_size")

        self.n_layer            : int                 = kwargs.parameter("n_layer", 4)
        self.n_embd             : int                 = kwargs.parameter("n_embd", 256)
        self.n_inner            : int                 = kwargs.parameter("n_inner", lambda: self.n_embd * 4)
        self.n_head             : int                 = kwargs.parameter("n_head", lambda: self.n_embd // 64)
        self.n_kv_head          : int                 = kwargs.parameter("n_kv_head", lambda: self.n_head)
        self.block_size         : int                 = kwargs.parameter("block_size", 1024)
        self.rope_theta         : int                 = kwargs.parameter("rope_theta", 10000)
        self.dropout            : float               = kwargs.parameter("dropout", 0.1)
        self.eps                : float               = kwargs.parameter("eps", 1e-8)
        self.hidden_act         : str                 = kwargs.parameter("hidden_act", "silu")
        self.tie_word_embeddings: bool                = kwargs.parameter("tie_word_embeddings", False)

        self.pad_token_ix       : int | None          = kwargs.parameter("pad_token_ix", None)
        self.bos_token_ix       : int | None          = kwargs.parameter("bos_token_ix", None)
        self.eos_token_ix       : int | None          = kwargs.parameter("eos_token_ix", None)


class GenerationConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.max_length         : int | None          = kwargs.parameter("max_length", 512)
        self.max_new_tokens     : int | None          = kwargs.parameter("max_new_tokens", None)
        self.max_time           : int | float         = kwargs.parameter("max_time", None)

        self.do_sample          : bool                = kwargs.parameter("do_sample", True)
        self.temperature        : float               = kwargs.parameter("temperature", 1.0)
        self.top_k              : int                 = kwargs.parameter("top_k", 50)
        self.top_p              : float               = kwargs.parameter("top_p", 1.0)
        self.repetition_penalty : float               = kwargs.parameter("repetition_penalty", 1.0)

        self.pad_token_ix       : int | None          = kwargs.parameter("pad_token_ix", None)
        self.bos_token_ix       : int | None          = kwargs.parameter("bos_token_ix", None)
        self.eos_token_ix       : int | None          = kwargs.parameter("eos_token_ix", None)


class TrainingConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.epochs             : int                 = kwargs.parameter("epochs", 1)
        self.batch_size         : int                 = kwargs.parameter("batch_size", 1)
        self.accumulation_steps : int                 = kwargs.parameter("accumulation_steps", 1)


class AdamConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.lr                 : float               = kwargs.parameter("lr", 1e-3)
        self.betas              : tuple[float, float] = kwargs.parameter("betas", (0.9, 0.999))
        self.eps                : float               = kwargs.parameter("eps", 1e-8)
        self.weight_decay       : float               = kwargs.parameter("weight_decay", 0.01)
        self.amsgrad            : bool                = kwargs.parameter("amsgrad", False)
        self.maximize           : bool                = kwargs.parameter("maximize", False)
        self.foreach            : bool | None         = kwargs.parameter("foreach", None)
        self.capturable         : bool                = kwargs.parameter("capturable", False)
        self.differentiable     : bool                = kwargs.parameter("differentiable", False)
        self.fused              : bool | None         = kwargs.parameter("fused", None)
