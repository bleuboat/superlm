from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
    from typing import Any, Self
    from collections.abc import Iterator, KeysView, ValuesView, ItemsView, MutableMapping

    class Config(MutableMapping[str, Any]):
        _data: dict[str, Any]

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        def __setitem__(self, key: str, value: Any) -> None:
            self._data[key] = value

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __delitem__(self, key: str) -> Any:
            raise NotImplementedError("don't delete config's items")

        def __setattr__(self, key: str, value: Any) -> None:
            self._data[key] = value

        def __getattr__(self, key: str) -> Any:
            return self._data[key]

        def __len__(self) -> int:
            return self._data.__len__()

        def __iter__(self) -> Iterator[str]:
            return self._data.__iter__()

        def __contains__(self, key: Any) -> bool:
            return self._data.__contains__(key)

        def __str__(self) -> str:
            return f"{type(self).__name__}(\n  {",\n  ".join(f"{k}={self[k]}" for k in self)},\n)"

        def __repr__(self) -> str:
            return f"{type(self).__name__}({", ".join(f"{k}={self[k]}" for k in self)})"

        def check(self) -> None:
            return

        def copy(self: Self) -> Self:
            return type(self)(**self)

        def keys(self) -> KeysView[str]:
            return self._data.keys()

        def values(self) -> ValuesView[Any]:
            return self._data.values()

        def items(self) -> ItemsView[str, Any]:
            return self._data.items()

        def update(self, other: Any = None, /, **kwargs: Any) -> None:
            if other:
                raise NotImplementedError("'other' argument is not implemented")
            for k, v in kwargs.items():
                self[k] = v

    @dataclass
    class ModelConfig(Config):
        architecture: str = "causallm"
        model_type: str = "transformer"
        vocab_size: int = 0
        num_labels: int = 0
        n_layer: int = 4
        n_embd: int = 256
        n_inner: int = ...  # pyright: ignore[reportAssignmentType]
        n_head: int = ...  # pyright: ignore[reportAssignmentType]
        n_kv_head: int = ...  # pyright: ignore[reportAssignmentType]
        block_size: int = 1024
        rope_theta: int = 10000
        dropout: float = 0.1
        bias: bool = False
        eps: float = 1e-8
        hidden_act: str = "silu"
        tie_word_embeddings: bool = False
        pad_token_ix: int | None = None
        bos_token_ix: int | None = None
        eos_token_ix: int | None = None

    @dataclass
    class GenerationConfig(Config):
        max_length: int | None = 512
        max_new_tokens: int | None = None
        max_time: int | float | None = None
        do_sample: bool = True
        temperature: float = 1.0
        top_k: int = 50
        top_p: float = 1.0
        repetition_penalty: float = 1.0
        pad_token_ix: int | None = None
        bos_token_ix: int | None = None
        eos_token_ix: int | None = None

    @dataclass
    class TrainingConfig(Config):
        epochs: int = 1
        batch_size: int = 1
        accumulation_steps: int = 1

    @dataclass
    class AdamConfig(Config):
        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0.01
        amsgrad: bool = False
        maximize: bool = False
        foreach: bool | None = None
        capturable: bool = False
        differentiable: bool = False
        fused: bool | None = None

else:
    from .activations import ACTIVATIONS
    from types import GenericAlias
    from typing import Any, Self, get_origin
    from collections.abc import Iterator, KeysView, ValuesView, ItemsView, MutableMapping

    class Parameter:
        __slots__ = ("annotation", "default", "value")

        def __init__(self, value: Any, annotation: type, default: Any) -> None:
            self.value = value
            if isinstance(annotation, GenericAlias):
                self.annotation = get_origin(annotation)
            else:
                self.annotation = annotation
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

        def param(self, name: str, annotation: type, default: Any) -> Any:
            return Parameter(self.kwargs.pop(name, None), annotation, default)

    class Config(MutableMapping[str, Any]):
        __slots__ = ("_data",)
        _data: dict[str, Parameter]

        def __init__(self, **kwargs: Any) -> None:
            super().__setattr__("_data", {})
            new_kwargs = Kwargs(kwargs)
            self.init(new_kwargs)
            if len(new_kwargs.kwargs) > 0:
                raise KeyError(*new_kwargs.kwargs.keys())

        def __setitem__(self, key: str, value: Any) -> None:
            if key in self:
                self._data[key].set(value)
            elif isinstance(value, Parameter):
                self._data[key] = value
            else:
                self._data[key] = Parameter(value, object, None)

        def __getitem__(self, key: str) -> Any:
            return self._data[key].get()

        def __delitem__(self, key: str) -> Any:
            raise NotImplementedError("don't delete config's items")

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

        def __getattr__(self, key: str) -> Any:
            return self[key]

        def __len__(self) -> int:
            return self._data.__len__()

        def __iter__(self) -> Iterator[str]:
            return self._data.__iter__()

        def __contains__(self, key: Any) -> bool:
            return self._data.__contains__(key)

        def __str__(self) -> str:
            return f"{type(self).__name__}(\n  {",\n  ".join(f"{k}={self[k]}" for k in self)},\n)"

        def __repr__(self) -> str:
            return f"{type(self).__name__}({", ".join(f"{k}={self[k]}" for k in self)})"

        def init(self, kwargs: Kwargs) -> None:
            return

        def check_params(self, err: list[Exception]) -> None:
            for name, param in self._data.items():
                if not isinstance(param.get(), param.annotation):
                    err.append(TypeError(
                        f"'{name}' must be a {param.annotation.__name__}, got {type(param.get())}",
                    ))

        def check_extras(self, err: list[Exception]) -> None:
            return

        def check(self) -> None:
            err: list[Exception] = []
            self.check_params(err)
            self.check_extras(err)
            if err:
                raise ExceptionGroup("Error:", err)

        def copy(self: Self) -> Self:
            return type(self)(**self)

        def keys(self) -> KeysView[str]:
            return self._data.keys()

        def values(self) -> ValuesView[Any]:
            return self._data.values()

        def items(self) -> ItemsView[str, Any]:
            return self._data.items()

        def update(self, other: Any = None, /, **kwargs: Any) -> None:
            if other:
                raise NotImplementedError("'other' argument is not implemented")
            for k, v in kwargs.items():
                self[k] = v

    class ModelConfig(Config):
        def init(self, kwargs: Kwargs) -> None:
            self.architecture = kwargs.param("architecture", str, default="causallm")
            self.model_type = kwargs.param("model_type", str, default="transformer")
            self.vocab_size = kwargs.param("vocab_size", int, default=0)
            self.num_labels = kwargs.param("num_labels", int, default=0)
            self.n_layer = kwargs.param("n_layer", int, default=4)
            self.n_embd = kwargs.param("n_embd", int, default=256)
            self.n_inner = kwargs.param("n_inner", int, default=lambda: self.n_embd * 4)
            self.n_head = kwargs.param("n_head", int, default=lambda: self.n_embd // 64)
            self.n_kv_head = kwargs.param("n_kv_head", int, default=lambda: self.n_head)
            self.block_size = kwargs.param("block_size", int, default=1024)
            self.rope_theta = kwargs.param("rope_theta", int, default=10000)
            self.dropout = kwargs.param("dropout", float, default=0.1)
            self.bias = kwargs.param("bias", bool, default=False)
            self.eps = kwargs.param("eps", float, default=1e-8)
            self.hidden_act = kwargs.param("hidden_act", str, default="silu")
            self.tie_word_embeddings = kwargs.param("tie_word_embeddings", bool, default=False)
            self.pad_token_ix = kwargs.param("pad_token_ix", int | None, default=None)
            self.bos_token_ix = kwargs.param("bos_token_ix", int | None, default=None)
            self.eos_token_ix = kwargs.param("eos_token_ix", int | None, default=None)

        def check_extras(self, err: list[Exception]) -> None:
            if self.model_type == "transformer" and self.n_embd % self.n_head != 0:
                err.append(ValueError(
                    f"n_embd must be divisible by n_head, got {self.n_embd} and {self.n_head}",
                ))
            if self.hidden_act not in ACTIVATIONS:
                err.append(ValueError(
                    f"unknown hidden activation '{self.hidden_act}'",
                ))

    class GenerationConfig(Config):
        def init(self, kwargs: Kwargs) -> None:
            self.max_length = kwargs.param("max_length", int | None, default=512)
            self.max_new_tokens = kwargs.param("max_new_tokens", int | None, default=None)
            self.max_time = kwargs.param("max_time", int | float | None, default=None)
            self.do_sample = kwargs.param("do_sample", bool, default=True)
            self.temperature = kwargs.param("temperature", float, default=1.0)
            self.top_k = kwargs.param("top_k", int, default=50)
            self.top_p = kwargs.param("top_p", float, default=1.0)
            self.repetition_penalty = kwargs.param("repetition_penalty", float, default=1.0)
            self.pad_token_ix = kwargs.param("pad_token_ix", int | None, default=None)
            self.bos_token_ix = kwargs.param("bos_token_ix", int | None, default=None)
            self.eos_token_ix = kwargs.param("eos_token_ix", int | None, default=None)

    class TrainingConfig(Config):
        def init(self, kwargs: Kwargs) -> None:
            self.epochs = kwargs.param("epochs", int, default=1)
            self.batch_size = kwargs.param("batch_size", int, default=1)
            self.accumulation_steps = kwargs.param("accumulation_steps", int, default=1)

    class AdamConfig(Config):
        def init(self, kwargs: Kwargs) -> None:
            self.lr = kwargs.param("lr", float, default=1e-3)
            self.betas = kwargs.param("betas", tuple[float, float], default=(0.9, 0.999))
            self.eps = kwargs.param("eps", float, default=1e-8)
            self.weight_decay = kwargs.param("weight_decay", float, default=0.01)
            self.amsgrad = kwargs.param("amsgrad", bool, default=False)
            self.maximize = kwargs.param("maximize", bool, default=False)
            self.foreach = kwargs.param("foreach", bool | None, default=None)
            self.capturable = kwargs.param("capturable", bool, default=False)
            self.differentiable = kwargs.param("differentiable", bool, default=False)
            self.fused = kwargs.param("fused", bool | None, default=None)


__all__ = ["ModelConfig", "GenerationConfig", "TrainingConfig", "AdamConfig"]
