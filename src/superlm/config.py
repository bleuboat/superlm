from typing import Any, TypeVar, Iterator
T = TypeVar('T', bound='Config')

__all__ = ['ModelConfig', 'GenerationConfig', 'TrainingConfig', 'AdamConfig']

class Parameter:
    __slots__ = 'value', 'default'

    def __init__(self, value: Any, default: Any = None) -> None:
        self.value = value
        if callable(default):
            self.default = default
        else:
            self.default = lambda: default

    def get(self) -> Any:
        if self.value is not None:
            return self.value
        else:
            return self.default()

    def set(self, value: Any) -> None:
        if isinstance(value, Parameter):
            self.value = value.value
            self.default = value.default
        else:
            self.value = value

    def __repr__(self) -> str:
        return f'Parameter({self.get()})'

class Kwargs:
    __slots__ = 'kwargs'

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs.copy()

    def parameter(self, name: str, default: Any = None) -> Parameter:
        return Parameter(self.kwargs.pop(name, None), default)

class Config(dict[str, Parameter]):
    __slots__ = ()

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
        return f'{self.__class__.__name__}({', '.join(f'{k}={self[k]}' for k in self)})'

    def init(self, kwargs: Kwargs) -> None:
        return

    def check(self) -> None:
        return

    def to_dict(self) -> dict[str, Any]:
        return {k: self[k] for k in self}

    def copy(self: T) -> T:
        return type(self)(**self)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def update(self, /, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def _get_params(cls) -> None:
        cls.params = cls().keys()

class ModelConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.model_type = kwargs.parameter('model_type')
        self.vocab_size = kwargs.parameter('vocab_size')
        self.n_layer = kwargs.parameter('n_layer', 4)
        self.n_embd = kwargs.parameter('n_embd', 256)
        self.n_inner = kwargs.parameter('n_inner', lambda: self.n_embd * 4)
        self.n_head = kwargs.parameter('n_head', lambda: self.n_embd // 64)
        self.rope_theta = kwargs.parameter('rope_theta', 10000)
        self.dropout = kwargs.parameter('dropout', 0.1)
        self.eps = kwargs.parameter('eps', 1e-8)
        self.tie_word_embeddings = kwargs.parameter('tie_word_embeddings', False)

    def check(self) -> None:
        assert self.n_embd % self.n_head == 0

class GenerationConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.max_length = kwargs.parameter('max_length', 512)
        self.max_new_tokens = kwargs.parameter('max_new_tokens', None)
        self.max_time = kwargs.parameter('max_time', None)

        self.do_sample = kwargs.parameter('do_sample', True)
        self.temperature = kwargs.parameter('temperature', 1.0)
        self.top_k = kwargs.parameter('top_k', 50)
        self.top_p = kwargs.parameter('top_p', 1.0)
        self.repetition_penalty = kwargs.parameter('repetition_penalty', 1.0)

        self.pad_token_ix = kwargs.parameter('pad_token_ix', None)
        self.bos_token_ix = kwargs.parameter('bos_token_ix', None)
        self.eos_token_ix = kwargs.parameter('eos_token_ix', None)

class TrainingConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.epochs = kwargs.parameter('epochs', 1)
        self.block_size = kwargs.parameter('block_size', 32)
        self.batch_size = kwargs.parameter('batch_size', 32)
        self.accumulation_steps = kwargs.parameter('accumulation_steps', 1)

class AdamConfig(Config):
    def init(self, kwargs: Kwargs) -> None:
        self.lr = kwargs.parameter('lr', 1e-3)
        self.betas = kwargs.parameter('betas', (0.9, 0.999))
        self.eps = kwargs.parameter('eps', 1e-8)
        self.weight_decay = kwargs.parameter('weight_decay', 0.01)

ModelConfig._get_params()
GenerationConfig._get_params()
TrainingConfig._get_params()
AdamConfig._get_params()