from pathlib import Path
import re


code = """from .activations import ACTIVATIONS
from typing import Any, Self
from collections.abc import Iterator, KeysView, ValuesView, ItemsView, MutableMapping

class Parameter:
    __slots__ = ("annotation", "default", "value")

    def __init__(self, value: Any, annotation: type, default: Any) -> None:
        self.value = value
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
        return f"{type(self).__name__}(\\n  {",\\n  ".join(f"{k}={self[k]}" for k in self)},\\n)"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({", ".join(f"{k}={self[k]}" for k in self)})"

    def init(self, kwargs: Kwargs) -> None:
        return

    def check_params(self, err: list[Exception]) -> None:
        for name, param in self._data.items():
            if not isinstance(param.get(), param.annotation):
                err.append(TypeError(
                    f"'{name}' must be a {param.annotation.__name__}, "
                    f"got {type(param.get()).__name__}",
                ))

    def check_extras(self, err: list[Exception]) -> None:
        return

    def check(self) -> None:
        err: list[Exception] = []
        self.check_params(err)
        self.check_extras(err)
        if len(err) == 1:
            raise err[0]
        if err:
            raise ExceptionGroup("ConfigError", err)

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
"""


typing = """from dataclasses import dataclass
from typing import Any, Self
from collections.abc import Iterator, KeysView, ValuesView, ItemsView, MutableMapping

class Config(MutableMapping[str, Any]):
    _data: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        super().__setattr__("_data", {})
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
        return f"{type(self).__name__}(\\n  {",\\n  ".join(f"{k}={self[k]}" for k in self)},\\n)"

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
"""


file_txt = Path("src") / "superlm" / ".config"
file_py = Path("src") / "superlm" / "config.py"

configs: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str, str]]]] = {}
last_config = None

with file_txt.open() as f:
    while (line := f.readline()):
        if line == "\n":
            continue
        line = line[:-1]
        line = re.sub(r" +", " ", line)
        if line[0] == "$":
            last_config = None
            continue
        if line[0] != " ":
            name = line[:-1]
            configs[name] = ([], [])
            last_config = configs[name]
            continue
        if not last_config:
            continue
        line = line[1:]
        if line.startswith("!check"):
            last_config[0].append((
                line.replace("!check ", ""),
                f.readline().strip(),
            ))
            continue
        match = re.fullmatch(r"(.*?): (.*?) = (.*?)", line)
        if match is None:
            raise RuntimeError
        last_config[1].append(match.groups())  # type: ignore

for config, data in configs.items():
    code += f"\nclass {config}(Config):\n"
    code += "    def init(self, kwargs: Kwargs) -> None:\n"
    for name, annotation, default in data[1]:
        origin = re.sub(r"\[.*?\]", "", annotation)
        code += (
            f'        self.{name} = kwargs.param("{name}", {origin}, default={default})\n'
        )
    if data[0]:
        code += "\n    def check_extras(self, err: list[Exception]) -> None:\n"
        for check, message in data[0]:
            code += f"        if {check}:\n"
            code += f"""            err.append(ValueError(
                f"{message}",
            ))
"""

    typing += "\n@dataclass\n"
    typing += f"class {config}(Config):\n"
    for name, annotation, default in data[1]:
        if "lambda" not in default:
            typing += f"    {name}: {annotation} = {default}\n"
        else:
            typing += f"    {name}: {annotation} = ...  # type: ignore\n"


def indent(code: str) -> str:
    return re.sub(r"^([^\n])", r"    \1", code, flags=re.MULTILINE)


final_code = f"""from typing import TYPE_CHECKING

if TYPE_CHECKING:
{indent(typing)}
else:
{indent(code)}
"""

final_code += f"\n__all__ = {list(configs.keys())}\n".replace("'", '"')

file_py.write_text(final_code)
