from . import templater
from pathlib import Path
import re

file_config = Path("build") / ".config"
file_py = Path("src") / "superlm" / "config.py"

configs: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str, str]]]] = {}
last_config = None

with file_config.open() as f:
    while (line := f.readline()):
        if line == "\n":
            continue
        if line[0] == "$":
            last_config = None
            continue
        line = line[:-1]
        line = re.sub(r" +", " ", line)
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

config_raw = config_typing = ""

for config, data in configs.items():
    config_raw += f"\nclass {config}(Config):\n"
    config_raw += "    def init(self, kwargs: Kwargs) -> None:\n"
    for name, annotation, default in data[1]:
        origin = re.sub(r"\[.*?\]", "", annotation)
        config_raw += (
            f"""        self.{name} = kwargs.param(
            "{name}", {origin}, default={default},
        )
"""
        )
    if data[0]:
        config_raw += "\n    def check_extras(self, err: list[Exception]) -> None:\n"
        for check, message in data[0]:
            config_raw += f"""        if {check}:
            err.append(ValueError(
                f"{message}",
            ))
"""

    config_typing += "\n@dataclass\n"
    config_typing += f"class {config}(Config):\n"
    for name, annotation, default in data[1]:
        if "lambda" not in default:
            config_typing += f"    {name}: {annotation} = {default}\n"
        else:
            config_typing += f"    {name}: {annotation} = ...  # type: ignore\n"

all = str(list(configs.keys())).replace("'", '"')

templater.apply(file_py)
