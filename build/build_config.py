from . import templater
from pathlib import Path
import re

file_config = Path("build") / ".config"
file_py = Path("src") / "superlm" / "config.py"
file_pyi = Path("src") / "superlm" / "config.pyi"

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

code = all = ""

for config, data in configs.items():
    code += f"\n\nclass {config}TypedDict(TypedDict):\n"
    for name, annotation, _ in data[1]:
        if not name.endswith("_ix"):
            code += f"    {name}: NotRequired[{annotation}]\n"

    code += f"\n\nclass {config}(Config):\n"
    for name, annotation, _ in data[1]:
        code += (
            f"    {name}: {annotation}\n"
        )
    code += f"\n    def __init__(self, **kwargs: Unpack[{config}TypedDict]) -> None:\n"
    code += "        super().__init__()\n"
    for name, annotation, default in data[1]:
        origin = re.sub(r"\[.*?\]", "", annotation)
        code += (
            f'        self.add_param("{name}", {origin}, default={default})\n'
        )
    code += "        self.update(kwargs)\n"
    if data[0]:
        code += "\n    def _check_extras(self) -> None:\n"
        for check, message in data[0]:
            code += f"""        if {check}:
            raise ValueError(
                f"{message}",
            )
"""

    all += f'"{config}",\n'
    all += f'"{config}TypedDict",\n'

code += "\n\nclass AllConfigTypedDict(TypedDict):\n"
all += '"AllConfigTypedDict",\n'

for data in configs.values():
    for name, annotation, _ in data[1]:
        if not name.endswith("_ix"):
            code += f"    {name}: NotRequired[{annotation}]\n"

code += "\n\nclass ConfigGroup(ConfigGroupBase):\n"
code += "    def __init__(self) -> None:\n"
code += "        super().__init__()\n"

for config in configs:
    code += f"        self.{config[:-6].lower()} = {config}()\n"

all += '"ConfigGroup",\n'

templater.apply(file_py)
