from . import templater
from pathlib import Path
import re


file_config = Path("build") / ".config"
file_py = Path("src") / "superlm" / "paths.py"
paths: list[tuple[str, str]] = []
code = ""

with file_config.open() as f:
    while True:
        line = f.readline()
        if line[0] == "$":
            break
    while (line := f.readline()):
        if line == "\n":
            continue
        if line[0] != " ":
            break
        line = line[:-1]
        line = re.sub(r" +", " ", line).strip()
        key, value = line.split(" = ")
        value = re.sub(r"/ (.*?)$", r'/ "\1"', value, flags=re.MULTILINE)
        code += f"self.{key} = root / {value}\n"

templater.apply(file_py, vars())
