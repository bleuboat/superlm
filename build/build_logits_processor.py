import re
from pathlib import Path

from . import templater

file_origin = Path("build") / "vars" / "logits_processor.txt"
file_py = Path("src") / "superlm" / "generation" / "logits_processor.py"

classes: dict[str, tuple[list[str], list[str]]] = {}
last_class = None

class_pattern = re.compile(r"^([A-Za-z]*)\((.*)\):$", re.MULTILINE)

with file_origin.open(encoding="utf-8") as f:
    while (line := f.readline()):
        if line == "\n":
            continue
        line = line[:-1]
        if line[0] != " ":
            match = class_pattern.fullmatch(line)
            if match is None:
                raise RuntimeError
            name, params = match.groups()
            classes[name] = (params.split(", "), [])
            last_class = classes[name]
            continue
        if not last_class:
            raise RuntimeError
        last_class[1].append(line[4:])

processors = ""

for name, (params, code) in classes.items():
    processors += f"\n\nclass {name}(LogitsProcessor):\n"
    processors += f"    def __init__(self, {", ".join(params)}) -> None:\n"
    for param in params:
        param_name = param.split(": ")[0]
        processors += f"        self.{param_name} = {param_name}\n"
    processors += "\n    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:\n"
    code[-1] = "return " + code[-1]
    for line in code:
        processors += f"        {line}\n"

templater.apply(file_py)
