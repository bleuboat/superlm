import re
from pathlib import Path

import torch

from . import templater

file = Path("src") / "superlm" / "workspace.py"
template = Path("src") / "superlm" / "workspace.py.in"

default_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
default_device = "cuda" if torch.cuda.is_available() else "cpu"

attrs: list[tuple[str, str]] = re.findall(
    r"^ +def _get_([a-z]+)\(self\) -> (.*?):$",
    template.read_text("utf-8"),
    re.MULTILINE,
)

init = properties = ""

for name, annotation in attrs:
    init += f"self._{name} = None\n"
    properties += f"""
@property
def {name}(self) -> {annotation}:
    if self._{name} is None:
        self._{name} = self._get_{name}()
    return self._{name}

@{name}.setter
def {name}(self, value: {annotation} | None) -> None:
    self._{name} = value
"""

templater.apply(file)
