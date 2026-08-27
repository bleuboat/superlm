import re
from pathlib import Path

import torch

from . import templater

deprecated = re.compile(r"bit|u?int[1-7]")

dtypes = (
    name
    for name in dir(torch)
    if isinstance(getattr(torch, name), torch.dtype)
    and not deprecated.fullmatch(name)
)
dtypes_dict = "".join(f'"{dtype}": torch.{dtype},\n' for dtype in dtypes)
dtypes_file = Path("src") / "superlm" / "dtypes.py"
templater.apply(dtypes_file)
