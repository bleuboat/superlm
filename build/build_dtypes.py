from . import templater
from importlib.util import find_spec
from pathlib import Path
import re

spec = find_spec("torch")
if not spec or not spec.origin:
    raise RuntimeError

torch = Path(spec.origin).parent
C = torch / "_C" / "__init__.pyi"
dtypes: list[str] = re.findall(r"^(.*?): dtype = ...$", C.read_text(), flags=re.MULTILINE)

torch_list = all_list = ""
for dtype in dtypes:
    torch_list += f"{dtype},\n"
    all_list += f'"{dtype}",\n'

dtypes_file = Path("src") / "superlm" / "dtypes.py"
templater.apply(dtypes_file, vars())
