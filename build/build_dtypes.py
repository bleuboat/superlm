import re
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("torch")
if not spec or not spec.origin:
    raise RuntimeError

torch = Path(spec.origin).parent
C = torch / "_C" / "__init__.pyi"
dtypes: list[str] = re.findall(r"^(.*?): dtype = ...$", C.read_text(), flags=re.MULTILINE)

code = "from torch import (\n"
all = "__all__ = [\n"
for dtype in dtypes:
    code += f"    {dtype},\n"
    all += f'    "{dtype}",\n'
code += ")\n"
all += "]\n"

code += all

dtypes_file = Path("src") / "superlm" / "dtypes.py"
dtypes_file.write_text(code)
