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

dtypes_dict = "".join(f'"{dtype}": torch.{dtype},\n' for dtype in dtypes)
dtypes_file = Path("src") / "superlm" / "dtypes.py"
templater.apply(dtypes_file)
