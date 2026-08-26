from pathlib import Path
from tomllib import loads

from . import templater

pyproject_file = Path("pyproject.toml")
init_file = Path("src") / "superlm" / "__init__.py"
pyproject = loads(pyproject_file.read_text("utf-8"))
description = pyproject["project"]["description"]
version = pyproject["project"]["version"]
templater.apply(init_file)
