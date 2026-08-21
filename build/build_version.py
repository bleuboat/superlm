from . import templater
from tomllib import loads
from pathlib import Path

pyproject_file = Path("pyproject.toml")
version_file = Path("src") / "superlm" / "version.py"
pyproject = loads(pyproject_file.read_text("utf-8"))
version = pyproject["project"]["version"]
templater.apply(version_file, vars())
