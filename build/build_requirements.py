from pathlib import Path
from tomllib import loads

pyproject_file = Path("pyproject.toml")
pyproject = loads(pyproject_file.read_text("utf-8"))
requirements = "\n".join(pyproject["project"]["dependencies"])
Path("requirements.txt").write_text(requirements, "utf-8")
