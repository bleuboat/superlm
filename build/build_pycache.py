import py_compile
import sys
from pathlib import Path

version = "cpython-" + "".join(map(str, sys.version_info[0:2]))
for file in Path("src").glob("**/__pycache__/*.pyc"):
    file.unlink()
for file in Path("src").glob("**/*.py"):
    py_compile.compile(str(file))
