from pathlib import Path
import sys
import py_compile
version = "cpython-" + "".join(map(str, sys.version_info[0:2]))
for file in Path("src").glob("**/__pycache__/*.pyc"):
    file.unlink()
for file in Path("src").glob("**/*.py"):
    py_compile.compile(str(file))
