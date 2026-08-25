from pathlib import Path
import sys
version = "cpython-" + "".join(map(str, sys.version_info[0:2]))
for file in Path("src").glob("**/__pycache__/*.pyc"):
    stem = file.stem
    dot = stem.find(".")
    file_name = stem[:dot]
    file_info = stem[dot + 1:]
    if file_info != version:
        file.unlink()
        continue
    if not (file.parent.parent / f"{file_name}.py").exists():
        file.unlink()
        continue
