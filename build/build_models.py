import re
from pathlib import Path

from . import templater

path = Path("src") / "superlm"
ignore = {"__init__.py", "__init__.py.in", "utils.py"}

for name in ("architectures", "models"):
    folder = path / name
    pattern = re.compile(rf"^class (.*?)\({name[:-1]}", flags=re.MULTILINE | re.IGNORECASE)
    imports = ""
    all = ""
    for file in folder.iterdir():
        if file.is_file() and file.name not in ignore:
            file_name = file.stem
            file_classes: list[str] = pattern.findall(file.read_text("utf-8"))
            file_import = ", ".join(f"{file_class} as {file_class}" for file_class in file_classes)
            imports += f"from .{file_name} import {file_import}\n"
            for file_class in file_classes:
                all += f'"{file_class}",\n'
    templater.apply(folder / "__init__.py")
