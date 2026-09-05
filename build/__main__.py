print("loading pytorch...")

import torch as torch  # ruff: ignore[module-import-not-at-top-of-file, useless-import-alias]

for name in (
    "config",
    "dtypes",
    "init",
    "paths",
    "models",
    "logits_processor",
    "generation",
    "workspace",
    "pycache",
):
    print(f"building {name}...")
    __import__("", globals(), locals(), (f"build_{name}",), 1)
