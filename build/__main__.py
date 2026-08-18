from pathlib import Path
scripts = Path(__file__).parent.glob("build_*.py")
for script in scripts:
    exec(script.read_text())  # ruff: ignore[exec-builtin]
