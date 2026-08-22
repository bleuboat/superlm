from pathlib import Path
scripts = Path(__file__).parent.glob("build_*.py")
for script in scripts:
    print(f"building {script.stem.split("_")[-1]}...")
    exec(script.read_text())  # ruff: ignore[exec-builtin]
