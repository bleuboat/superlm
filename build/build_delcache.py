from pathlib import Path
for cache in Path("src").glob("**/__pycache__"):
    for file in cache.iterdir():
        file.unlink()
    cache.rmdir()
