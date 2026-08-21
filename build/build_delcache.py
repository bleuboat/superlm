from pathlib import Path
for cache in Path("src").glob("**/__pycache__"):
    cache.rmdir()
