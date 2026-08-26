for name in (
    "build_requirements",
    "build_config",
    "build_dtypes",
    "build_init",
    "build_models",
    "build_paths",
    "build_pycache",
):
    print(f"building {name.split("_")[-1]}...")
    __import__("", globals(), locals(), (name,), 1)
