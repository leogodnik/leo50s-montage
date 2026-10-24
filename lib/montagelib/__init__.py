import os
from pathlib import Path

VERSION = "0.1.0"


def home() -> Path:
    return Path(os.environ.get("LEO50S_HOME", "~/.cache/leo50s-montage")).expanduser()
