"""Looks: one ffmpeg filter chain per text file."""
from __future__ import annotations

from pathlib import Path

LOOKS_DIR = Path(__file__).resolve().parents[2] / "skills/match-grade/references/looks"


def resolve_look(name_or_path: str, base_dir=None) -> Path:
    """A plugin look by name, or a `.txt`/path; a relative path is taken from base_dir
    (the build folder for trailer.json, the current folder when base_dir is None)."""
    if "/" in name_or_path or name_or_path.endswith(".txt"):
        path = Path(name_or_path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        return path
    return LOOKS_DIR / f"{name_or_path}.txt"


def load_look(name_or_path: str, base_dir=None) -> str:
    path = resolve_look(name_or_path, base_dir)
    if not path.is_file():
        available = sorted(p.stem for p in LOOKS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"look not found: {name_or_path} ({path}; plugin looks: "
                                f"{', '.join(available)})")
    return path.read_text(encoding="utf-8").strip()
