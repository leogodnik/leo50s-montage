"""Looks: one ffmpeg filter chain per text file."""
from __future__ import annotations

from pathlib import Path

LOOKS_DIR = Path(__file__).resolve().parents[2] / "skills/match-grade/references/looks"


def load_look(name_or_path: str) -> str:
    if "/" in name_or_path or name_or_path.endswith(".txt"):
        path = Path(name_or_path)
    else:
        path = LOOKS_DIR / f"{name_or_path}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in LOOKS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"look not found: {name_or_path} "
                                f"(available: {', '.join(available)})")
    return path.read_text(encoding="utf-8").strip()
