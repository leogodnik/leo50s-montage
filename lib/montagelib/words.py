"""Word timestamps (whisper) and how they sit against segment boundaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    w: str
    s: float
    e: float


def load_words(path) -> dict[str, list[Word]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {alias: [Word(str(x["w"]), float(x["s"]), float(x["e"])) for x in ws]
            for alias, ws in data.items()}


def inside(ws: list[Word], a: float, b: float, tol: float = 0.03) -> list[Word]:
    return [w for w in ws if w.s >= a - tol and w.e <= b + tol]


def straddling(ws: list[Word], a: float, b: float, tol: float = 0.03) -> list[Word]:
    return [w for w in ws if w.s + tol < a < w.e - tol or w.s + tol < b < w.e - tol]
