"""Shared command line of the match-grade scripts: a trailer.json or --clips alias=path."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .grade import read_grade
from .project import Project, load_project


@dataclass
class Sources:
    """Clips to grade, their look and per-clip (gamma, contrast); `project` is None in clip mode."""
    clips: dict[str, Path]
    look: str
    grade: dict[str, tuple[float, float]] = field(default_factory=dict)
    project: Project | None = None


def add_source_args(ap: argparse.ArgumentParser, grade=True) -> None:
    ap.add_argument("project", nargs="?", type=Path)
    ap.add_argument("--clips", nargs="+", metavar="ALIAS=PATH")
    ap.add_argument("--look", help="look name or .txt (default: the project's, or noir)")
    if grade:
        ap.add_argument("--grade", type=Path,
                        help="alias gamma contrast file (default: the project's, or 1.0/1.0)")


def pairs(items, what) -> dict[str, str]:
    out = {}
    for item in items or []:
        alias, sep, value = item.partition("=")
        if not sep or not alias or not value:
            raise ValueError(f"{what}: expected alias=value, got {item!r}")
        out[alias] = value
    return out


def contrasts(items) -> dict[str, float]:
    out = {}
    for alias, value in pairs(items, "--contrast").items():
        try:
            c = float(value)
        except ValueError:
            c = 0.0
        if not c > 0:
            raise ValueError(f"--contrast {alias}: contrast must be a positive number, got {value!r}")
        out[alias] = c
    return out


def sources(ap: argparse.ArgumentParser, a: argparse.Namespace) -> Sources:
    """Resolve parsed args; raises ProjectError, ValueError, OSError on bad input."""
    if (a.project is None) == (a.clips is None):
        ap.error("give either <trailer.json> or --clips")
    gpath = getattr(a, "grade", None)
    if a.project is not None:
        p = load_project(a.project)
        return Sources(dict(p.clips), a.look or p.look,
                       read_grade(gpath) if gpath else dict(p.grade), p)
    clips = {k: Path(v) for k, v in pairs(a.clips, "--clips").items()}
    return Sources(clips, a.look or "noir", read_grade(gpath) if gpath else {})
