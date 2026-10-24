"""Per-clip gamma/contrast before the look: luma measurement, solver, grade file I/O."""
from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path

import numpy as np

from .ffm import FfmpegError
from .looks import load_look

GAMMA_MIN, GAMMA_MAX = 0.1, 10.0  # ffmpeg eq's gamma range


def measure(src: Path, look: str, gamma: float, contrast: float,
            sample_fps=4) -> tuple[float, float]:
    """Mean and std of the gray luma (0..255) after eq=gamma:contrast and the look."""
    vf = f"eq=gamma={gamma}:contrast={contrast},{load_look(look)},scale=192:108,fps={sample_fps}"
    cmd = (f"ffmpeg -v error -i {shlex.quote(str(src))} -vf \"{vf}\" "
           f"-f rawvideo -pix_fmt gray -")
    r = subprocess.run(cmd, shell=True, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise FfmpegError(cmd, r.stderr.decode(errors="replace")[-2000:])
    a = np.frombuffer(r.stdout, np.uint8).astype(float)
    return float(a.mean()), float(a.std())


def _loglum(m: float) -> float:
    """ln(-ln(mean/255)): eq gamma g scales -ln(mean/255) by roughly g^-k."""
    return math.log(-math.log(min(max(m, 1.0), 254.0) / 255))


def solve(clips: dict[str, Path], look: str, target_mean=82.0,
          target_std: float | None = None, contrast_groups: dict[str, float] | None = None,
          iters=4) -> dict[str, tuple[float, float]]:
    """Per-clip (gamma, contrast) before the look.

    Contrast goes toward target_std (default: mean std of the clips at 1.0/1.0) unless fixed
    by contrast_groups. Gamma follows g *= log(mean/255) / log(target/255), raised to 1/k
    where k is the exponent measured from the previous gamma step (k = 1 on the first step,
    which is the plain formula): the look's curve flattens the response, and with k = 1 a
    dark or bright clip would still be far off after four steps.
    """
    groups = contrast_groups or {}
    start = {a: measure(src, look, 1.0, 1.0) for a, src in clips.items()}
    if target_std is None:
        target_std = float(np.mean([s for _, s in start.values()]))
    goal = _loglum(target_mean)
    result = {}
    for alias, src in clips.items():
        g, k = 1.0, 1.0
        if alias in groups:
            c = float(groups[alias])
            m, s = measure(src, look, g, c)
        else:
            c = 1.0
            m, s = start[alias]
        for _ in range(iters):
            if alias not in groups:
                c = round(c * target_std / max(s, 1e-6), 3)
                m, s = measure(src, look, g, c)
            g0, m0 = g, m
            g = round(min(max(g * math.exp((_loglum(m) - goal) / k), GAMMA_MIN), GAMMA_MAX), 3)
            m, s = measure(src, look, g, c)
            step = math.log(g) - math.log(g0)
            if abs(step) > 0.02:
                k = min(max(-(_loglum(m) - _loglum(m0)) / step, 0.2), 5.0)
        result[alias] = (round(g, 3), round(c, 3))
    return result


def parse_grade(text: str, name: str = "grade") -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Parse 'alias gamma [contrast]' lines; returns (grade, problems)."""
    grade, problems = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        try:
            if len(parts) not in (2, 3):
                raise ValueError
            grade[parts[0]] = (float(parts[1]), float(parts[2]) if len(parts) == 3 else 1.0)
        except ValueError:
            problems.append(f"grade: {name} line {n}: expected 'alias gamma [contrast]', got {line!r}")
    return grade, problems


def read_grade(path) -> dict[str, tuple[float, float]]:
    path = Path(path)
    grade, problems = parse_grade(path.read_text(encoding="utf-8"), path.name)
    if problems:
        raise ValueError("\n".join(problems))
    return grade


def write_grade(path, result: dict[str, tuple[float, float]]) -> Path:
    path = Path(path)
    path.write_text("".join(f"{a} {g} {c}\n" for a, (g, c) in sorted(result.items())),
                    encoding="utf-8")
    return path
