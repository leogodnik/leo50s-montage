"""Frame-grid timeline: every duration and overlap is snapped to whole frames."""
from __future__ import annotations


def snap(x: float, fps: int) -> float:
    return round(x * fps) / fps


def frames(d: float, fps: int) -> int:
    return int(round(d * fps))


def timeline(edl, fps: int):
    """Return ([(segment, start)], total). Mutates d/ov onto the frame grid.

    A hard cut is a one-frame xfade overlap; the first segment never overlaps.
    """
    t = 0.0
    res = []
    for i, s in enumerate(edl):
        s.d = snap(s.d, fps)
        s.ov = 0.0 if i == 0 else (1 / fps if s.trans == "cut" else snap(s.ov, fps))
        start = t - s.ov
        res.append((s, start))
        t = start + s.d
    return res, t


def visual_start(seg, start: float) -> float:
    """Moment the segment becomes visually dominant (middle of the crossfade)."""
    return start if seg.trans == "cut" else start + seg.ov / 2
