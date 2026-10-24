"""trailer.json schema: dataclasses and hand-written validation (all problems at once)."""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .grade import parse_grade
from .looks import load_look, resolve_look

KINDS = ("clip", "card", "title", "black", "white")
TRANSITIONS = ("fade", "fadewhite", "cut")
STYLES = ("plain", "painted")
BGS = ("black", "clip")
HIT_TYPES = ("boom", "bigboom", "braam", "hit", "riser", "explosion", "titleboom")
FX = ("push", "slow2", "slow4", "shake")
DEFAULT_FONTS = {
    "card": "/System/Library/Fonts/HelveticaNeue.ttc#1",
    "title": "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
}
DEFAULT_OUTPUT = {"name": "trailer", "crf": 16, "preview_height": 720}


class ProjectError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("trailer.json: %d problem(s):\n- " % len(self.problems)
                         + "\n- ".join(self.problems))


@dataclass
class Segment:
    id: str
    kind: Literal["clip", "card", "title", "black", "white"]
    d: float
    ov: float = 0
    trans: Literal["fade", "fadewhite", "cut"] = "fade"
    src: Optional[str] = None
    ss: float = 0
    fx: list[str] = field(default_factory=list)
    shake: float = 1.0
    speech: bool = False
    duck: float = 0.5
    amb: Optional[float] = None
    hit: Optional[dict] = None
    text: Optional[str] = None
    lines: Optional[list[tuple[str, int]]] = None
    small: bool = False
    style: Literal["plain", "painted"] = "plain"
    bg: Literal["black", "clip"] = "black"
    bg_src: Optional[str] = None
    fade_in: float = 0


@dataclass
class Project:
    path: Path
    build_dir: Path
    clips: dict[str, Path]
    fps: int
    width: int
    height: int
    letterbox: Optional[int]
    look: str
    grade: dict[str, tuple[float, float]]
    fonts: dict[str, str]
    edl: list[Segment]
    score: dict
    output: dict

    @property
    def seg_dir(self) -> Path:
        return self.build_dir / "seg"

    @property
    def cards_dir(self) -> Path:
        return self.build_dir / "cards"

    @property
    def out_dir(self) -> Path:
        return self.build_dir / "out"

    @property
    def wav_dir(self) -> Path:
        return self.build_dir / "wav"

    def by_id(self, id: str) -> Segment:
        for s in self.edl:
            if s.id == id:
                return s
        raise KeyError(id)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _font_exists(spec: str) -> bool:
    return Path(spec.split("#", 1)[0]).exists()


def _parse_grade(path: Path, problems: list[str]) -> dict[str, tuple[float, float]]:
    if not path.exists():
        problems.append(f"grade: file not found: {path}")
        return {}
    grade, bad = parse_grade(path.read_text(encoding="utf-8"), path.name)
    problems.extend(bad)
    return grade


def _parse_hit(sid: str, raw, ids: set, problems: list[str]) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"type": raw}
    if not isinstance(raw, dict):
        problems.append(f"edl {sid}: hit must be a string or an object, got {raw!r}")
        return None
    t = raw.get("type")
    if t not in HIT_TYPES:
        problems.append(f"edl {sid}: unknown hit.type {t!r} (allowed: {' '.join(HIT_TYPES)})")
    wf = raw.get("whistle_from")
    if wf is not None and wf not in ids:
        problems.append(f"edl {sid}: hit.whistle_from {wf!r} is not a segment id")
    gain = raw.get("gain")
    if gain is not None and not _num(gain):
        problems.append(f"edl {sid}: hit.gain must be a number, got {gain!r}")
        gain = None
    offset = raw.get("offset", 0)
    if not _num(offset):
        problems.append(f"edl {sid}: hit.offset must be a number, got {offset!r}")
        offset = 0
    return {"type": t, "gain": None if gain is None else float(gain),
            "offset": offset, "whistle_from": wf}


def _parse_segment(i: int, e, aliases: set, ids: set, problems: list[str]) -> Optional[Segment]:
    if not isinstance(e, dict):
        problems.append(f"edl[{i}]: must be an object")
        return None
    sid = e.get("id")
    if not isinstance(sid, str) or not sid:
        problems.append(f"edl[{i}]: missing id")
        sid = f"#{i}"
    label = f"edl {sid}"
    kind = e.get("kind")
    if kind not in KINDS:
        problems.append(f"{label}: unknown kind {kind!r} (allowed: {' '.join(KINDS)})")
    d = e.get("d")
    if not _num(d) or d <= 0:
        problems.append(f"{label}: d must be a positive number, got {d!r}")
        d = 0.0
    trans = e.get("trans", "fade")
    if trans not in TRANSITIONS:
        problems.append(f"{label}: unknown trans {trans!r} (allowed: {' '.join(TRANSITIONS)})")
    style = e.get("style", "plain")
    if style not in STYLES:
        problems.append(f"{label}: unknown style {style!r}")
    bg = e.get("bg", "black")
    if bg not in BGS:
        problems.append(f"{label}: unknown bg {bg!r}")
    src = e.get("src")
    if kind == "clip" and src is None:
        problems.append(f"{label}: clip needs src")
    if src is not None and src not in aliases:
        problems.append(f"{label}: unknown src alias {src!r}")
    bg_src = e.get("bg_src")
    if bg_src is not None and bg_src not in aliases:
        problems.append(f"{label}: unknown bg_src alias {bg_src!r}")
    fx = e.get("fx", [])
    if isinstance(fx, str):
        fx = [fx]
    if not isinstance(fx, list):
        problems.append(f"{label}: fx must be a list of strings")
        fx = []
    for f in fx:
        if f not in FX:
            problems.append(f"{label}: unknown fx {f!r} (allowed: {' '.join(FX)})")
    nums = {}
    for key, default in (("ov", 0), ("ss", 0), ("shake", 1.0), ("duck", 0.5),
                         ("amb", None), ("fade_in", 0)):
        v = e.get(key, default)
        if not _num(v) and not (key == "amb" and v is None):
            problems.append(f"{label}: {key} must be a number, got {v!r}")
            v = default
        nums[key] = v
    if _num(d) and d > 0 and not (0 <= nums["ov"] < d):
        problems.append(f"{label}: ov must satisfy 0 <= ov < d (ov={nums['ov']!r}, d={d!r})")
    lines = e.get("lines")
    if lines is not None:
        try:
            lines = [(str(t), int(sz)) for t, sz in lines]
        except (TypeError, ValueError):
            problems.append(f"{label}: lines must be a list of [text, size]")
            lines = None
    return Segment(
        id=sid, kind=kind, d=d, ov=nums["ov"], trans=trans, src=src,
        ss=nums["ss"], fx=list(fx), shake=nums["shake"],
        speech=bool(e.get("speech", False)), duck=nums["duck"],
        amb=nums["amb"], hit=_parse_hit(sid, e.get("hit"), ids, problems),
        text=e.get("text"), lines=lines, small=bool(e.get("small", False)),
        style=style, bg=bg, bg_src=bg_src, fade_in=nums["fade_in"])


def _check_score(score, ids: set, problems: list[str]) -> None:
    from .synth import INSTRUMENTS  # lazy: loading a project should not need scipy otherwise
    if not isinstance(score, dict):
        problems.append("score: must be an object")
        return
    layers = score.get("layers", [])
    if not isinstance(layers, list):
        problems.append(f"score.layers: must be a list, got {layers!r}")
        layers = []
    for n, layer in enumerate(layers):
        if not isinstance(layer, dict):
            problems.append(f"score.layers[{n}]: must be an object")
            continue
        inst = layer.get("instrument")
        if inst not in INSTRUMENTS:
            problems.append(f"score.layers[{n}]: unknown instrument {inst!r} "
                            f"(allowed: {' '.join(INSTRUMENTS)})")
        fr, to = layer.get("from"), layer.get("to")
        if fr not in ids:
            problems.append(f"score.layers[{n}]: from {fr!r} is not a segment id")
        if to != "end" and to not in ids:
            problems.append(f"score.layers[{n}]: to {to!r} is not a segment id or 'end'")
        if "ramp" in layer:
            ramp = layer["ramp"]
            if not isinstance(ramp, list) or not ramp:
                problems.append(f"score.layers[{n}]: ramp must be a non-empty list of [id, gain], "
                                f"got {ramp!r}")
            else:
                for pt in ramp:
                    if not (isinstance(pt, list) and len(pt) == 2 and _num(pt[1])):
                        problems.append(f"score.layers[{n}]: ramp point must be [id, gain], got {pt!r}")
                    elif pt[0] not in ids:
                        problems.append(f"score.layers[{n}]: ramp id {pt[0]!r} is not a segment id")
        params = layer.get("params", {})
        if not isinstance(params, dict):
            problems.append(f"score.layers[{n}]: params must be an object, got {params!r}")
        elif inst in INSTRUMENTS:
            allowed = list(inspect.signature(INSTRUMENTS[inst]).parameters)[1:]  # after d
            for k in params:
                if k not in allowed:
                    problems.append(f"score.layers[{n}]: {inst} has no param {k!r} "
                                    f"(allowed: {' '.join(allowed) or 'none'})")
    hits = score.get("hits", {})
    if not isinstance(hits, dict):
        problems.append("score.hits: must be an object {type: gain}")
        hits = {}
    for t in hits:
        if t not in HIT_TYPES:
            problems.append(f"score.hits: unknown hit type {t!r}")


def load_project(path: Path) -> Project:
    path = Path(path).resolve()
    base = path.parent
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise ProjectError([f"cannot read {path}: {ex}"])
    if not isinstance(data, dict):
        raise ProjectError(["top level must be an object"])
    problems: list[str] = []

    raw_clips = data.get("clips", {})
    if not isinstance(raw_clips, dict):
        problems.append("clips: must be an object {alias: path}")
        raw_clips = {}
    clips = {}
    for a, p in raw_clips.items():
        if isinstance(p, str) and p:
            clips[a] = base / p
        else:
            problems.append(f"clips.{a}: path must be a string, got {p!r}")

    fps = data.get("fps", 24)
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        problems.append(f"fps: must be a positive integer, got {fps!r}")
        fps = 24
    size = data.get("size", [1920, 1080])
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)):
        problems.append(f"size: must be [width, height], got {size!r}")
        size = [1920, 1080]
    letterbox = data.get("letterbox")
    if letterbox is not None and not (isinstance(letterbox, int) and letterbox >= 0):
        problems.append(f"letterbox: must be a non-negative integer or null, got {letterbox!r}")
        letterbox = None

    grade_file = data.get("grade")
    grade = _parse_grade(base / grade_file, problems) if grade_file else {}

    raw_fonts = data.get("fonts") or {}
    if not isinstance(raw_fonts, dict):
        problems.append(f"fonts: must be an object, got {raw_fonts!r}")
        raw_fonts = {}
    fonts = {**DEFAULT_FONTS, **raw_fonts}
    for role, spec in fonts.items():
        if not isinstance(spec, str):
            problems.append(f"fonts.{role}: must be a string, got {spec!r}")
        elif not _font_exists(spec):
            problems.append(f"fonts.{role}: font file not found: {spec}")

    raw_output = data.get("output") or {}
    if not isinstance(raw_output, dict):
        problems.append(f"output: must be an object, got {raw_output!r}")
        raw_output = {}
    output = {**DEFAULT_OUTPUT, **raw_output}

    raw_edl = data.get("edl", [])
    if not isinstance(raw_edl, list) or not raw_edl:
        problems.append("edl: must be a non-empty list")
        raw_edl = []
    ids = {e["id"] for e in raw_edl if isinstance(e, dict) and isinstance(e.get("id"), str)}
    seen = set()
    for e in raw_edl:
        if isinstance(e, dict) and e.get("id") in seen:
            problems.append(f"edl: duplicate id {e['id']!r}")
        if isinstance(e, dict):
            seen.add(e.get("id"))
    aliases = set(raw_clips)
    edl = [s for i, e in enumerate(raw_edl)
           if (s := _parse_segment(i, e, aliases, ids, problems)) is not None]

    look = data.get("look", "noir")
    if not isinstance(look, str) or not look:
        problems.append(f"look: must be a look name or a path to a .txt, got {look!r}")
        look = "noir"
    else:
        try:
            load_look(look, base)
            if "/" in look or look.endswith(".txt"):
                look = str(resolve_look(look, base))  # absolute: works from any cwd
        except FileNotFoundError as ex:
            problems.append(f"look: {ex}")

    score = data.get("score", {})
    _check_score(score, ids, problems)

    if problems:
        raise ProjectError(problems)
    return Project(path=path, build_dir=base, clips=clips, fps=fps, width=size[0],
                   height=size[1], letterbox=letterbox, look=look,
                   grade=grade, fonts=fonts, edl=edl, score=score, output=output)
