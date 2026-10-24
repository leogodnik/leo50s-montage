"""Score from trailer.json: instrument layers, hits, clip ambience, ducking, mastering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .project import Project
from .synth import (EVENTS, INSTRUMENTS, SR, add, env_asr, expdec, lp, noise, read_wav,
                    reverb, riser, whistle)
from .timeline import timeline, visual_start

# riser is 1/3 of the original instrument level (synth.LOUD), so 0.9 here is the original 0.3
DEFAULT_HIT_GAINS = {"boom": 0.75, "bigboom": 1.0, "braam": 0.7, "hit": 0.55, "riser": 0.9,
                     "titleboom": 0.7, "explosion": 1.0}
DEFAULT_MASTER = {"speech_rms": 0.09, "ambience_default": 0.35, "target_peak": 0.92,
                  "fade_out": 1.2}
DUCK_RAMP = 0.25


def _layer(music, layer, T):
    t0 = T[layer["from"]]
    d = T[layer["to"]] - layer.get("trim_end", 0.0) - t0
    if d <= 0:
        return
    x = INSTRUMENTS[layer["instrument"]](d, **layer.get("params", {}))
    x = x * env_asr(len(x), layer.get("attack", 3.0), layer.get("release", 0.4))
    if "ramp" in layer:
        pts = sorted((T[sid], g) for sid, g in layer["ramp"])
        tt = t0 + np.arange(len(x)) / SR
        x = x * np.interp(tt, [t for t, _ in pts], [g for _, g in pts])
    else:
        x = x * layer.get("gain", 1.0)
    add(music, x, t0)


def _hit(music, hits, seg, start, T, gains):
    h = seg.hit
    kind = h["type"]
    g = h["gain"] if h["gain"] is not None else gains.get(kind, DEFAULT_HIT_GAINS[kind])
    t0 = T[seg.id] + h["offset"]
    if kind == "riser":
        add(music, riser(seg.d - 0.15), t0, g)
    elif kind == "braam":
        add(hits, EVENTS["braam"](), t0 - 0.15, g)
        add(hits, EVENTS["boom"](), t0, 0.5)
    elif kind == "explosion":
        tx = start + 0.3  # the moment the frame goes white
        if h["whistle_from"] is not None:
            ws = T[h["whistle_from"]] + 0.1
            if tx - 0.1 > ws:
                add(hits, whistle(tx - 0.1 - ws), ws, 0.55 * g)
        add(hits, EVENTS["bigboom"](), tx, g)
        add(hits, lp(noise(1.6), 1800) * expdec(1.6, 0.35), tx, 0.9 * g)
    else:
        add(hits, EVENTS[kind](), t0, g)


def _ambience(amb, p, seg, start, master):
    path = p.wav_dir / f"{seg.src}.wav"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 'build.py <trailer.json> wav' first")
    a = read_wav(path)
    i0 = int(seg.ss * SR)
    x = a[i0:i0 + int(seg.d * SR)].copy()
    if not len(x):
        return
    x *= env_asr(len(x), 0.03, 0.08)[:, None]
    x = np.stack([lp(x[:, 0], 7000), lp(x[:, 1], 7000)], 1)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    x *= min(master["speech_rms"] / rms, 8.0)
    gain = seg.amb if seg.amb is not None else (1.0 if seg.speech else master["ambience_default"])
    i = int(start * SR)
    m = min(len(x), len(amb) - i)
    if m > 0:
        amb[i:i + m] += x[:m] * gain


def _duck(n, tl):
    duck = np.ones(n)
    r = int(DUCK_RAMP * SR)
    for s, st in tl:
        if not s.speech:
            continue
        g = s.duck
        i0, i1 = int(st * SR), min(int((st + s.d) * SR), n)
        duck[i0:i1] = np.minimum(duck[i0:i1], g)
        a = max(i0 - r, 0)
        duck[a:i0] = np.minimum(duck[a:i0], np.linspace(1, g, r)[r - (i0 - a):])
        b = min(i1 + r, n)
        duck[i1:b] = np.minimum(duck[i1:b], np.linspace(g, 1, r)[:b - i1])
    return duck


def build_audio(p: Project) -> Path:
    tl, total = timeline(p.edl, p.fps)
    master = {**DEFAULT_MASTER, **p.score.get("master", {})}
    n = int((total + 1.0) * SR)
    music, hits, amb = np.zeros(n), np.zeros(n), np.zeros((n, 2))
    T = {s.id: visual_start(s, st) for s, st in tl}
    T["end"] = total + 0.5

    for layer in p.score.get("layers", []):
        _layer(music, layer, T)
    gains = p.score.get("hits", {})
    for s, st in tl:
        if s.hit:
            _hit(music, hits, s, st, T, gains)
    for s, st in tl:
        if s.kind == "clip" and not any(f.startswith("slow") for f in s.fx):
            _ambience(amb, p, s, st, master)

    music *= _duck(n, tl)
    mono = music + reverb(hits)
    # widen slightly: a short opposite delay on each side
    left, right = mono + 0.08 * np.roll(mono, 240), mono + 0.08 * np.roll(mono, -240)
    mix = np.stack([left, right], 1) + amb
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix /= peak
        mix = np.tanh(mix * 0.85) / np.tanh(0.85)
        mix *= master["target_peak"] / np.max(np.abs(mix))
    mix *= env_asr(n, 0.0, master["fade_out"])[:, None]

    p.out_dir.mkdir(parents=True, exist_ok=True)
    out = p.out_dir / "score.wav"
    wavfile.write(str(out), SR, (mix * 32767).astype(np.int16))
    print(f"audio written: {out}, length {n / SR:.1f} s, peak {np.max(np.abs(mix)):.3f}")
    return out
