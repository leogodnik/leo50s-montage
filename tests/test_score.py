import json

import numpy as np
import pytest
from scipy.io import wavfile

from montagelib import ffm
from montagelib.project import load_project
from montagelib.score import build_audio
from montagelib.timeline import timeline, visual_start


def _prepared(p, mutate=None):
    """Extract clip audio into wav/, optionally edit trailer.json, reload."""
    for alias, src in p.clips.items():
        ffm.extract_wav(src, p.wav_dir / f"{alias}.wav")
    if mutate:
        data = json.loads(p.path.read_text(encoding="utf-8"))
        mutate(data)
        p.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_project(p.path)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def _seg(p, sid):
    tl, _ = timeline(p.edl, p.fps)
    return next((s, st) for s, st in tl if s.id == sid)


def test_score_length(mini_project):
    p = _prepared(mini_project)
    out = build_audio(p)
    assert out == p.out_dir / "score.wav"
    sr, mix = wavfile.read(out)
    assert sr == 48000 and mix.ndim == 2 and mix.shape[1] == 2 and mix.dtype == np.int16
    _, total = timeline(p.edl, p.fps)
    assert abs(len(mix) - round((total + 1.0) * 48000)) <= 480


def test_ducking_under_speech(mini_project):
    def m(d):
        # short attack: the window before s02 must already carry the full drone
        d["score"]["layers"] = [{"instrument": "drone", "from": "s00", "to": "s04",
                                 "gain": 0.3, "attack": 0.1}]
        ed = {e["id"]: e for e in d["edl"]}
        ed["s02"].update(duck=0.2, amb=0)
        ed["s03"].update(amb=0)
    p = _prepared(mini_project, m)
    sr, mix = wavfile.read(build_audio(p))
    mix = mix.astype(float) / 32768
    seg, start = _seg(p, "s02")
    inside = mix[int((start + 0.3) * sr):int((start + seg.d - 0.3) * sr)]
    before = mix[int((start - 1.0) * sr):int(start * sr)]
    drop = 20 * np.log10(_rms(before) / _rms(inside))
    assert drop >= 10, f"music only {drop:.1f} dB lower under speech"


def test_hit_lands_on_visual_start(mini_project):
    def m(d):
        d["score"]["layers"] = []
        for e in d["edl"]:
            if e["kind"] == "clip":
                e["amb"] = 0
    p = _prepared(mini_project, m)
    sr, mix = wavfile.read(build_audio(p))
    seg, start = _seg(p, "s03")
    level = np.abs(mix.astype(float)).max(1)
    # the reverb tail of a low boom sums to its loudest sample ~0.2 s later, so the hit is where
    # it first reaches half the peak (its attack), which is what the eye syncs to
    hit_t = int(np.argmax(level >= 0.5 * level.max())) / sr
    assert abs(hit_t - visual_start(seg, start)) <= 0.040


def test_peak_limited(mini_project):
    p = _prepared(mini_project)
    _, mix = wavfile.read(build_audio(p))
    assert np.max(np.abs(mix.astype(float) / 32768)) <= 0.93


def test_missing_wav_names_the_wav_step(mini_project):
    with pytest.raises(FileNotFoundError, match=r"build\.py .* wav"):
        build_audio(mini_project)
