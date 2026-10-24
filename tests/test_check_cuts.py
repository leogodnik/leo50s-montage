import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from montagelib import ffm

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "skills" / "speech-safe-cuts" / "scripts" / "check_cuts.py"
MINI_WORDS = Path(__file__).parent / "fixtures" / "words-mini.json"


@pytest.fixture
def speech_project(mini_project):
    p = mini_project
    clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", 2.0, tone_hz=440,
                              bursts=[(0.5, 0.8), (1.2, 1.5)])
    ffm.extract_wav(clip, p.wav_dir / "a.wav")
    shutil.copy(MINI_WORDS, p.build_dir / "words.json")
    return p


def _edit(p, **seg_fields):
    data = json.loads(p.path.read_text(encoding="utf-8"))
    for e in data["edl"]:
        e.update(seg_fields.get(e["id"], {}))
    p.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _check(p):
    return subprocess.run([sys.executable, str(CHECK), str(p.path)],
                          capture_output=True, text=True)


def test_clean_project_passes(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.9})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "один" in r.stdout and "два" in r.stdout
    assert "problems: 0" in r.stdout


def test_cut_word_fails(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0.6, "d": 1.3})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "CUT WORD" in r.stdout


def test_tail_word_requires_hard_cut(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "fade"})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "needs cut" in r.stdout
    assert "end not quiet" in r.stdout
    # the hard cut waives only the END: a start inside a word is still a problem
    _edit(p, s02={"ss": 0.6, "d": 0.95}, s03={"trans": "cut", "hit": None})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "tail: hard cut ok" in r.stdout
    assert "start not quiet" in r.stdout


def test_missing_wav_is_loud_failure(speech_project):
    p = speech_project
    (p.wav_dir / "a.wav").unlink()
    r = _check(p)
    assert r.returncode == 2
    assert "wav/a.wav" in r.stdout + r.stderr


def test_tail_word_with_hard_cut_passes(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "cut", "hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: hard cut ok" in r.stdout
    assert "problems: 0" in r.stdout


def _words(p, ws):
    (p.build_dir / "words.json").write_text(
        json.dumps({"a": [{"w": w, "s": s, "e": e} for w, s, e in ws]}, ensure_ascii=False),
        encoding="utf-8")


def _clip(p, seconds, bursts):
    clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", seconds, tone_hz=440,
                              bursts=bursts)
    ffm.extract_wav(clip, p.wav_dir / "a.wav")


def _continuous(p):
    # one shout from 0.0 to 1.5 s, a 14 dB dip (no silence) at 1.18-1.20, then silence
    import numpy as np
    from scipy.io import wavfile
    sr = 48000
    t = np.arange(int(3.0 * sr)) / sr
    amp = np.where(t < 1.5, 0.5, 0.0)
    amp[(t >= 1.18) & (t < 1.20)] = 0.1
    wavfile.write(p.wav_dir / "a.wav", sr, (amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))


def test_start_after_previous_word_passes(speech_project):
    # "два" follows "один" with no pause: the start cannot be quiet, but it sits
    # in the envelope dip before the word, so only a note is printed
    p = speech_project
    _continuous(p)
    _words(p, [("один", 0.0, 1.2), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 1.18, "d": 0.8})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "start: not quiet (previous sound tail, dip 1.18)" in r.stdout
    assert "problems: 0" in r.stdout


def test_start_mid_word_in_continuous_speech_fails(speech_project):
    # whisper lags and puts "два" at 1.30; no pause before it, so onset_before
    # falls back to whisper; the dip at 1.18 shows the start is mid-word
    p = speech_project
    _continuous(p)
    _words(p, [("один", 0.0, 1.30), ("два", 1.30, 1.5)])
    _edit(p, s02={"ss": 1.30, "d": 0.7})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "start cuts into a word (dip 1.18)" in r.stdout


def test_start_cutting_into_word_fails(speech_project):
    # whisper puts "один" at 0.7, the sound starts at 0.5; a start at 0.6 cuts it
    p = speech_project
    _words(p, [("один", 0.7, 0.8), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 0.6, "d": 1.3})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "start cuts into a word" in r.stdout
