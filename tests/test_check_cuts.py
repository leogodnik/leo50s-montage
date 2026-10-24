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
