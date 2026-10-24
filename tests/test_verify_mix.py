import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from montagelib import ffm
from montagelib.assemble import silent_score
from montagelib.project import load_project
from montagelib.timeline import timeline

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "skills" / "speech-safe-cuts" / "scripts" / "verify_mix.py"


def _mix(mini_project, ss):
    """Speech clip a (bursts 0.5-0.8, 1.2-1.5) from `ss` laid into out/score.wav at its start."""
    p = mini_project
    clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", 2.0, tone_hz=440,
                              bursts=[(0.5, 0.8), (1.2, 1.5)])
    ffm.extract_wav(clip, p.wav_dir / "a.wav")
    data = json.loads(p.path.read_text(encoding="utf-8"))
    data["edl"][2].update(ss=ss, d=1.9)
    p.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    p = load_project(p.path)
    tl, _ = timeline(p.edl, p.fps)
    seg, start = next((s, st) for s, st in tl if s.id == "s02")
    sr, mix = wavfile.read(silent_score(p))
    _, src = wavfile.read(p.wav_dir / "a.wav")
    i, n = int(round(start * sr)), int(round(seg.d * sr))
    mix[i:i + n] = src[int(round(seg.ss * sr)):][:n]
    wavfile.write(p.out_dir / "score.wav", sr, mix)
    return p, start, sr


@pytest.fixture
def mixed(mini_project):
    return _mix(mini_project, 0)


def _verify(p):
    return subprocess.run([sys.executable, str(VERIFY), str(p.path)],
                          capture_output=True, text=True)


def test_all_syllables_heard_passes(mixed):
    p, _, _ = mixed
    r = _verify(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "peaks src/matched 2/2" in r.stdout
    assert "problems: 0" in r.stdout


def test_masked_syllable_fails(mixed):
    p, start, sr = mixed
    score = p.out_dir / "score.wav"
    _, mix = wavfile.read(score)
    mix[int((start + 1.15) * sr):int((start + 1.55) * sr)] = 0
    wavfile.write(score, sr, mix)
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "syllable lost" in r.stdout


def test_off_grid_segment_matches(mini_project):
    # ss is not on the 20 ms grid: source and mix windows must still share the phase
    p, _, _ = _mix(mini_project, 0.013)
    r = _verify(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "peaks src/matched 2/2" in r.stdout
