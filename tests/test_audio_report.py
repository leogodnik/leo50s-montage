"""audio_report.py: time axis and segment ids under the spectrogram and waveform."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "synth-score" / "scripts" / "audio_report.py"
RAW_HEIGHT = 600  # showspectrumpic 400 + showwavespic 200


def _module():
    spec = importlib.util.spec_from_file_location("audio_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wav(path, seconds):
    sr = 48000
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    wavfile.write(path, sr, (np.stack([x, x], 1) * 32767).astype(np.int16))


def test_marks_are_segment_visual_starts(mini_project):
    marks = _module().segment_marks(mini_project)
    # s01/s02/s05 enter with a 0.25 s fade (moment = start + ov/2), s03 with a cut
    # (one-frame overlap), s04 with a zero-length fade
    expected = [("s00", 0.0), ("s01", 0.375), ("s02", 1.125), ("s03", 2.5 - 1 / 24),
                ("s04", 3.5 - 1 / 24), ("s05", 5.0 - 1 / 24 - 0.125)]
    assert [k for k, _ in marks] == [k for k, _ in expected]
    assert np.allclose([t for _, t in marks], [t for _, t in expected])


def test_time_ticks_every_5_s():
    assert _module().time_ticks(16.5) == [0, 5, 10, 15]


def test_report_for_a_project_is_annotated(mini_project, ffmpeg_ok):
    p = mini_project
    _wav(p.out_dir / "score.wav", 6.0)
    r = subprocess.run([sys.executable, str(SCRIPT), str(p.path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "I:" in r.stdout
    with Image.open(p.out_dir / "audio_report.png") as im:
        assert im.size[0] == 1600 and im.size[1] > RAW_HEIGHT


def test_report_for_a_wav_has_the_time_axis(tmp_path, ffmpeg_ok):
    _wav(tmp_path / "x.wav", 3.0)
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "x.wav")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with Image.open(tmp_path / "audio_report.png") as im:
        assert im.size[1] > RAW_HEIGHT
