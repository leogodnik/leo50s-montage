"""Calibration on the real «Землянка» v6 build: both checkers stay at zero, and the
speech-safe-cuts regression (s01 from whisper's 1.68) is still caught."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from montagelib import ffm

ROOT = Path(__file__).resolve().parents[1]
SSC = ROOT / "skills" / "speech-safe-cuts" / "scripts"
V6 = Path("/Users/leogodnik/Землянка/_сборка")
C6 = Path("/Users/leogodnik/Землянка/gemini_generated_video_d6940d17.mp4")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not all((V6 / n).exists() for n in
                               ("trailer.json", "wav", "words.json", "out/score.wav")),
                       reason="Землянка v6 build not found"),
]


def _run(script, *args):
    return subprocess.run([sys.executable, str(SSC / script), *map(str, args)],
                          capture_output=True, text=True)


def test_v6_check_cuts_is_clean():
    r = _run("check_cuts.py", V6 / "trailer.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "problems: 0" in r.stdout


def test_v6_verify_mix_is_clean():
    r = _run("verify_mix.py", V6 / "trailer.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "problems: 0" in r.stdout


def test_ssc_regression_whisper_start_is_caught(tmp_path):
    """tests/skills/speech-safe-cuts/README.md: s01 at whisper's 1.68 loses «тя»."""
    if not C6.exists():
        pytest.skip(f"{C6} not found")
    root = tmp_path / "_сборка"
    (root / "wav").mkdir(parents=True)
    edl = [
        {"id": "s00", "kind": "black", "d": 1.0},
        {"id": "s01", "kind": "clip", "src": "shout", "ss": 1.68, "d": 0.67, "ov": 0.3,
         "trans": "fade", "speech": True},
        {"id": "s02", "kind": "clip", "src": "shout", "ss": 3.24, "d": 1.3, "ov": 0.2,
         "trans": "fadewhite", "hit": "hit"},
        {"id": "s03", "kind": "card", "text": "СКОРО", "d": 2.0},
        {"id": "s04", "kind": "black", "d": 1.0},
    ]
    (root / "trailer.json").write_text(json.dumps(
        {"clips": {"shout": str(C6)}, "fps": 24, "size": [1920, 1080], "look": "noir",
         "edl": edl}, ensure_ascii=False), encoding="utf-8")
    words = json.loads((V6 / "words.json").read_text(encoding="utf-8"))
    (root / "words.json").write_text(json.dumps({"shout": words["c6"]}, ensure_ascii=False),
                                     encoding="utf-8")
    ffm.extract_wav(C6, root / "wav" / "shout.wav")
    r = _run("check_cuts.py", root / "trailer.json")
    assert r.returncode == 1, r.stdout + r.stderr
    problems = int(r.stdout.strip().splitlines()[-1].split(":")[1])
    assert problems >= 3, r.stdout
