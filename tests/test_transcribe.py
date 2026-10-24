import json
import subprocess
import sys
from pathlib import Path

import pytest

from montagelib import home

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE = ROOT / "skills" / "speech-safe-cuts" / "scripts" / "transcribe_words.py"
C6 = Path("/Users/leogodnik/Землянка/gemini_generated_video_d6940d17.mp4")


@pytest.mark.slow
def test_transcribe_zemlyanka_c6(tmp_path):
    if not C6.exists():
        pytest.skip(f"{C6} not found")
    if not (home() / "venv-whisper" / "bin" / "python").exists():
        pytest.skip("venv-whisper not installed")
    out = tmp_path / "words.json"
    r = subprocess.run([sys.executable, str(TRANSCRIBE), str(C6), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    words = json.loads(out.read_text(encoding="utf-8"))
    assert any(w["w"].startswith("Тяни") for w in words[C6.stem])
    assert "огибающей" in r.stdout
