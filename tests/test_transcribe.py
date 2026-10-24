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


def test_missing_whisper_venv_hint_names_the_plugin_script(tmp_path, monkeypatch):
    from montagelib import whisper
    monkeypatch.setenv("LEO50S_HOME", str(tmp_path))
    with pytest.raises(whisper.WhisperError) as ex:
        whisper.transcribe([tmp_path / "x.mp4"])
    assert "${CLAUDE_PLUGIN_ROOT:-" in str(ex.value)
    assert str(ROOT) in str(ex.value) and "scripts/setup_env.sh" in str(ex.value)


def test_empty_whisper_output_is_a_whisper_error(tmp_path, monkeypatch, ffmpeg_ok):
    from montagelib import ffm, whisper
    py = tmp_path / "venv-whisper" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    monkeypatch.setenv("LEO50S_HOME", str(tmp_path))
    clip = ffm.make_test_clip(tmp_path / "c.mp4", 0.3)
    monkeypatch.setattr(whisper.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="", stderr=""))
    with pytest.raises(whisper.WhisperError, match="no output"):
        whisper.transcribe([clip])
