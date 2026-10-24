"""faster-whisper through the venv-whisper interpreter (the main venv cannot import it)."""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from . import ffm, home

REMINDER = ("напоминание: таймкоды whisper на криках опаздывают на 0,2–0,35 с; "
            "границы ставить по огибающей (envelope.py)")

_SCRIPT = r"""
import json, sys
import numpy as np
from faster_whisper import WhisperModel
model, lang, words = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
m = WhisperModel(model, device="cpu", compute_type="int8")
out = []
for raw in sys.argv[4:]:
    a = np.fromfile(raw, dtype=np.int16).astype(np.float32) / 32768
    segs, info = m.transcribe(a, language=lang, word_timestamps=words, beam_size=5,
                              vad_filter=False)
    out.append([dict(text=s.text.strip(), s=round(s.start, 2), e=round(s.end, 2),
                     words=[dict(w=w.word.strip(), s=round(w.start, 2), e=round(w.end, 2))
                            for w in (s.words or [])]) for s in segs])
print(json.dumps(out, ensure_ascii=False))
"""


class WhisperError(Exception):
    pass


def python() -> Path:
    return home() / "venv-whisper" / "bin" / "python"


def transcribe(paths: list[Path], lang="ru", model="medium", words=True) -> dict[str, list[dict]]:
    """{str(path): [{text, s, e, words: [{w, s, e}]}]} per input media file."""
    py = python()
    if not py.exists():
        raise WhisperError("venv-whisper not found, run: bash scripts/setup_env.sh")
    with tempfile.TemporaryDirectory() as tmp:
        raws = []
        for i, src in enumerate(paths):
            raw = Path(tmp) / f"{i}.raw"
            ffm.run(f"ffmpeg -v error -y -i {shlex.quote(str(src))} -vn -ac 1 -ar 16000 "
                    f"-f s16le {shlex.quote(str(raw))}")
            raws.append(str(raw))
        r = subprocess.run([str(py), "-c", _SCRIPT, model, lang, "1" if words else "0", *raws],
                           capture_output=True, text=True)
    if r.returncode != 0:
        raise WhisperError(f"whisper failed:\n{r.stderr[-2000:]}")
    return {str(p): segs for p, segs in zip(paths, json.loads(r.stdout.splitlines()[-1]))}
