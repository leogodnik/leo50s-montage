#!/usr/bin/env bash
# Создаёт окружения Python для leo50s-montage. Идемпотентно. Флаг --no-whisper пропускает venv-whisper.
set -euo pipefail

HOME_DIR="${LEO50S_HOME:-$HOME/.cache/leo50s-montage}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
WHISPER=1
for a in "$@"; do
  [ "$a" = "--no-whisper" ] && WHISPER=0
done

mkdir -p "$HOME_DIR"

if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  python3 -m venv "$HOME_DIR/venv"
fi
"$HOME_DIR/venv/bin/python" -m pip install -q numpy scipy pillow pytest
echo "venv: $HOME_DIR/venv"

if [ "$WHISPER" = 1 ]; then
  if [ ! -x "$HOME_DIR/venv-whisper/bin/python" ]; then
    "$UV" venv --python 3.12 "$HOME_DIR/venv-whisper"
  fi
  "$UV" pip install -q --python "$HOME_DIR/venv-whisper/bin/python" faster-whisper
  echo "venv-whisper: $HOME_DIR/venv-whisper"
  # the first transcription would otherwise download the model (~1.5 GB) inside a shell timeout
  if "$HOME_DIR/venv-whisper/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"; then
    echo "whisper model: medium ready"
  else
    echo "warning: whisper model 'medium' not downloaded; the first transcription will fetch it" >&2
  fi
fi
