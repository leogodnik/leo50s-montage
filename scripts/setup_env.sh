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
fi
