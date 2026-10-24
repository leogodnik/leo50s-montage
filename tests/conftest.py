import shutil

import pytest


@pytest.fixture
def tmp_project(tmp_path):
    root = tmp_path / "_сборка"
    for sub in ("seg", "cards", "out", "wav"):
        (root / sub).mkdir(parents=True)
    return root


@pytest.fixture
def ffmpeg_ok():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не найден в PATH")
