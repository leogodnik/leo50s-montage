import shutil
from pathlib import Path

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


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_project(tmp_path, ffmpeg_ok):
    from montagelib import ffm
    from montagelib.project import load_project

    root = tmp_path / "_сборка"
    for sub in ("seg", "cards", "out", "wav", "clips"):
        (root / sub).mkdir(parents=True)
    shutil.copy(FIXTURES / "mini-trailer.json", root / "trailer.json")
    ffm.make_test_clip(root / "clips" / "a.mp4", 2.0)
    ffm.make_test_clip(root / "clips" / "b.mp4", 2.0, brightness=0.2)
    return load_project(root / "trailer.json")
