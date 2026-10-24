import shutil
from pathlib import Path

import pytest

from montagelib.project import DEFAULT_FONTS, load_project
from montagelib.timeline import timeline

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "skills/trailer-build/references/example-trailer.json")
ZEMLYANKA = Path.home() / "Землянка"


def _fonts_present():
    return all(Path(spec.split("#", 1)[0]).exists() for spec in DEFAULT_FONTS.values())


@pytest.fixture
def example(tmp_path):
    if not _fonts_present():
        pytest.skip("default fonts (macOS) not found")
    root = tmp_path / "_сборка"
    root.mkdir()
    shutil.copy(EXAMPLE, root / "trailer.json")
    real = ZEMLYANKA / "_сборка" / "gammas.txt"
    if real.exists():
        shutil.copy(real, root / "gammas.txt")
    else:  # the grade values do not affect the timeline
        (root / "gammas.txt").write_text(
            "".join(f"c{i} 1.0 1.0\n" for i in range(1, 8)), encoding="utf-8")
    return load_project(root / "trailer.json")


def test_example_timeline(example):
    assert len(example.edl) == 25
    _, total = timeline(example.edl, example.fps)
    assert total == pytest.approx(66.42, abs=1 / 48)
    assert example.by_id("s20").hit == {
        "type": "explosion", "gain": None, "offset": 0, "whistle_from": "s19"}
    assert example.output["name"] == "Землянка_трейлер"


def test_example_clips_resolve_in_project_folder():
    if not ZEMLYANKA.is_dir():
        pytest.skip("«Землянка» clips folder not found")
    import json
    clips = json.loads(EXAMPLE.read_text(encoding="utf-8"))["clips"]
    for alias, rel in clips.items():
        assert (ZEMLYANKA / "_сборка" / rel).resolve().is_file(), alias
