"""Clip mode (--clips, no trailer.json) of preview_grid.py and render_graded.py."""
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from montagelib import ffm

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "match-grade" / "scripts"


@pytest.fixture
def two_clips(tmp_path, ffmpeg_ok):
    src = tmp_path / "src"
    src.mkdir()
    return {alias: ffm.make_test_clip(src / f"{alias}.mp4", 1.0, size="384x216", brightness=b)
            for alias, b in (("a", -0.2), ("b", 0.2))}


def _run(script, args, cwd):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=cwd,
                          capture_output=True, text=True)


def test_preview_grid_clip_mode(tmp_path, two_clips):
    work = tmp_path / "work"
    work.mkdir()
    r = _run("preview_grid.py", ["--clips", f"a={two_clips['a']}", f"b={two_clips['b']}",
                                 "--t", "0.5"], work)
    assert r.returncode == 0, r.stdout + r.stderr
    grid = work / "grade_grid.jpg"  # --out defaults to the current directory
    assert grid.is_file() and f"ok {grid.name}" in r.stdout
    with Image.open(grid) as im:
        assert im.size[0] == 2 * 320
    (work / "gammas.txt").write_text("a 1.4 1.0\nb 0.9 1.1\n")
    r = _run("preview_grid.py", ["--clips", f"a={two_clips['a']}", f"b={two_clips['b']}",
                                 "--look", "clean-bw", "--grade", "gammas.txt",
                                 "--out", "g2.jpg"], work)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (work / "g2.jpg").is_file()


def test_render_graded_clip_mode(tmp_path, two_clips):
    work = tmp_path / "work"
    work.mkdir()
    (work / "gammas.txt").write_text("a 1.4 1.0\n")
    r = _run("render_graded.py", ["--clips", f"a={two_clips['a']}", f"x={two_clips['b']}",
                                  "--grade", "gammas.txt"], work)
    assert r.returncode == 0, r.stdout + r.stderr
    out = work / "graded_noir"
    # alias equal to the file stem: <stem>_<look>.mp4; otherwise <alias>_<look>_<stem>.mp4
    assert sorted(f.name for f in out.iterdir()) == ["a_noir.mp4", "x_noir_b.mp4"]
    for f in out.iterdir():
        assert abs(ffm.duration(f) - 1.0) < 0.1


def test_clip_mode_needs_exactly_one_source(tmp_path, two_clips):
    for script in ("preview_grid.py", "render_graded.py"):
        r = _run(script, [], tmp_path)
        assert r.returncode == 2 and "--clips" in r.stderr
