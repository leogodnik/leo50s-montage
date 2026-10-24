import numpy as np
import pytest

from montagelib import ffm
from montagelib.grade import measure, read_grade, solve, write_grade
from montagelib.looks import load_look


@pytest.fixture
def three_clips(tmp_path, ffmpeg_ok):
    clips = {}
    for alias, b in (("a", -0.3), ("b", 0.0), ("c", 0.3)):
        clips[alias] = ffm.make_test_clip(tmp_path / f"{alias}.mp4", 2.0, size="384x216",
                                          brightness=b)
    return clips


def test_solve_converges_mean(three_clips):
    result = solve(three_clips, "noir")
    assert set(result) == {"a", "b", "c"}
    for alias, (g, c) in result.items():
        mean, _ = measure(three_clips[alias], "noir", g, c)
        assert abs(mean - 82) <= 2, (alias, g, c, mean)


def test_contrast_groups_fix_contrast(three_clips):
    result = solve(three_clips, "noir", contrast_groups={"a": 1.2})
    assert result["a"][1] == 1.2
    mean, _ = measure(three_clips["a"], "noir", *result["a"])
    assert abs(mean - 82) <= 2


def test_grade_roundtrip(tmp_path):
    result = {"c2": (0.95, 1.0), "c1": (1.233, 0.86)}
    path = tmp_path / "gammas.txt"
    write_grade(path, result)
    assert path.read_text() == "c1 1.233 0.86\nc2 0.95 1.0\n"
    assert read_grade(path) == result
    (tmp_path / "old.txt").write_text("c1 1.233\n\n# note\n")
    assert read_grade(tmp_path / "old.txt") == {"c1": (1.233, 1.0)}


def _y4m_planes(path):
    """Y, U, V planes of a single-frame yuv420p .y4m."""
    raw = path.read_bytes()
    header, rest = raw.split(b"\n", 1)
    w, h = (int(t[1:]) for t in header.split() if t[:1] in (b"W", b"H"))
    data = np.frombuffer(rest.split(b"\n", 1)[1], np.uint8)
    y, u = w * h, (w // 2) * (h // 2)
    return data[:y], data[y:y + u], data[y + u:y + 2 * u]


def test_noir_is_monochrome(tmp_path, ffmpeg_ok):
    clip = ffm.make_test_clip(tmp_path / "c.mp4", 1.0, size="384x216")  # testsrc2: colourful
    _, u0, _ = _y4m_planes(ffm.grab_frame(clip, 0.5, tmp_path / "raw.y4m",
                                          vf="format=yuv420p"))
    assert np.ptp(u0) > 100
    _, u, v = _y4m_planes(ffm.grab_frame(
        clip, 0.5, tmp_path / "noir.y4m",
        vf=f"eq=gamma=1:contrast=1,{load_look('noir')},format=yuv420p"))
    # no colour left anywhere; ffmpeg eq=saturation=0 parks neutral chroma at 127, not 128
    assert np.ptp(u) == 0 and np.ptp(v) == 0
    assert abs(u.mean() - 128) <= 1 and abs(v.mean() - 128) <= 1
