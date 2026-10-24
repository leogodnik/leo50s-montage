import numpy as np
import pytest
from PIL import Image

from montagelib import ffm
from montagelib.cards import render_card
from montagelib.looks import load_look
from montagelib.paint_title import paint_title
from montagelib.project import Segment
from montagelib.segments import SegmentError, render_segments, segment_cmd
from montagelib.timeline import frames


def test_make_test_clip_frames(tmp_path, ffmpeg_ok):
    clip = ffm.make_test_clip(tmp_path / "t.mp4", 2.0, tone_hz=440, bursts=[(0.5, 1.0)])
    assert ffm.count_frames(clip) == 48


def test_card_png_size_and_autoshrink(mini_project):
    p = mini_project
    seg = Segment(id="long", kind="card", d=1.0, text="Ш" * 40)
    png = render_card(seg, p)
    assert png == p.cards_dir / "long.png"
    img = Image.open(png)
    assert img.size == (p.width, p.height)
    left, top, right, bottom = img.convert("L").getbbox()
    assert left >= 120 and right <= p.width - 120


def test_painted_title_has_drips(tmp_path, mini_project):
    font = mini_project.fonts["title"]
    dry = np.asarray(Image.open(paint_title("ТИТР", tmp_path / "dry.png", font, drip_prob=0.0)))
    wet = np.asarray(Image.open(paint_title("ТИТР", tmp_path / "wet.png", font)))
    assert wet.shape == (1080, 1920, 4)
    glyph_bottom = np.where((dry[:, :, 3] > 0).any(axis=1))[0].max()
    assert (wet[glyph_bottom + 5:, :, 3] > 128).any()


def test_painted_title_fits_small_frame(tmp_path, mini_project):
    font = mini_project.fonts["title"]
    img = np.asarray(Image.open(paint_title("ZЕМЛЯНКА", tmp_path / "z.png", font, size=(1280, 720))))
    assert img.shape == (720, 1280, 4)
    alpha = img[:, :, 3]
    assert not (alpha[:, :60] > 0).any() and not (alpha[:, -60:] > 0).any()


def test_painted_title_clip_background_is_graded(mini_project):
    p = mini_project
    p.grade = {"a": (1.2, 1.1)}
    seg = Segment(id="tc", kind="title", text="ТИТР", d=1.5, style="painted",
                  bg="clip", bg_src="a")
    cmd = segment_cmd(seg, p)
    assert f"[0:v]eq=gamma=1.2:contrast=1.1,{load_look(p.look)},scale=" in cmd
    assert "title_paint_tc.png" in cmd


def test_render_segments_frame_counts(mini_project):
    p = mini_project
    res = render_segments(p)
    assert [r[0] for r in res] == [s.id for s in p.edl]
    for sid, got, expected in res:
        assert got == expected == frames(p.by_id(sid).d, p.fps), sid
        assert (p.seg_dir / f"{sid}.mp4").exists()
    assert (p.cards_dir / "title_paint_s04.png").exists()


def test_short_clip_reports_mismatch(mini_project):
    p = mini_project
    seg = p.by_id("s02")
    seg.ss, seg.d = 1.0, 3.0
    with pytest.raises(SegmentError) as ex:
        render_segments(p, only=["s02"])
    (sid, got, expected), = ex.value.mismatches
    assert (sid, expected) == ("s02", 72) and got < expected
    msg = str(ex.value)
    assert "s02" in msg and str(got) in msg and "72" in msg
