import numpy as np
import pytest
from PIL import Image

from montagelib import ffm
from montagelib.cards import render_card
from montagelib.paint_title import paint_title
from montagelib.project import Segment
from montagelib.segments import SegmentError, render_segments
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


def test_render_segments_frame_counts(mini_project):
    p = mini_project
    res = render_segments(p)
    assert [r[0] for r in res] == [s.id for s in p.edl]
    for sid, got, expected in res:
        assert got == expected == frames(p.by_id(sid).d, p.fps), sid
        assert (p.seg_dir / f"{sid}.mp4").exists()


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
