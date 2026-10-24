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


def _gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def test_shake_on_a_1080p_source_is_cut_from_the_720p_frame(mini_project, tmp_path):
    p = mini_project
    ffm.make_test_clip(p.build_dir / "clips" / "big.mp4", 1.0, size="1920x1080")
    p.clips["big"] = p.build_dir / "clips" / "big.mp4"
    plain = tmp_path / "plain.txt"
    plain.write_text("null\n", encoding="utf-8")  # no look: geometry only
    p.look = str(plain)
    seg = Segment(id="sh", kind="clip", src="big", ss=0, d=0.5, fx=["shake"], shake=0)
    cmd = segment_cmd(seg, p)
    assert 0 <= cmd.index("scale=1280:720:flags=lanczos") < cmd.index("crop=1216:684")
    ffm.run(cmd)
    out = p.seg_dir / "sh.mp4"
    assert ffm.count_frames(out) == frames(0.5, p.fps)
    ffm.grab_frame(out, 0, tmp_path / "got.png")
    # shake 0: a fixed centred crop of the 720p frame, back to the project size
    ffm.grab_frame(p.clips["big"], 0, tmp_path / "ref.png",
                   vf=f"scale=1280:720:flags=lanczos,crop=1216:684:32:18,"
                      f"scale={p.width}:{p.height}:flags=lanczos")
    ffm.grab_frame(p.clips["big"], 0, tmp_path / "wrong.png",
                   vf=f"crop=1216:684:32:18,scale={p.width}:{p.height}:flags=lanczos")
    got, ref, wrong = (_gray(tmp_path / f"{n}.png") for n in ("got", "ref", "wrong"))
    assert np.abs(got - ref).mean() < 4
    assert np.abs(got - wrong).mean() > 4 * np.abs(got - ref).mean()


def test_push_and_plain_clips_of_720p_sources_keep_their_chain(mini_project):
    p = mini_project
    seg = Segment(id="pu", kind="clip", src="a", ss=0, d=0.5, fx=["push"])
    cmd = segment_cmd(seg, p)
    assert cmd.index("scale=1280:720:flags=lanczos") < cmd.index("eq=gamma") < cmd.index("zoompan")
    plain = Segment(id="pl", kind="clip", src="a", ss=0, d=0.5)
    assert "scale=1280:720" not in segment_cmd(plain, p)


def test_probe_clips_warns_about_non_720p_sources(tmp_path, ffmpeg_ok):
    import subprocess
    import sys
    from pathlib import Path
    probe = Path(__file__).resolve().parents[1] / "skills/trailer-build/scripts/probe_clips.py"
    d = tmp_path / "clips"
    d.mkdir()
    ffm.make_test_clip(d / "hd.mp4", 0.5)
    ffm.make_test_clip(d / "full.mp4", 0.5, size="1920x1080")
    r = subprocess.run([sys.executable, str(probe), str(d)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    warnings = [line for line in r.stdout.splitlines() if line.startswith("warning:")]
    assert len(warnings) == 1
    assert "full.mp4" in warnings[0] and "not 1280x720" in warnings[0]
