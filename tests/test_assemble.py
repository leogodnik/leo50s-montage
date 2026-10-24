import shlex
import subprocess
import sys
from pathlib import Path

from montagelib import ffm
from montagelib.assemble import _graph, assemble, film_overlay
from montagelib.segments import render_segments
from montagelib.timeline import frames, timeline

BUILD = Path(__file__).resolve().parents[1] / "skills" / "trailer-build" / "scripts" / "build.py"


def _build_video(p):
    render_segments(p)
    film_overlay(p)
    return assemble(p)


def test_assembled_length_equals_timeline(mini_project):
    p = mini_project
    out = _build_video(p)
    assert out == p.out_dir / "video_only.mp4"
    _, total = timeline(p.edl, p.fps)
    assert ffm.count_frames(out) == frames(total, p.fps)


def test_stale_film_is_regenerated(mini_project):
    p = mini_project
    render_segments(p)
    film_overlay(p)
    _, old_total = timeline(p.edl, p.fps)
    p.edl = p.edl[:-1]
    p.by_id("s02").d = 0.75
    _, new_total = timeline(p.edl, p.fps)
    assert abs((old_total - new_total) - 1.0) < 1e-6
    # old film is still long enough; make it stale by shortening below the new total
    ffm.run(f"ffmpeg -v error -y -i {shlex.quote(str(p.out_dir / 'film.mp4'))} -frames:v 10 "
            f"-c:v libx264 -pix_fmt gray {shlex.quote(str(p.out_dir / 'film_short.mp4'))}")
    (p.out_dir / "film_short.mp4").replace(p.out_dir / "film.mp4")
    render_segments(p, only=["s02"])
    out = assemble(p)
    assert ffm.count_frames(out) == frames(new_total, p.fps)
    assert ffm.count_frames(p.out_dir / "film.mp4") == frames(new_total, p.fps) + p.fps


def test_chroma_neutral(mini_project):
    p = mini_project
    out = _build_video(p)
    _, total = timeline(p.edl, p.fps)
    mid = total / 2
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "lavfi", "-i",
         f"movie={out},trim=start={mid:.3f},setpts=PTS-STARTPTS,signalstats",
         "-read_intervals", "%+#1",
         "-show_entries", "frame_tags=lavfi.signalstats.UAVG,lavfi.signalstats.VAVG",
         "-of", "default=noprint_wrappers=1"],
        capture_output=True, text=True, check=True)
    tags = dict(line.split("=", 1) for line in r.stdout.split() if "=" in line)
    u = float(tags["TAG:lavfi.signalstats.UAVG"])
    v = float(tags["TAG:lavfi.signalstats.VAVG"])
    assert abs(u - 128) < 1.5 and abs(v - 128) < 1.5


def test_build_cli_edl_prints_total(mini_project):
    p = mini_project
    r = subprocess.run([sys.executable, str(BUILD), str(p.path), "edl"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _, total = timeline(p.edl, p.fps)
    assert f"total {total:.2f}" in r.stdout.splitlines()


def test_letterbox_drawboxes_only_when_positive(mini_project):
    p = mini_project
    tl, _ = timeline(p.edl, p.fps)
    p.letterbox = 138
    g = _graph(p, tl)
    assert g.count("drawbox=") == 2
    assert f"drawbox=x=0:y=0:w={p.width}:h=138" in g
    assert f"drawbox=x=0:y={p.height - 138}:w={p.width}:h=138" in g
    for none in (0, None):
        p.letterbox = none
        assert "drawbox" not in _graph(p, tl)
