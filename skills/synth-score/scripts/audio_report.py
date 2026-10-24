"""Spectrogram + waveform PNG of the score and its loudness (ebur128 I, LRA, Peak).

usage: audio_report.py <trailer.json | file.wav> [--out report.png]
Under the picture: a time axis (a tick every 5 s) and, for a trailer.json, every segment
id at its moment (visual start: start + ov/2 after fade/fadewhite, start after a cut).
"""
import argparse
import pathlib
import shlex
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from montagelib import ffm  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.timeline import timeline, visual_start  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
TICK_EVERY = 5
STRIP = 58  # px under the picture: time row, then two alternating id rows


def segment_marks(p) -> list[tuple[str, float]]:
    """(id, moment) of every segment: the visual start hits and layers are placed on."""
    tl, _ = timeline(p.edl, p.fps)
    return [(s.id, visual_start(s, st)) for s, st in tl]


def time_ticks(seconds: float) -> list[int]:
    return list(range(0, int(seconds) + 1, TICK_EVERY))


def annotate_report(png, seconds: float, marks=()) -> None:
    """Add the time axis and segment marks under the report picture, in place."""
    im = Image.open(png).convert("RGB")
    w, h = im.size
    out = Image.new("RGB", (w, h + STRIP), (0, 0, 0))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(FONT, 13)
    except OSError:
        font = ImageFont.load_default()

    def x(t):
        return min(max(round(t / seconds * w), 0), w - 1)

    for t in time_ticks(seconds):
        d.line([(x(t), h), (x(t), h + 6)], fill=(200, 200, 200))
        d.text((x(t) + 2, h + 4), f"{t} s" if t == 0 else str(t), fill=(200, 200, 200), font=font)
    for i, (sid, t) in enumerate(marks):
        y = h + 22 + (i % 2) * 18
        d.line([(x(t), h), (x(t), y + 14)], fill=(255, 200, 60))
        d.text((x(t) + 3, y), sid, fill=(255, 200, 60), font=font)
    out.save(png)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("src", help="trailer.json (uses out/score.wav) or a wav file")
    ap.add_argument("--out", help="PNG path (default: next to the wav, audio_report.png)")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src)
    try:
        marks = []
        if src.suffix.lower() == ".json":
            p = load_project(src)
            wav, out = p.out_dir / "score.wav", p.out_dir / "audio_report.png"
            marks = segment_marks(p)
        else:
            wav, out = src, src.with_name("audio_report.png")
        out = pathlib.Path(a.out) if a.out else out
        if not wav.exists():
            raise FileNotFoundError(f"{wav} not found: run 'build.py <trailer.json> audio' first")
        q = shlex.quote
        ffm.run(f"ffmpeg -v error -y -i {q(str(wav))} -filter_complex "
                f"\"[0:a]asplit[a][b];[a]showspectrumpic=s=1600x400:legend=0:scale=log[s];"
                f"[b]showwavespic=s=1600x200:colors=white[w];[s][w]vstack\" "
                f"-frames:v 1 {q(str(out))}")
        annotate_report(out, ffm.duration(wav), marks)
    except (ProjectError, ffm.FfmpegError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(wav), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"error: ebur128 failed\n{r.stderr[-2000:]}", file=sys.stderr)
        return 1
    summary = r.stderr.split("Summary:")[-1]
    for line in summary.splitlines():
        if line.strip().startswith(("I:", "LRA:", "Peak:")):
            print(line.strip())
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
