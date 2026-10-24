"""Storyboard of the finished trailer for checking by eye: a frame every N seconds,
each labelled with its time and the segment visually dominant at that moment.

usage: contact_sheet.py <trailer.json> <out.jpg> [--every 2.0]
"""
import io
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.timeline import timeline, visual_start  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"  # has Cyrillic
TW, COLS, LABEL = 384, 5, 22


def font(size=14):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def dominant(tl, t: float) -> str:
    sid = tl[0][0].id
    for s, start in tl:
        if visual_start(s, start) <= t:
            sid = s.id
    return sid


def main(argv):
    every = 2.0
    if "--every" in argv:
        i = argv.index("--every")
        every = float(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        p = load_project(argv[0])
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    src = p.out_dir / f"{p.output['name']}.mp4"
    if not src.exists():
        print(f"error: {src} not found: run the 'mux' step first", file=sys.stderr)
        return 1
    tl, total = timeline(p.edl, p.fps)
    times = [round(k * every, 2) for k in range(int(total / every) + 1) if k * every < total]
    th = round(TW * p.height / p.width)
    rows = (len(times) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TW, rows * (th + LABEL)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    f = font()
    for i, t in enumerate(times):
        b = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(src),
                            "-frames:v", "1", "-vf", f"scale={TW}:{th}",
                            "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True).stdout
        x, y = (i % COLS) * TW, (i // COLS) * (th + LABEL)
        if b:
            sheet.paste(Image.open(io.BytesIO(b)), (x, y + LABEL))
        d.text((x + 4, y + 3), f"{t:.2f}s  {dominant(tl, t)}", fill=(255, 220, 90), font=f)
    sheet.save(argv[1], quality=85)
    print("ok", argv[1], len(times), "frames")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
