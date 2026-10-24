"""Grade check by eye: one frame per clip, source on top, graded (gamma/contrast + look) below.

usage: preview_grid.py <trailer.json> [--t 4.0] [--out grid.jpg]
       preview_grid.py --clips a=path b=path ... [--look noir] [--grade gammas.txt]
                       [--t 4.0] [--out grid.jpg]
default out: <build_dir>/out/grade_grid.jpg; in clip mode ./grade_grid.jpg
(no --grade in clip mode: gamma/contrast 1.0/1.0, i.e. the look alone)
"""
import argparse
import io
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from montagelib import ffm  # noqa: E402
from montagelib.gradecli import add_source_args, sources  # noqa: E402
from montagelib.looks import load_look  # noqa: E402
from montagelib.project import ProjectError  # noqa: E402

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"  # has Cyrillic
TW, LABEL = 320, 22


def font(size=14):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def aliases(p) -> list[str]:
    """Clip aliases in EDL order of first use, then the unused ones in `clips` order."""
    used = []
    for s in p.edl:
        for a in (s.src, s.bg_src):
            if a is not None and a not in used:
                used.append(a)
    return used + [a for a in p.clips if a not in used]


def frame(src, t: float, vf: str) -> Image.Image:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(src), "-frames:v", "1",
           "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise ffm.FfmpegError(" ".join(cmd), r.stderr.decode(errors="replace")[-2000:])
    return Image.open(io.BytesIO(r.stdout)).convert("RGB")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(ap)
    ap.add_argument("--t", type=float, default=4.0)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    try:
        src = sources(ap, a)
        look = load_look(src.look)
    except (ProjectError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    p = src.project
    names = aliases(p) if p else list(src.clips)
    missing = [f"{k}: {src.clips[k]}" for k in names if not src.clips[k].is_file()]
    if missing:
        for m in missing:
            print(f"error: clip not found: {m}", file=sys.stderr)
        return 1
    try:
        w, h = (p.width, p.height) if p else ffm.frame_size(src.clips[names[0]])
    except ffm.FfmpegError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    th = round(TW * h / w)
    scale = f"scale={TW}:{th},setsar=1"
    grid = Image.new("RGB", (len(names) * TW, 2 * (th + LABEL)), (20, 20, 20))
    d = ImageDraw.Draw(grid)
    f = font()
    try:
        for i, k in enumerate(names):
            clip = src.clips[k]
            t = max(0.0, min(a.t, ffm.duration(clip) - 0.1))
            g, c = src.grade.get(k, (1.0, 1.0))
            x = i * TW
            for row, (vf, label) in enumerate(((scale, f"{k}  source"),
                                               (f"eq=gamma={g}:contrast={c},{look},{scale}",
                                                f"{k}  γ {g} c {c}"))):
                y = row * (th + LABEL)
                grid.paste(frame(clip, t, vf), (x, y + LABEL))
                d.text((x + 4, y + 3), label, fill=(255, 220, 90), font=f)
    except ffm.FfmpegError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = a.out or (p.out_dir if p else pathlib.Path()) / "grade_grid.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=88)
    print("ok", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
