"""Full graded copies of every clip (gamma/contrast + look), audio kept.

usage: render_graded.py <trailer.json> [--out DIR]
default out: <clips' common folder>/graded_<look>; files <alias>_<look>_<stem>.mp4
"""
import argparse
import os
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm  # noqa: E402
from montagelib.looks import load_look  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    try:
        p = load_project(a.project)
        look = load_look(p.look)
    except (ProjectError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    missing = [f"{k}: {v}" for k, v in p.clips.items() if not v.is_file()]
    if missing:
        for m in missing:
            print(f"error: clip not found: {m}", file=sys.stderr)
        return 1
    tag = pathlib.Path(p.look).stem
    if a.out:
        out_dir = a.out
    else:
        parent = pathlib.Path(os.path.commonpath([v.parent for v in p.clips.values()]))
        out_dir = parent / f"graded_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, src in p.clips.items():
        g, c = p.grade.get(k, (1.0, 1.0))
        dst = out_dir / f"{k}_{tag}_{src.stem}.mp4"
        try:
            ffm.run(f"ffmpeg -v error -y -i {shlex.quote(str(src))} "
                    f"-vf \"eq=gamma={g}:contrast={c},{look},format=yuv420p\" "
                    f"-c:v libx264 -crf 16 -preset medium -c:a copy -movflags +faststart "
                    f"{shlex.quote(str(dst))}")
        except ffm.FfmpegError as e:
            print(f"error: {k}: {e}", file=sys.stderr)
            return 1
        print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
