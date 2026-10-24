"""Full graded copies of every clip (gamma/contrast + look), audio kept.

usage: render_graded.py <trailer.json> [--out DIR]
       render_graded.py --clips a=path b=path ... [--look noir] [--grade gammas.txt] [--out DIR]
default out: <build_dir>/graded_<look>; in clip mode ./graded_<look>
files: <stem>_<look>.mp4 when the alias is the file stem, else <alias>_<look>_<stem>.mp4
(no --grade in clip mode: gamma/contrast 1.0/1.0, i.e. the look alone)
"""
import argparse
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm  # noqa: E402
from montagelib.gradecli import add_source_args, sources  # noqa: E402
from montagelib.looks import load_look  # noqa: E402
from montagelib.project import ProjectError  # noqa: E402


def copy_name(alias: str, src: pathlib.Path, tag: str) -> str:
    return f"{src.stem}_{tag}.mp4" if alias == src.stem else f"{alias}_{tag}_{src.stem}.mp4"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(ap)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    try:
        sc = sources(ap, a)
        look = load_look(sc.look)
    except (ProjectError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    missing = [f"{k}: {v}" for k, v in sc.clips.items() if not v.is_file()]
    if missing:
        for m in missing:
            print(f"error: clip not found: {m}", file=sys.stderr)
        return 1
    tag = pathlib.Path(sc.look).stem
    base = sc.project.build_dir if sc.project else pathlib.Path()
    out_dir = a.out or base / f"graded_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, src in sc.clips.items():
        g, c = sc.grade.get(k, (1.0, 1.0))
        dst = out_dir / copy_name(k, src, tag)
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
