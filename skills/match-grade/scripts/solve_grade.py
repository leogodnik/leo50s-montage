"""Solve per-clip gamma/contrast so every clip lands on one luma target after the look.

usage: solve_grade.py <trailer.json> [--target-mean 82] [--contrast alias=1.22 ...] [--out FILE]
       solve_grade.py --clips a=path b=path ... [--look noir] [--target-mean 82]
                      [--contrast alias=1.22 ...] [--out gammas.txt]
default out: gammas.txt next to trailer.json; in clip mode in the current directory
"before" is measured after the look at gamma/contrast 1.0; the mean is full-range 0..255
exit: 0 all clips on target, 1 error or a clip more than 2 off the target mean (file still written)
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib.ffm import FfmpegError  # noqa: E402
from montagelib.grade import solve, write_grade  # noqa: E402
from montagelib.gradecli import add_source_args, contrasts, sources  # noqa: E402
from montagelib.project import ProjectError  # noqa: E402

TOLERANCE = 2.0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(ap, grade=False)
    ap.add_argument("--target-mean", type=float, default=82.0)
    ap.add_argument("--contrast", action="append", metavar="ALIAS=C")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    try:
        groups = contrasts(a.contrast)
        src = sources(ap, a)
        clips, look = src.clips, src.look
        out = a.out or (src.project.build_dir if src.project else pathlib.Path()) / "gammas.txt"
    except (ProjectError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    unknown = sorted(set(groups) - set(clips))
    missing = [f"{k}: {v}" for k, v in clips.items() if not v.is_file()]
    if unknown or missing:
        for k in unknown:
            print(f"error: --contrast {k}: no such clip alias", file=sys.stderr)
        for m in missing:
            print(f"error: clip not found: {m}", file=sys.stderr)
        return 1
    try:
        result, before, after = solve(clips, look, target_mean=a.target_mean,
                                      contrast_groups=groups, return_stats=True)
    except (FfmpegError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write_grade(out, result)
    print(f"{'alias':10} {'before mean/std':>16} {'after mean/std':>16} {'gamma':>7} {'contrast':>8}")
    for k, (g, c) in result.items():
        (m0, s0), (m1, s1) = before[k], after[k]
        print(f"{k:10} {m0:7.1f} /{s0:6.1f}  {m1:7.1f} /{s1:6.1f}  {g:7.3f} {c:8.3f}")
    off = [k for k, (m, _) in after.items() if abs(m - a.target_mean) > TOLERANCE]
    for k in off:
        print(f"warning: {k}: mean {after[k][0]:.1f} is more than {TOLERANCE:g} from "
              f"target {a.target_mean:g}")
    print("ok", out)
    return 1 if off else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
