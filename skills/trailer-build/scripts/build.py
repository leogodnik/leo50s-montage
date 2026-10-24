"""Build a trailer from trailer.json step by step.

usage: build.py <trailer.json> <step> [ids...]
steps: edl wav cards segments [ids] film assemble audio mux preview all
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import assemble as asm  # noqa: E402
from montagelib import ffm  # noqa: E402
from montagelib.cards import render_card  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.score import build_audio  # noqa: E402
from montagelib.segments import SegmentError, render_segments  # noqa: E402
from montagelib.timeline import timeline  # noqa: E402

STEPS = ("edl", "wav", "cards", "segments", "film", "assemble", "audio", "mux", "preview", "all")
ALL = ("wav", "cards", "segments", "film", "assemble", "audio", "mux", "preview")


def step_edl(p, ids):
    tl, total = timeline(p.edl, p.fps)
    for s, st in tl:
        hit = s.hit["type"] if s.hit else ""
        print(f"{s.id} {st:6.2f} +{s.d:4.1f} {s.kind:5} {s.src or ''} {s.text or ''} {hit}")
    print("total %.2f" % total)


def step_wav(p, ids):
    for alias, src in p.clips.items():
        dst = p.wav_dir / f"{alias}.wav"
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        ffm.extract_wav(src, dst)
        print("wav", dst)


def step_cards(p, ids):
    for s in p.edl:
        if s.kind in ("card", "title"):
            print("card", render_card(s, p))


def step_segments(p, ids):
    for sid, got, _ in render_segments(p, only=ids or None):
        print(sid, "frames", got)


def step_film(p, ids):
    print("film", asm.film_overlay(p))


def step_assemble(p, ids):
    _, total = timeline(p.edl, p.fps)
    print("assembled", asm.assemble(p), "expected length %.2f" % total)


def step_audio(p, ids):
    build_audio(p)


def step_mux(p, ids):
    print("muxed", asm.mux(p))


def step_preview(p, ids):
    print("preview", asm.preview(p, p.output["preview_height"]))


def main(argv):
    if len(argv) < 2 or argv[1] not in STEPS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path, step, ids = argv[0], argv[1], argv[2:]
    try:
        p = load_project(path)
        unknown = [i for i in ids if i not in {s.id for s in p.edl}]
        if unknown:
            print(f"error: unknown segment id(s): {' '.join(unknown)}", file=sys.stderr)
            return 1
        for d in (p.seg_dir, p.cards_dir, p.out_dir, p.wav_dir):
            d.mkdir(parents=True, exist_ok=True)
        for name in (ALL if step == "all" else (step,)):
            globals()[f"step_{name}"](p, ids)
    except (ProjectError, SegmentError, asm.StaleFileError, ffm.FfmpegError,
            FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
