"""Check every speech clip boundary against word timestamps and the RMS envelope.

usage: check_cuts.py <trailer.json> [--words words.json]
exit: 0 clean, 1 problems found, 2 missing input (words.json, alias, wav)
A segment whose last word runs into its end needs the next segment to be
trans "cut" without a hit; then the end boundary need not be quiet
("tail: hard cut ok"). The start boundary must always be quiet.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib.envelope import quiet_at, read_mono, rms_db  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.words import inside, load_words, straddling  # noqa: E402

TAIL = 0.15


class Missing(Exception):
    pass


def check(p, words):
    """Yield (segment, table line, problem count)."""
    envs = {}
    for i, s in enumerate(p.edl):
        if s.kind != "clip" or not s.speech:
            continue
        a, b = s.ss, s.ss + s.d
        if s.src not in words:
            raise Missing(f"words.json has no entry for clip {s.src!r} (segment {s.id}); "
                          f"run transcribe_words.py")
        ws = words[s.src]
        ins, cut = inside(ws, a, b), straddling(ws, a, b)
        head = f"{s.id} {s.src} {a:5.2f}-{b:5.2f} words: {' '.join(w.w for w in ins)}"
        if cut:
            head += " | straddling: " + " ".join(f"{w.w}({w.s}-{w.e})" for w in cut)
        if any(f.startswith("slow") for f in s.fx):
            yield s, head + " (slow, no audio)", 0
            continue
        if s.src not in envs:
            wav = p.wav_dir / f"{s.src}.wav"
            if not wav.exists():
                raise Missing(f"wav/{s.src}.wav missing, run: build.py {p.path} wav")
            envs[s.src] = rms_db(*read_mono(wav))
        times, db = envs[s.src]
        flags, notes = [], []
        if cut:
            flags.append("CUT WORD")
        nxt = p.edl[i + 1] if i + 1 < len(p.edl) else None
        hard_cut = nxt is not None and nxt.trans == "cut" and nxt.hit is None
        tail = bool(ins) and ins[-1].e > b - TAIL
        if not quiet_at(times, db, a):
            flags.append("boundary not quiet")
        elif not quiet_at(times, db, b):
            # a hard cut right after the last word is the accepted pattern
            if tail and hard_cut:
                notes.append("tail: hard cut ok")
            else:
                flags.append("boundary not quiet")
        if tail and nxt is not None and not hard_cut:
            flags.append(f"needs cut (tail word): next {nxt.id} must be trans cut, no hit")
        line = head + "".join(f"  <-- {f}" for f in flags) + "".join(f"  ({n})" for n in notes)
        yield s, line, len(flags)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", type=pathlib.Path)
    ap.add_argument("--words", type=pathlib.Path)
    a = ap.parse_args(argv)
    try:
        p = load_project(a.project)
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    wpath = a.words or p.build_dir / "words.json"
    try:
        if not wpath.exists():
            raise Missing(f"{wpath} missing, run: transcribe_words.py {p.path}")
        problems = 0
        for _, line, n in check(p, load_words(wpath)):
            print(line)
            problems += n
    except Missing as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("problems:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
