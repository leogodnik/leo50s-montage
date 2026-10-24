"""Print the RMS envelope (20 ms bars) of a wav span, or around one word of a clip.

usage: envelope.py <wav> --from T --to T [--step 0.02]
       envelope.py <trailer.json> --clip alias --word TEXT [--pad 0.4] [--which N]
onset_before is the start of the sound run that holds the word. In continuous speech
(no pause after the previous word) that run starts at the phrase, so the script also
prints dip_before, the last envelope dip since the previous word's start: a segment
starting mid-phrase starts there, the rule check_cuts applies.
"""
import argparse
import pathlib
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib.envelope import (bars, dip_between, onset_before, quiet_at,  # noqa: E402
                                 read_mono, rms_db, syllable_peaks)
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.words import load_words  # noqa: E402

PUNCT = string.punctuation + "«»—–…"


def norm(text: str) -> str:
    return text.strip(PUNCT + " ").lower()


def span(times, db, t0, t1):
    sel = (times >= t0 - 1e-9) & (times <= t1 + 1e-9)
    return bars(times[sel], db[sel])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=pathlib.Path)
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--clip")
    ap.add_argument("--word")
    ap.add_argument("--pad", type=float, default=0.4)
    ap.add_argument("--which", type=int, default=1)
    ap.add_argument("--words", type=pathlib.Path)
    a = ap.parse_args(argv)

    if a.input.suffix != ".json":
        if a.t0 is None or a.t1 is None:
            ap.error("a wav needs --from and --to")
        times, db = rms_db(*read_mono(a.input), win=a.step)
        print(span(times, db, a.t0, a.t1))
        return 0

    if not a.clip or not a.word:
        ap.error("a trailer.json needs --clip and --word")
    try:
        p = load_project(a.input)
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    wpath = a.words or p.build_dir / "words.json"
    if not wpath.exists():
        print(f"error: {wpath} missing, run: transcribe_words.py {p.path}", file=sys.stderr)
        return 2
    want = norm(a.word)
    clip_words = load_words(wpath).get(a.clip, [])
    hits = [w for w in clip_words if norm(w.w).startswith(want)]
    if len(hits) < a.which:
        print(f"error: word {a.word!r} #{a.which} not found in clip {a.clip!r}", file=sys.stderr)
        return 2
    w = hits[a.which - 1]
    wav = p.wav_dir / f"{a.clip}.wav"
    if not wav.exists():
        print(f"error: wav/{a.clip}.wav missing, run: build.py {p.path} wav", file=sys.stderr)
        return 2
    times, db = rms_db(*read_mono(wav), win=a.step)
    print(span(times, db, w.s - a.pad, w.e + a.pad))
    onset = onset_before(times, db, w.s)
    peaks = syllable_peaks(times, db, onset, w.e + 0.1)
    print(f"whisper: s={w.s} e={w.e}")
    print(f"onset_before = {onset:.2f}")
    print("syllable_peaks = [" + ", ".join(f"{t:.2f}" for t in peaks) + "]")
    prev = max((x for x in clip_words if x.s < w.s), key=lambda x: x.s, default=None)
    grid = times[(times >= prev.s) & (times <= w.s)] if prev is not None else []
    if prev is not None and not any(quiet_at(times, db, t) for t in grid):
        dip = dip_between(times, db, prev.s, w.s + 0.05)
        print(f"dip_before = {dip:.2f} (no pause after {prev.w!r}: a start mid-phrase goes here)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
