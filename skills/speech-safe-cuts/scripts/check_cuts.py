"""Check every speech clip boundary against word timestamps and the RMS envelope.

usage: check_cuts.py <trailer.json> [--words words.json]
exit: 0 clean, 1 problems found, 2 missing input (words.json, alias, wav)
Start: passes when no word straddles it and the envelope onset of the first
word inside is at most 30 ms before it. In continuous speech (no pause before
the word, so no onset) it must sit at most 20 ms after the last envelope dip
between the previous word's start and the word. A loud start that passes gets the note
"start: not quiet (previous sound tail[, dip T])".
End: checked at the rendered end ss + d snapped to the frame grid. It must be quiet
(quiet_at: 25 dB under the nearby peak or at most -45 dBFS, one click window tolerated)
unless the last word runs into it: whisper ends it within 0.15 s of the end, or the
sound never goes quiet between the word and the end. Such a tail needs no quiet end,
but the next segment's hit must land at least 0.12 s after the last syllable peak
("hit on last syllable" otherwise). Visual transitions do not fade the outgoing speech.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib.envelope import (dip_between, onset_before, peak_near, quiet_at,  # noqa: E402
                                 quiet_detail, read_mono, rms_db, syllable_peaks)
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.timeline import snap, timeline, visual_start  # noqa: E402
from montagelib.words import inside, load_words, straddling  # noqa: E402

TAIL = 0.15  # whisper's last word ending this close to the end runs into it
EVENT_GAP = 0.12  # a hit closer than this after the last syllable peak masks it (as verify_mix)
DROP = 25  # quiet_at drop, for the diagnostic note
TOL = 0.03  # word timestamp tolerance, also the start slack
DIP_SLACK = 0.02
NO_PREV = 0.6


class Missing(Exception):
    pass


def _tail_note(nxt, starts, s, a, last, flags, notes):
    """Hit rule for a segment whose last word runs into its end."""
    if nxt.hit is None:
        notes.append("tail: hard cut ok" if nxt.trans == "cut"
                     else f"tail: next {nxt.id} {nxt.trans}, no hit")
        return
    # clip time of the next segment's hit, on the same timeline verify_mix uses
    hit_t = a + visual_start(nxt, starts[nxt.id]) + (nxt.hit.get("offset") or 0) - starts[s.id]
    if last is not None and hit_t < last + EVENT_GAP:
        flags.append(f"hit on last syllable: next {nxt.id} hit at {hit_t:.2f}, "
                     f"last syllable {last:.2f}")
    else:
        notes.append(f"tail: next {nxt.id} hit {hit_t - last:.2f} s after last syllable"
                     if last is not None else f"tail: next {nxt.id} hit, no syllable peak")


def check(p, words):
    """Yield (segment, table line, problem count)."""
    tl, _ = timeline(p.edl, p.fps)
    starts = {seg.id: st for seg, st in tl}
    envs = {}
    for i, s in enumerate(p.edl):
        if s.kind != "clip" or not s.speech:
            continue
        a, b = s.ss, s.ss + snap(s.d, p.fps)  # the end the render produces
        if s.src not in words:
            raise Missing(f"words.json has no entry for clip {s.src!r} (segment {s.id}); "
                          f"run transcribe_words.py")
        ws = words[s.src]
        ins, cut = inside(ws, a, b, TOL), straddling(ws, a, b, TOL)
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
        start_quiet = quiet_at(times, db, a)
        onset = onset_before(times, db, ins[0].s) if ins else None
        if any(w.s < a < w.e for w in cut):
            if not start_quiet:
                flags.append("start not quiet")
        elif onset is not None and onset != ins[0].s and onset < a - TOL:
            flags.append("start cuts into a word")
        elif not start_quiet and onset is None:
            flags.append("start not quiet")
        elif not start_quiet and onset == ins[0].s:
            # no pause before the word: onset_before fell back to whisper's start,
            # which lags on shouts; the envelope dip before the word decides
            t0 = max((w.s for w in ws if w.s < ins[0].s), default=ins[0].s - NO_PREV)
            t_min = dip_between(times, db, t0, ins[0].s + 0.05)
            if a <= t_min + DIP_SLACK + 1e-9:
                notes.append(f"start: not quiet (previous sound tail, dip {t_min:.2f})")
            else:
                flags.append(f"start cuts into a word (dip {t_min:.2f})")
        elif not start_quiet:
            notes.append("start: not quiet (previous sound tail)")
        end_quiet, thr, loud = quiet_detail(times, db, b)
        tail = bool(ins) and ins[-1].e > b - TAIL
        if not end_quiet and ins and not tail:
            # whisper ends words early on shouts: no quiet window between the last word
            # and the end means its sound runs on into the cut
            grid = times[(times >= ins[-1].e) & (times <= b)]
            tail = not any(quiet_at(times, db, t) for t in grid)
            if tail:
                notes.append("tail: sound runs on to the end")
        if end_quiet and loud:
            notes.append(f"end: quiet, {loud} click window tolerated")
        elif end_quiet and thr > peak_near(times, db, b) - DROP + 1e-9:
            notes.append(f"end: quiet by the {thr:.0f} dBFS floor")
        if not end_quiet and not tail:
            flags.append("end not quiet")
        if tail and nxt is not None:
            peaks = syllable_peaks(times, db, a, b)
            _tail_note(nxt, starts, s, a, peaks[-1] if peaks else None, flags, notes)
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
