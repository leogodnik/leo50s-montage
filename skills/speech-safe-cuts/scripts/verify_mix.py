"""Check that every speech segment survives the mix: at each source syllable peak the mix
level follows the source (lost: more than 6 dB under, masked: more than 8 dB over the
segment's common speech gain), the gain itself is at most +8 dB (otherwise the score
buries the whole segment), and another segment's hit near or after the last syllable
lands at least 120 ms after its peak with the source there (or at the segment end, if
the hit is later) 7 dB under it.

usage: verify_mix.py <trailer.json> [--segment id ...] [--transcribe] [-v]
exit: 0 clean, 1 envelope problems, 2 missing input. --transcribe is advisory only.
"""
import argparse
import pathlib
import string
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm, whisper  # noqa: E402
from montagelib.envelope import (hit_on_syllable, read_mono, rms_db, smooth,  # noqa: E402
                                 syllable_peaks)
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.timeline import timeline, visual_start  # noqa: E402
from montagelib.words import inside, load_words  # noqa: E402

EVENT_GAP = 0.12  # hits of other segments from this long before the last syllable are checked
WIN = 0.02  # rms_db default window
MATCH = 0.04  # room before the segment start in the mix envelope
LOST_DB = 6  # mix this far under the source at a syllable: it is gone (v6 min -1.9)
MASKED_DB = 8  # mix this far over the source: something covers it (v6 max +5.7)
AGREE_DB = 3  # syllables within this of each other share the segment's speech gain
BURIED_DB = 8  # segment gain above this: the score buries the whole line (v6 max +6.0)
PUNCT = string.punctuation + "«»—–…"


def norm(text: str) -> list[str]:
    return [t.strip(PUNCT) for t in text.lower().split() if t.strip(PUNCT)]


def mix_envelope(mono, sr, st, ss, win=WIN):
    """Mix envelope on windows in phase with the source's (grid fixed to the clip start)."""
    n = int(round(win * sr))
    start = int(round(st * sr)) - int(round(ss * sr)) % n
    start -= n * (-(-int(round(2 * MATCH * sr)) // n))  # room for the early match margin
    while start < 0:
        start += n
    times, db = rms_db(mono[start:], sr, win)
    return times + start / sr, db


def speech_gain(devs):
    """Gain from source to mix shared by most syllables (ties: the louder one), so a few
    lost or masked syllables do not shift it."""
    best = max(devs, key=lambda d: (sum(abs(e - d) <= AGREE_DB for e in devs), d))
    return float(np.median([e for e in devs if abs(e - best) <= AGREE_DB]))


def check_envelope(p, tl, segs, verbose=False):
    """Return problem count; prints one line per segment."""
    mono, sr = read_mono(p.out_dir / "score.wav")
    events = [(s.id, visual_start(s, st) + (s.hit.get("offset") or 0)) for s, st in tl if s.hit]
    envs, problems = {}, 0
    for s, st in segs:
        if any(f.startswith("slow") for f in s.fx):
            print(f"{s.id} {s.src} (slow, no audio) skipped")
            continue
        wav = p.wav_dir / f"{s.src}.wav"
        if s.src not in envs:
            if not wav.exists():
                raise FileNotFoundError(f"wav/{s.src}.wav missing, run: build.py {p.path} wav")
            times, db = rms_db(*read_mono(wav))
            envs[s.src] = times, db, smooth(db)
        times, db, sdb = envs[s.src]
        src = syllable_peaks(times, db, s.ss, s.ss + s.d)
        mt, mdb = mix_envelope(mono, sr, st, s.ss)
        mdb = smooth(mdb)
        want = [st + (t - s.ss) for t in src]
        devs = [float(mdb[np.argmin(abs(mt - w))] - sdb[np.argmin(abs(times - t))])
                for t, w in zip(src, want)]
        gain = speech_gain(devs) if devs else 0.0
        flags, rows, heard = [], [], 0
        if gain > BURIED_DB:
            flags.append(f"segment buried under score (gain {gain:+.1f} dB)")
        for t, w, d in zip(src, want, devs):
            d -= gain
            rows.append(f"{t:.2f}:{d:+.1f}")
            if d < -LOST_DB:
                flags.append(f"syllable lost at {t:.2f} ({d:+.1f} dB)")
            elif d > MASKED_DB:
                flags.append(f"syllable masked at {t:.2f} ({d:+.1f} dB)")
            else:
                heard += 1
        tail, on_last = [], False
        for sid, t in events:
            if not want or sid == s.id or not want[-1] - EVENT_GAP <= t <= st + s.d + EVENT_GAP:
                continue
            on, drop = hit_on_syllable(times, db, src[-1], s.ss + (t - st), s.ss + s.d)
            tail.append(f"{sid}@{t:.2f}({drop:.1f} dB down)")
            on_last |= on
        if on_last:
            flags.append("score event on last syllable")
        problems += len(flags)
        span = f"{min(devs) - gain:+.1f}..{max(devs) - gain:+.1f}" if devs else "-"
        print(f"{s.id} {s.src} {st:6.2f}-{st + s.d:6.2f} peaks src/matched "
              f"{len(src)}/{heard} gain {gain:+.1f} dB, syllables {span} dB, "
              f"events near tail: {' '.join(tail) or '-'}"
              + "".join(f"  <-- {f}" for f in flags))
        if verbose and rows:
            print("    " + " ".join(rows))
    return problems


def advisory_transcript(p, segs):
    video = p.out_dir / f"{p.output['name']}.mp4"
    if not video.exists():
        print(f"transcribe: {video} missing, run: build.py {p.path} mux")
        return
    try:
        res = whisper.transcribe([video], words=False)[str(video)]
    except (whisper.WhisperError, ffm.FfmpegError) as e:
        print(f"transcribe: skipped ({e})")
        return
    text = " ".join(s["text"] for s in res)
    print("transcript:", text)
    wpath = p.build_dir / "words.json"
    if not wpath.exists():
        print(f"transcribe: {wpath} missing, no expected words to compare")
        return
    words, heard = load_words(wpath), set(norm(text))
    for s, _ in segs:
        exp = [w.w for w in inside(words.get(s.src, []), s.ss, s.ss + s.d)]
        got = [w for w in exp if norm(w) and norm(w)[0] in heard]
        share = f"{len(got)}/{len(exp)}" if exp else "-"
        print(f"{s.id} expected: {' '.join(exp) or '-'} | found {share} (advisory)")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", type=pathlib.Path)
    ap.add_argument("--segment", nargs="+", default=[])
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print each syllable's level against the source")
    a = ap.parse_args(argv)
    try:
        p = load_project(a.project)
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    tl, _ = timeline(p.edl, p.fps)
    if a.segment:
        unknown = set(a.segment) - {s.id for s, _ in tl}
        if unknown:
            print(f"error: unknown segment(s): {' '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        segs = [(s, st) for s, st in tl if s.id in a.segment and s.kind == "clip"]
    else:
        segs = [(s, st) for s, st in tl if s.kind == "clip" and s.speech]
    if not (p.out_dir / "score.wav").exists():
        print(f"error: out/score.wav missing, run: build.py {p.path} audio", file=sys.stderr)
        return 2
    try:
        problems = check_envelope(p, tl, segs, a.verbose)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("problems:", problems)
    if a.transcribe:
        advisory_transcript(p, segs)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
