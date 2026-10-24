"""Check that every speech segment survives the mix: each source syllable peak reappears
in the mix within 40 ms, and no score event sits within 120 ms of the last one.

usage: verify_mix.py <trailer.json> [--segment id ...] [--transcribe]
exit: 0 clean, 1 envelope problems, 2 missing input. --transcribe is advisory only.
"""
import argparse
import pathlib
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm, whisper  # noqa: E402
from montagelib.envelope import read_mono, rms_db, syllable_peaks  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402
from montagelib.timeline import timeline, visual_start  # noqa: E402
from montagelib.words import inside, load_words  # noqa: E402

EVENT_GAP = 0.12
MATCH = 0.04  # a source syllable must reappear in the mix within this distance
PUNCT = string.punctuation + "«»—–…"


def norm(text: str) -> list[str]:
    return [t.strip(PUNCT) for t in text.lower().split() if t.strip(PUNCT)]


def check_envelope(p, tl, segs):
    """Return problem count; prints one line per segment."""
    mt, mdb = rms_db(*read_mono(p.out_dir / "score.wav"))
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
            envs[s.src] = rms_db(*read_mono(wav))
        src = syllable_peaks(*envs[s.src], s.ss, s.ss + s.d)
        mix = syllable_peaks(mt, mdb, st - MATCH, st + s.d + MATCH)
        matched, lost = [], []
        for t in src:
            want = st + (t - s.ss)
            near = [m for m in mix if abs(m - want) <= MATCH]
            if near:
                matched.append(min(near, key=lambda m: abs(m - want)))
            else:
                lost.append(t)
        tail = [f"{sid}@{t:.2f}" for sid, t in events
                if matched and abs(t - matched[-1]) <= EVENT_GAP]
        flags = [f"syllable lost at {t:.2f}" for t in lost]
        if tail:
            flags.append("score event on last syllable")
        problems += len(flags)
        print(f"{s.id} {s.src} {st:6.2f}-{st + s.d:6.2f} peaks src/matched "
              f"{len(src)}/{len(matched)} events near tail: {' '.join(tail) or '-'}"
              + "".join(f"  <-- {f}" for f in flags))
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
        problems = check_envelope(p, tl, segs)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("problems:", problems)
    if a.transcribe:
        advisory_transcript(p, segs)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
