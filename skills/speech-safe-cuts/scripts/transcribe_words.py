"""Word timestamps for every clip of a project (or for given media files) -> words.json.

usage: transcribe_words.py <trailer.json | files...> [--out words.json] [--lang ru] [--model medium]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm, whisper  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--model", default="medium")
    a = ap.parse_args(argv)
    try:
        if len(a.inputs) == 1 and a.inputs[0].suffix == ".json":
            p = load_project(a.inputs[0])
            media, out = p.clips, a.out or p.build_dir / "words.json"
        else:
            media, out = {f.stem: f for f in a.inputs}, a.out or pathlib.Path("words.json")
        for alias, f in media.items():
            if not f.exists():
                raise FileNotFoundError(f"clip {alias}: {f} not found")
        res = whisper.transcribe(list(media.values()), a.lang, a.model, words=True)
    except (whisper.WhisperError, ProjectError, ffm.FfmpegError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    words = {}
    for alias, f in media.items():
        segs = res[str(f)]
        words[alias] = [w for s in segs for w in s["words"]]
        print(f"=== {alias}")
        for s in segs:
            print(f"  [{s['s']:5.2f}-{s['e']:5.2f}] {s['text']}")
            print("     " + " ".join(f"{w['w']}@{w['s']:.2f}-{w['e']:.2f}" for w in s["words"]))
    out.write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
    print("saved", out, {k: len(v) for k, v in words.items()})
    print(whisper.REMINDER)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
