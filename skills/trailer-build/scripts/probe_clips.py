"""Survey source clips: duration, size, fps, loudness and a 4-frame contact sheet each.

usage: probe_clips.py <dir|files...> [--out <dir>]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

EXTS = (".mp4", ".mov")


def probe(f: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                        "-of", "json", str(f)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    st = j["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    return {"duration": float(j["format"]["duration"]), "w": st["width"], "h": st["height"],
            "fps": float(num) / float(den) if float(den) else 0.0}


def levels(f: Path) -> tuple[str, str]:
    r = subprocess.run(["ffmpeg", "-i", str(f), "-af", "volumedetect", "-vn", "-f", "null", "-"],
                       capture_output=True, text=True)
    get = lambda k: (re.search(rf"{k}: (-?[\d.]+|-inf) dB", r.stderr) or [None, "n/a"])[1]
    return get("mean_volume"), get("max_volume")


def sheet(f: Path, duration: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf",
                    f"fps=4/{duration},scale=320:-1,tile=4x1", "-frames:v", "1", str(out)],
                   check=True)


def main(argv):
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    paths = [Path(a) for a in argv]
    if len(paths) == 1 and paths[0].is_dir():
        files = sorted(f for f in paths[0].iterdir() if f.suffix.lower() in EXTS)
        out = out or paths[0] / "_probe"
    else:
        files = paths
        out = out or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            m = probe(f)
            mean, mx = levels(f)
            sheet(f, m["duration"], out / f"{f.stem}.jpg")
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as e:
            print(f"error: {f}: {e}", file=sys.stderr)
            return 1
        print(f"{f.name}  {m['duration']:.2f}s  {m['w']}x{m['h']}  {m['fps']:.3g}fps  "
              f"mean {mean} dB  max {mx} dB")
    print("contact sheets in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
