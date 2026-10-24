"""Spectrogram + waveform PNG of the score and its loudness (ebur128 I, LRA, Peak).

usage: audio_report.py <trailer.json | file.wav> [--out report.png]
"""
import argparse
import pathlib
import shlex
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib"))

from montagelib import ffm  # noqa: E402
from montagelib.project import ProjectError, load_project  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("src", help="trailer.json (uses out/score.wav) or a wav file")
    ap.add_argument("--out", help="PNG path (default: next to the wav, audio_report.png)")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src)
    try:
        if src.suffix.lower() == ".json":
            p = load_project(src)
            wav, out = p.out_dir / "score.wav", p.out_dir / "audio_report.png"
        else:
            wav, out = src, src.with_name("audio_report.png")
        out = pathlib.Path(a.out) if a.out else out
        if not wav.exists():
            raise FileNotFoundError(f"{wav} not found: run 'build.py <trailer.json> audio' first")
        q = shlex.quote
        ffm.run(f"ffmpeg -v error -y -i {q(str(wav))} -filter_complex "
                f"\"[0:a]asplit[a][b];[a]showspectrumpic=s=1600x400:legend=0:scale=log[s];"
                f"[b]showwavespic=s=1600x200:colors=white[w];[s][w]vstack\" "
                f"-frames:v 1 {q(str(out))}")
    except (ProjectError, ffm.FfmpegError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(wav), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"error: ebur128 failed\n{r.stderr[-2000:]}", file=sys.stderr)
        return 1
    summary = r.stderr.split("Summary:")[-1]
    for line in summary.splitlines():
        if line.strip().startswith(("I:", "LRA:", "Peak:")):
            print(line.strip())
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
