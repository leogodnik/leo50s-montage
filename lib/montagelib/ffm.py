"""Thin wrappers over ffmpeg/ffprobe."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


class FfmpegError(Exception):
    def __init__(self, cmd: str, stderr_tail: str):
        self.cmd = cmd
        self.stderr_tail = stderr_tail
        super().__init__(f"command failed: {cmd}\n{stderr_tail}")


def run(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise FfmpegError(cmd, r.stderr[-2000:])


def _probe(path, entries: str, extra: str = "") -> str:
    cmd = (f"ffprobe -v error {extra} -show_entries {entries} "
           f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(path))}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise FfmpegError(cmd, r.stderr[-2000:])
    return r.stdout.strip()


def count_frames(path) -> int:
    return int(_probe(path, "stream=nb_read_frames", "-count_frames -select_streams v:0"))


def duration(path) -> float:
    return float(_probe(path, "format=duration"))


def extract_wav(src, dst, sr=48000) -> Path:
    run(f"ffmpeg -v error -y -i {shlex.quote(str(src))} -vn -ac 2 -ar {sr} "
        f"-c:a pcm_s16le {shlex.quote(str(dst))}")
    return Path(dst)


def grab_frame(src, t, dst, vf: str | None = None) -> Path:
    filt = f' -vf "{vf}"' if vf else ""
    run(f"ffmpeg -v error -y -ss {t} -i {shlex.quote(str(src))}{filt} -frames:v 1 "
        f"{shlex.quote(str(dst))}")
    return Path(dst)


def make_test_clip(dst, seconds: float, fps=24, size="1280x720", brightness: float = 0.0,
                   tone_hz: int | None = None,
                   bursts: list[tuple[float, float]] | None = None) -> Path:
    """testsrc2 video + 48 kHz stereo audio: silence, or a sine heard only inside `bursts`."""
    video = f"testsrc2=size={size}:rate={fps},eq=brightness={brightness}"
    if tone_hz is None:
        audio = "anullsrc=channel_layout=stereo:sample_rate=48000"
    else:
        audio = f"sine=frequency={tone_hz}:sample_rate=48000"
        if bursts is not None:
            inside = "+".join(f"between(t,{a},{b})" for a, b in bursts) or "0"
            audio += f",volume=volume=0:enable='not({inside})'"
    run(f"ffmpeg -v error -y -f lavfi -i \"{video}\" -f lavfi -i \"{audio}\" -t {seconds} "
        f"-pix_fmt yuv420p -c:v libx264 -crf 18 -c:a aac -ac 2 -ar 48000 "
        f"{shlex.quote(str(dst))}")
    return Path(dst)
