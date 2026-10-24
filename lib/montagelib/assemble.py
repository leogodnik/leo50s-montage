"""Final picture: film-damage layer, xfade assembly with post chain, mux and preview."""
from __future__ import annotations

import shlex
import subprocess
import wave
from pathlib import Path

import numpy as np

from . import ffm
from .project import Project
from .timeline import frames, timeline


def _q(path) -> str:
    return shlex.quote(str(path))


def film_overlay(p: Project) -> Path:
    """Dust and scratches layer: mid-grey is neutral for the 'overlay' blend."""
    W, H, fps = p.width, p.height, p.fps
    _, total = timeline(p.edl, fps)
    n = frames(total, fps) + fps
    p.out_dir.mkdir(parents=True, exist_ok=True)
    out = p.out_dir / "film.mp4"
    cmd = (f"ffmpeg -v error -y -f rawvideo -pix_fmt gray -s {W}x{H} -r {fps} -i - "
           f"-c:v libx264 -crf 10 -preset fast -pix_fmt gray {_q(out)}")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    rg = np.random.default_rng(42)
    scratches = []  # [x, frames_left, drift, brightness, width]
    try:
        for _ in range(n):
            f = np.full((H, W), 128, np.uint8)
            if rg.random() < 0.12:
                scratches.append([rg.uniform(60, W - 60), rg.integers(3, 14), rg.normal(0, 0.8),
                                  rg.integers(175, 235), rg.integers(1, 3)])
            for sc in scratches:
                x, w = int(sc[0]), int(sc[4])
                y0 = rg.integers(0, H // 3) if rg.random() < 0.3 else 0
                y1 = H - rg.integers(0, H // 3) if rg.random() < 0.3 else H
                f[y0:y1, max(x, 0):min(x + w, W)] = sc[3]
                sc[0] += sc[2]
                sc[1] -= 1
            scratches = [sc for sc in scratches if sc[1] > 0]
            for _ in range(rg.integers(0, 5)):  # dust specks, one frame each
                x, y = rg.integers(0, W - 8), rg.integers(0, H - 8)
                r = rg.integers(1, 4)
                f[y:y + r, x:x + r + rg.integers(0, 3)] = rg.integers(30, 80)
            if rg.random() < 0.02:  # occasional short hair
                x, y = rg.integers(0, W - 40), rg.integers(0, H - 40)
                L = rg.integers(12, 36)
                for k in range(L):
                    f[min(y + k, H - 1), min(x + int(k * rg.uniform(0.2, 0.6)), W - 1)] = 40
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise ffm.FfmpegError(cmd, err[-2000:])
    return out


def _graph(p: Project, tl) -> str:
    W, H = p.width, p.height
    # one common timebase, so xfade accepts both segments and concat output
    parts = [f"[{i}:v]settb=AVTB[v{i}]" for i in range(len(tl))]
    prev = "[v0]"
    for i, (s, start) in enumerate(tl):
        if i == 0:
            continue
        trans = "fade" if s.trans == "cut" else s.trans
        lbl = f"[x{i}]"
        if s.ov == 0:
            # a zero-length xfade ends the stream there; butt-join instead
            parts.append(f"{prev}[v{i}]concat=n=2:v=1:a=0{lbl}")
        else:
            parts.append(f"{prev}[v{i}]xfade=transition={trans}:duration={s.ov:.6f}"
                         f":offset={start:.6f}{lbl}")
        prev = lbl
    parts.append(f"[{len(tl)}:v]format=yuv420p[ovl]")
    parts.append(f"{prev}eq=brightness='0.012*(sin(41*t)+sin(97*t))':eval=frame,"
                 f"pad=iw+8:ih+8:4:4:black,crop={W}:{H}:x='4+2*sin(13*t)+1.5*sin(29*t)'"
                 f":y='4+1.5*cos(17*t)+1.2*sin(37*t)'[wv]")
    parts.append("[wv][ovl]blend=all_mode=overlay:all_opacity=0.55:shortest=1[flm]")
    post = "[flm]lutyuv=u=128:v=128,noise=c0s=11:c0f=t+u,"
    if p.letterbox:  # None and 0 both mean no bars (drawbox h=0 fills the frame)
        bar = p.letterbox
        post += (f"drawbox=x=0:y=0:w={W}:h={bar}:color=black:t=fill,"
                 f"drawbox=x=0:y={H - bar}:w={W}:h={bar}:color=black:t=fill,")
    post += "format=yuv420p[vout]"
    return ";".join(parts + [post])


def assemble(p: Project) -> Path:
    tl, total = timeline(p.edl, p.fps)
    film = p.out_dir / "film.mp4"
    if not film.exists() or ffm.count_frames(film) < frames(total, p.fps):
        film_overlay(p)
        print("film layer regenerated (missing or shorter than the timeline)")
    inputs = " ".join(f"-i {_q(p.seg_dir / f'{s.id}.mp4')}" for s, _ in tl)
    inputs += f" -i {_q(film)}"
    fcpath = p.build_dir / "assemble.txt"
    fcpath.write_text(_graph(p, tl), encoding="utf-8")
    out = p.out_dir / "video_only.mp4"
    ffm.run(f"ffmpeg -v error -y {inputs} -filter_complex_script {_q(fcpath)} -map {_q('[vout]')} "
            f"-r {p.fps} -c:v libx264 -crf {p.output['crf']} -preset medium {_q(out)}")
    return out


def silent_score(p: Project) -> Path:
    """TODO task 6: placeholder score, silent stereo 48 kHz 16-bit, total + 1 s long."""
    _, total = timeline(p.edl, p.fps)
    sr = 48000
    out = p.out_dir / "score.wav"
    p.out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(4 * int(round((total + 1.0) * sr))))
    return out


def mux(p: Project) -> Path:
    video, score = p.out_dir / "video_only.mp4", p.out_dir / "score.wav"
    if not score.exists():
        raise FileNotFoundError(f"{score} not found: run the 'audio' step first")
    out = p.out_dir / f"{p.output['name']}.mp4"
    ffm.run(f"ffmpeg -v error -y -i {_q(video)} -i {_q(score)} -map 0:v -map 1:a "
            f"-c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart {_q(out)}")
    return out


def preview(p: Project, height=720) -> Path:
    src = p.out_dir / f"{p.output['name']}.mp4"
    out = p.out_dir / f"{p.output['name']}_{height}p.mp4"
    ffm.run(f"ffmpeg -v error -y -i {_q(src)} -vf scale=-2:{height} "
            f"-c:v libx264 -crf 23 -preset medium -c:a copy -movflags +faststart {_q(out)}")
    return out
