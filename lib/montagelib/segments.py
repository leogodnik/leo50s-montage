"""Per-segment ffmpeg commands (frame-exact) and their parallel rendering."""
from __future__ import annotations

import shlex
from concurrent.futures import ThreadPoolExecutor

from . import ffm
from .cards import render_card
from .looks import load_look
from .project import Project, Segment
from .timeline import frames

CENTER = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
X264 = "-an -c:v libx264 -crf 14 -preset fast"


class SegmentError(Exception):
    def __init__(self, mismatches: list[tuple[str, int, int]]):
        self.mismatches = list(mismatches)
        super().__init__("segment frame count mismatch:\n" + "\n".join(
            f"{sid}: got {got} frames, expected {exp}" for sid, got, exp in self.mismatches))


def _q(path) -> str:
    return shlex.quote(str(path))


def _grade(alias: str, p: Project) -> str:
    g, c = p.grade.get(alias, (1.0, 1.0))
    return f"eq=gamma={g}:contrast={c},{load_look(p.look)}"


def _clip_chain(seg: Segment, p: Project) -> list[str]:
    fps = p.fps
    chain = []
    slow = next((f for f in seg.fx if f.startswith("slow")), None)
    if slow:
        k = int(slow[4:])
        chain.append(f"trim=start={seg.ss}:duration={seg.d / k},setpts=PTS-STARTPTS")
        chain.append(f"minterpolate=fps={fps * k}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
                     f"setpts={k}*PTS,fps={fps}")
    else:
        chain.append(f"trim=start={seg.ss}:duration={seg.d},setpts=PTS-STARTPTS")
    chain.append(_grade(seg.src, p))
    if "shake" in seg.fx:
        a = seg.shake
        chain.append(f"crop=1216:684:x='32+{a}*(26*sin(t*47)+18*sin(t*71+1)+8*sin(t*131))'"
                     f":y='18+{a}*(14*sin(t*53+2)+10*sin(t*89)+6*sin(t*149))',scale=1280:720")
    if "push" in seg.fx:
        rate = 0.0005 if slow else 0.0007
        chain.append(f"zoompan=z='1.0+{rate}*on':{CENTER}:d=1:s=1280x720:fps={fps}")
    if seg.fade_in > 0:
        chain.append(f"fade=t=in:st=0:d={seg.fade_in}")
    return chain


def segment_cmd(seg: Segment, p: Project) -> str:
    W, H, fps = p.width, p.height, p.fps
    out = _q(p.seg_dir / f"{seg.id}.mp4")
    n = frames(seg.d, fps)
    tail = f"scale={W}:{H}:flags=lanczos,setsar=1,fps={fps},format=yuv420p"
    color = f"-f lavfi -i color=c={{}}:s={W}x{H}:r={fps}"
    if seg.kind == "black":
        return (f"ffmpeg -v error -y {color.format('black')} -frames:v {n} "
                f"-vf \"{tail}\" {X264} {out}")
    if seg.kind == "white":
        # hold absolute white, then settle into black
        return (f"ffmpeg -v error -y {color.format('white')} -frames:v {n} "
                f"-vf \"fade=t=out:st=0.25:d=0.85,{tail}\" {X264} {out}")
    if seg.kind == "title" and seg.style == "painted":
        png = _q(p.cards_dir / f"title_paint_{seg.id}.png")
        if seg.bg == "clip":
            bg = f"-ss {seg.ss} -i {_q(p.clips[seg.bg_src])}"
            bg_grade = _grade(seg.bg_src, p) + ","
        else:
            bg = color.format("black")
            bg_grade = ""
        # slam in, slow drift, fade out
        zp = (f"zoompan=z='if(lt(on,5),1.28-0.056*on,1.0+0.0007*(on-5))':{CENTER}"
              f":d=1:s={W}x{H}:fps={fps}")
        fc = (f"[0:v]{bg_grade}scale={W}:{H},setsar=1[bg];[bg][1:v]overlay=0:0:format=auto,{zp},"
              f"fade=t=out:st={seg.d - 0.9}:d=0.9,{tail}")
        return (f"ffmpeg -v error -y {bg} -loop 1 -framerate {fps} -i {png} -frames:v {n} "
                f"-filter_complex \"{fc}\" {X264} {out}")
    if seg.kind in ("card", "title"):
        png = _q(p.cards_dir / f"{seg.id}.png")
        zp = f"zoompan=z='1.0+0.0004*on':{CENTER}:d=1:s={W}x{H}:fps={fps}"
        fo = 0.5 if seg.small else 0.6
        vf = f"{zp},fade=t=in:st=0:d=0.5,fade=t=out:st={seg.d - fo}:d={fo},{tail}"
        return (f"ffmpeg -v error -y -loop 1 -framerate {fps} -i {png} -frames:v {n} "
                f"-vf \"{vf}\" {X264} {out}")
    if seg.kind == "clip":
        vf = ",".join(_clip_chain(seg, p) + [tail])
        return (f"ffmpeg -v error -y -i {_q(p.clips[seg.src])} -frames:v {n} "
                f"-filter_complex \"[0:v]{vf}\" {X264} {out}")
    raise ValueError(f"{seg.id}: unknown kind {seg.kind!r}")


def render_segments(p: Project, only: list[str] | None = None,
                    workers=6) -> list[tuple[str, int, int]]:
    segs = [s for s in p.edl if only is None or s.id in only]
    for s in segs:
        if s.kind in ("card", "title"):
            render_card(s, p)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(ffm.run, [segment_cmd(s, p) for s in segs]))
    res = [(s.id, ffm.count_frames(p.seg_dir / f"{s.id}.mp4"), frames(s.d, p.fps)) for s in segs]
    bad = [r for r in res if r[1] != r[2]]
    if bad:
        raise SegmentError(bad)
    return res
