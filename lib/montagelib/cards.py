"""Text cards drawn with Pillow (the ffmpeg build has no drawtext)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .paint_title import SIDE_MARGIN, load_font as _font, paint_title
from .project import Project, Segment

CARD_FILL = (228, 228, 228)
TITLE_FILL = (235, 235, 235)

def _tracked_width(text: str, font, track_em: float) -> float:
    widths = [font.getlength(ch) for ch in text]
    return sum(widths) + font.size * track_em * (len(text) - 1)


def _fitted(spec: str, size: int, text: str, track_em: float, max_w: float):
    font = _font(spec, size)
    w = _tracked_width(text, font, track_em)
    if w > max_w:
        font = _font(spec, max(1, int(size * max_w / w)))
    return font


def _draw_tracked(img, text, font, track_em, y, fill):
    d = ImageDraw.Draw(img)
    track = font.size * track_em
    widths = [font.getlength(ch) for ch in text]
    x = (img.width - (sum(widths) + track * (len(text) - 1))) / 2
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=font, fill=fill)
        x += w + track


def render_card(seg: Segment, p: Project) -> Path:
    W, H = p.width, p.height
    max_w = W - 2 * SIDE_MARGIN
    if seg.kind == "title" and seg.style == "painted":
        out = p.cards_dir / f"title_paint_{seg.id}.png"
        return paint_title(seg.text, out, p.fonts["title"], size=(W, H),
                           stretch_to=round(620 * H / 1080))
    if seg.kind not in ("card", "title"):
        raise ValueError(f"{seg.id}: render_card does not draw kind {seg.kind!r}")
    img = Image.new("RGB", (W, H), (0, 0, 0))
    if seg.kind == "title":
        font = _fitted(p.fonts["title"], 400, seg.text, 0.10, max_w)
        top, bottom = font.getbbox(seg.text)[1::2]
        _draw_tracked(img, seg.text, font, 0.10, H / 2 - (top + bottom) / 2, TITLE_FILL)
    elif seg.lines:
        gap = 34
        fonts = [_fitted(p.fonts["card"], sz, t, 0.28, max_w) for t, sz in seg.lines]
        boxes = [f.getbbox(t) for (t, _), f in zip(seg.lines, fonts)]
        heights = [b[3] - b[1] for b in boxes]
        y = H / 2 - (sum(heights) + gap * (len(heights) - 1)) / 2
        for (t, _), f, b, h in zip(seg.lines, fonts, boxes, heights):
            _draw_tracked(img, t, f, 0.28, y - b[1], CARD_FILL)
            y += h + gap
    else:
        font = _fitted(p.fonts["card"], 50 if seg.small else 58, seg.text, 0.28, max_w)
        top, bottom = font.getbbox(seg.text)[1::2]
        _draw_tracked(img, seg.text, font, 0.28, H / 2 - (top + bottom) / 2, CARD_FILL)
    out = p.cards_dir / f"{seg.id}.png"
    img.save(out)
    return out
