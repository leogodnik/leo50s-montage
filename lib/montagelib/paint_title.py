"""Hand-painted dripping title: tall condensed letters, ragged brush edges, streaks, long drips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def paint_title(text: str, out: Path, font_path: str, size=(1920, 1080), seed=3,
                drip_prob=0.03, stretch_to=620) -> Path:
    """Write white paint on a transparent background (RGBA PNG) to `out`."""
    W, H = size
    rng = np.random.default_rng(seed)

    def smooth_noise(shape, sigma):
        n = rng.standard_normal(shape).astype(np.float32)
        img = Image.fromarray(((n - n.min()) / (n.max() - n.min()) * 255).astype(np.uint8))
        a = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255
        return (a - a.mean()) / (a.std() + 1e-6)

    # 1. base glyphs, tracked, then stretched vertically so the letters are tall and narrow
    fsize = 430
    track = fsize * 0.07
    font = ImageFont.truetype(font_path.split("#", 1)[0], fsize)
    widths = [font.getlength(ch) for ch in text]
    total = sum(widths) + track * (len(text) - 1)
    canvas = Image.new("L", (W, 700), 0)
    d = ImageDraw.Draw(canvas)
    x = (W - total) / 2
    for ch, w in zip(text, widths):
        d.text((x, 60), ch, font=font, fill=255)
        x += w + track
    bbox = canvas.getbbox()
    glyphs = canvas.crop((0, bbox[1], W, bbox[3]))
    glyphs = glyphs.resize((W, int(glyphs.height * stretch_to / glyphs.height)), Image.LANCZOS)
    mask = np.zeros((H, W), np.float32)
    y0 = (H - glyphs.height) // 2 - 20
    mask[y0:y0 + glyphs.height] = np.asarray(glyphs, np.float32) / 255

    # 2. ragged brush edges: displace the threshold with correlated noise
    soft = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(2.5)), np.float32) / 255
    edge = soft + 0.09 * smooth_noise((H, W), 1.6) + 0.05 * smooth_noise((H, W), 5.0)
    ragged = (edge > 0.5).astype(np.float32)

    # 3. vertical brush streaks inside the letters (dry-brush gaps)
    col = smooth_noise((1, W), 1.2)[0]
    col2 = smooth_noise((1, W), 4.0)[0]
    streak = 0.9 + 0.1 * np.tanh(col * 1.2 + col2 * 0.6)
    streak2d = np.repeat(streak[None, :], H, 0) + 0.05 * smooth_noise((H, W), 1.5)
    gaps = col < -2.3  # rare thin dry gaps
    paint = ragged.copy()  # solid fill; streaks only modulate tone
    paint[:, gaps] *= 0.35

    # 4. drips from the bottom edges of letters
    drip = Image.new("L", (W, H), 0)
    dd = ImageDraw.Draw(drip)
    solid = mask > 0.5  # drips hang from the clean glyph bottoms only
    for x in range(2, W - 2):
        colm = solid[:, x]
        for yb in np.where(colm[:-1] & ~colm[1:])[0]:  # bottom edges (pixel on, next off)
            if rng.random() > drip_prob:
                continue
            L = int(min(rng.exponential(170) + 30, 480))
            if yb + L > H - 150:
                L = max(H - 150 - yb, 0)
            if L < 20:
                continue
            w = rng.uniform(2.5, 7)
            wob = rng.uniform(-0.3, 0.3)
            for k in range(L):
                yy = yb + k
                ww = w * (1 - 0.55 * k / L)
                xx = x + wob * np.sin(k / 37)
                dd.line([(xx - ww / 2, yy), (xx + ww / 2, yy)], fill=255, width=1)
            r = w * rng.uniform(0.6, 1.0)  # droplet at the end
            dd.ellipse([x - r, yb + L - r, x + r, yb + L + r * 1.3], fill=255)
    drip_a = np.asarray(drip.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    alpha = np.clip(paint + drip_a * 0.95, 0, 1)

    # 5. slight paint thickness variation and output
    tone = np.clip(0.86 + 0.14 * np.clip((streak2d - 0.8) / 0.2, 0, 1)
                   + 0.04 * np.clip(smooth_noise((H, W), 8.0), -1, 1), 0.8, 1.0)
    rgb = np.clip(246 * tone, 200, 248).astype(np.uint8)
    rgba = np.dstack([rgb, rgb, rgb, (alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(out)
    return Path(out)
