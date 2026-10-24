"""Synth instruments, one-shot events and DSP helpers for the score (48 kHz mono float)."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
rng = np.random.default_rng(7)  # deterministic noise: the same project gives the same score
# drone and riser bodies peak at 3-4; their output is divided by LOUD so every instrument stays
# under 1.5 (a layer gain of LOUD * g reproduces the original level g)
LOUD = 3.0


def tvec(d):
    return np.arange(int(d * SR)) / SR


def env_asr(n, a, r, s=1.0):
    """Linear attack/release envelope of n samples; a and r are clamped to n."""
    e = np.ones(n)
    ia, ir = min(int(a * SR), n), min(int(r * SR), n)
    if ia:
        e[:ia] = np.linspace(0, 1, ia)
    if ir:
        e[-ir:] *= np.linspace(1, 0, ir)
    return e * s


def expdec(d, tau):
    return np.exp(-tvec(d) / tau)


def saw(f, t):
    ph = np.cumsum(np.atleast_1d(f) * np.ones_like(t)) / SR if np.ndim(f) else f * t
    return 2 * (ph - np.floor(ph + 0.5))


def sine(f, t):
    ph = np.cumsum(np.atleast_1d(f) * np.ones_like(t)) / SR if np.ndim(f) else f * t
    return np.sin(2 * np.pi * ph)


def lp(x, fc, order=4):
    sos = signal.butter(order, fc / (SR / 2), btype="low", output="sos")
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc / (SR / 2), btype="high", output="sos")
    return signal.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return signal.sosfilt(sos, x)


def noise(d):
    return rng.standard_normal(int(d * SR))


def add(bus, x, at, gain=1.0):
    """Mix x into bus starting at `at` seconds (clipped to the bus; negative `at` trims the head)."""
    i = int(at * SR)
    if i < 0:
        x, i = x[-i:], 0
    n = min(len(x), len(bus) - i)
    if n > 0:
        bus[i:i + n] += x[:n] * gain


# ---------------------------------------------------------------- events

def boom(big=False):
    d = 3.5 if big else 2.5
    t = tvec(d)
    f = (70 if big else 60) * np.exp(-t * 2.2) + 28
    body = sine(f, t) * expdec(d, 0.9 if big else 0.6)
    thump = lp(noise(0.25), 700) * expdec(0.25, 0.05)
    x = body * 1.0
    add(x, thump, 0, 0.9)
    if big:
        x += 0.35 * lp(saw(36.7, t) + saw(36.9, t), 200) * expdec(d, 1.4)
    return np.tanh(x * 1.6) / 1.6


def braam(d=3.2):
    t = tvec(d)
    bend = 1 + 0.04 * np.exp(-t * 2.5)
    x = np.zeros_like(t)
    for f0, g in [(36.7, 1.0), (73.4, 0.8), (87.3, 0.45), (110, 0.5), (146.8, 0.25)]:
        for det in (-0.6, 0, 0.6):
            x += g * saw((f0 + det) * bend, t)
    e = np.minimum(t / 0.35, 1) * np.exp(-np.maximum(t - 1.2, 0) / 0.9)
    dark = lp(x, 160) * 1.2
    bright = hp(lp(x, 1400), 150) * np.minimum(t / 0.9, 1) * 0.5
    y = (dark + bright) * e
    return np.tanh(y * 0.9) / 0.9


def hit():
    d = 0.6
    t = tvec(d)
    x = hp(noise(d), 900) * expdec(d, 0.05) * 0.8 + sine(85 * np.exp(-t * 6) + 45, t) * expdec(d, 0.18)
    return np.tanh(x * 1.3) / 1.3


def titleboom():
    """Boom with a long braam under it (braam at 0.55/0.7 of the boom level)."""
    x = braam(4.0) * (0.55 / 0.7)
    add(x, boom(), 0, 1.0)
    return x


# ---------------------------------------------------------------- instruments

def riser(d=4.0):
    n = int(d * SR)
    t = tvec(d)
    x = np.zeros(n)
    nz = noise(d)
    chunk = int(0.03 * SR)
    for i in range(0, n, chunk):
        frac = i / n
        fc = 150 * (30 ** frac)
        lo, hi = fc * 0.7, min(fc * 1.4, 20000)
        x[i:i + chunk] = bp(nz[i:i + chunk], lo, hi)[: len(x[i:i + chunk])]
    x *= (t / d) ** 1.5 * 2.2
    tone = sine(110 * (4 ** (t / d)), t) * (t / d) ** 2 * 0.5
    # accelerating ticks
    tk = np.zeros(n)
    pos, gap = 0.0, 0.30
    while pos < d:
        add(tk, hp(noise(0.012), 2500) * expdec(0.012, 0.004), pos, 0.6)
        pos += gap
        gap = max(gap * 0.88, 0.045)
    return (x + tone + tk) / LOUD


def sweep_noise(d, f_of_t, lo=0.8, hi=1.25):
    """Band-passed noise whose centre follows f_of_t (Hz array), filter state carried across chunks."""
    n = int(d * SR)
    nz = noise(d)
    out = np.zeros(n)
    ch = int(0.02 * SR)
    zi = None
    for i in range(0, n, ch):
        fc = float(f_of_t[min(i, n - 1)])
        hi_f = min(fc * hi, 20000)
        sos = signal.butter(2, [fc * lo / (SR / 2), hi_f / (SR / 2)], btype="band", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = signal.sosfilt(sos, nz[i:i + ch], zi=zi)
        out[i:i + len(y)] = y
    return out


def whistle(d):
    """Falling bomb: pitch glides down, vibrato, breath noise, swelling louder towards impact."""
    t = tvec(d)
    f = 2300 * (0.21 ** (t / d)) * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    ph = np.cumsum(f) / SR
    tone = np.sin(2 * np.pi * ph) + 0.35 * np.sin(4 * np.pi * ph) + 0.12 * np.sin(6 * np.pi * ph)
    breath = sweep_noise(d, f) * 1.6
    amp = 0.08 + 0.92 * (t / d) ** 1.4
    x = (tone * 0.7 + breath) * amp
    x[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return np.tanh(x * 1.2) / 1.2


def heartbeat(d, bpm=58):
    n = int(d * SR)
    x = np.zeros(n)
    per = 60 / bpm
    pos = 0.0
    tt = tvec(0.5)
    th = sine(75 * np.exp(-tt * 9) + 38, tt) * expdec(0.5, 0.09)
    while pos < d:
        add(x, th, pos, 1.0)
        add(x, th, pos + 0.19, 0.6)
        pos += per
    return x


def ticks(d, gap=0.5):
    n = int(d * SR)
    x = np.zeros(n)
    pos = 0.0
    tk = bp(noise(0.01), 2000, 6000) * expdec(0.01, 0.003)
    while pos < d:
        add(x, tk, pos, 1.0)
        pos += gap
    return x


def drone(d):
    t = tvec(d)
    x = np.zeros_like(t)
    for f0, g in [(36.7, 1.0), (55.0, 0.45), (73.4, 0.3)]:
        for det in (-0.35, 0, 0.35):
            x += g * saw(f0 + det, t)
    x = lp(x, 220) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.13 * t))
    x += 0.5 * sine(36.7, t)
    # slow breathing high component
    x += 0.12 * hp(lp(saw(146.8, t) + saw(147.3, t), 900), 120) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + 1))
    return x / LOUD


OSTINATO_PATTERN = [73.4, 73.4, 73.4, 73.4, 87.3, 87.3, 65.4, 65.4]


def ostinato(d, bpm=140, pattern=OSTINATO_PATTERN):
    n = int(d * SR)
    x = np.zeros(n)
    step = 60 / bpm / 2
    tt = tvec(0.16)
    k, pos = 0, 0.0
    while pos < d:
        f = pattern[k % len(pattern)]
        note = lp(saw(f, tt) + saw(f * 1.004, tt) + 0.5 * saw(f * 2, tt), 700) * env_asr(len(tt), 0.004, 0.06)
        add(x, note, pos, 1.0)
        pos += step
        k += 1
    return np.tanh(x * 1.2) / 1.2


def reverb(x, rt=2.4, wet=0.35, tail=3.0):
    ir = lp(noise(tail), 3500) * np.exp(-tvec(tail) / (rt / 6.9))
    ir /= np.sqrt(np.sum(ir ** 2))
    return x + wet * signal.fftconvolve(x, ir)[: len(x)]


def read_wav(path) -> np.ndarray:
    """48 kHz 16-bit wav -> float (n, 2) in [-1, 1]; mono is duplicated to two channels."""
    sr, a = wavfile.read(str(path))
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {SR}")
    a = a.astype(np.float64) / 32768.0
    return np.stack([a, a], 1) if a.ndim == 1 else a[:, :2]


INSTRUMENTS = {"drone": drone, "heartbeat": heartbeat, "ticks": ticks,
               "ostinato": ostinato, "riser": riser, "whistle": whistle}
EVENTS = {"boom": lambda: boom(), "bigboom": lambda: boom(big=True), "braam": braam,
          "hit": hit, "titleboom": titleboom}
