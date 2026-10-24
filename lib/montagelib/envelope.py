"""RMS envelope of speech audio: where a word really starts, where it is quiet enough to cut."""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

FLOOR = -55.0
EPS = 1e-9


def read_mono(path) -> tuple[np.ndarray, int]:
    """16-bit wav -> mono float in [-1, 1] (channels averaged), sample rate."""
    sr, x = wavfile.read(str(path))
    x = x.astype(float) / 32768
    return (x.mean(1) if x.ndim == 2 else x), sr


def rms_db(x: np.ndarray, sr: int, win: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping windows; times[i] is the start of window i."""
    n = int(round(win * sr))
    k = len(x) // n
    frames = np.asarray(x[:k * n], dtype=float).reshape(k, n)
    rms = np.sqrt((frames ** 2).mean(1))
    return np.arange(k) * n / sr, 20 * np.log10(rms + 1e-9)


def bars(times, db, floor=FLOOR) -> str:
    return "\n".join(f"{t:6.2f} {d:6.1f} " + "#" * int(max(0, d - floor))
                     for t, d in zip(times, db))


def _win(times) -> float:
    return float(times[1] - times[0]) if len(times) > 1 else 0.02


def peak_near(times, db, t, radius=0.6) -> float:
    sel = (times >= t - radius - EPS) & (times <= t + radius + EPS)
    return float(db[sel].max()) if sel.any() else FLOOR - 100


def quiet_at(times, db, t, margin=0.05, drop=25) -> bool:
    """Every window overlapping [t-margin, t+margin] is `drop` dB below the nearby peak."""
    peak = peak_near(times, db, t)
    if peak < FLOOR + 10:
        return True
    w = _win(times)
    sel = (times < t + margin - EPS) & (times + w > t - margin + EPS)
    return bool(np.all(db[sel] <= peak - drop))


def onset_before(times, db, t_guess, drop=25, search=0.5) -> float:
    """Real start of the word whisper places at t_guess (whisper lags on shouts).

    Scans forward from t_guess-search, or from the last >= 40 ms gap below the threshold
    before t_guess if that is later (so a previous word inside the search span is skipped),
    and returns the first window that stays above the threshold for 40 ms more.
    """
    thr = peak_near(times, db, t_guess) - drop
    above = db > thr
    span = np.flatnonzero((times >= t_guess - search - EPS) & (times <= t_guess + EPS))
    if len(span) == 0:
        return float(t_guess)
    first = span[0]
    for j in span[::-1]:
        if j > 0 and not above[j] and not above[j - 1]:
            first = max(first, j)
            break
    for i in range(first, span[-1] + 1):
        if above[i] and i + 2 < len(db) and above[i + 1] and above[i + 2]:
            return float(times[i])
    return float(t_guess)


def syllable_peaks(times, db, t0, t1, prominence=6) -> list[float]:
    sel = np.flatnonzero((times >= t0 - EPS) & (times <= t1 + EPS))
    if len(sel) == 0:
        return []
    idx, _ = find_peaks(db[sel], prominence=prominence)
    return [float(times[sel[i]]) for i in idx]
