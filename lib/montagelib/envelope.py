"""RMS envelope of speech audio: where a word really starts, where it is quiet enough to cut."""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

FLOOR = -55.0  # display floor of `bars`
QUIET_ABS = -45.0  # dBFS: at or below this a window is quiet whatever the nearby peak
CLICK_WINDOWS = 1  # isolated loud 20 ms windows tolerated inside a quiet margin
SMOOTH = 3  # windows in the moving average before syllable peak picking
SYL_BELOW = 18  # dB under the loudest syllable of the span: quieter peaks are noise
HIT_GAP = 0.12  # s: a hit at least this long after the last syllable peak
HIT_DROP = 7  # dB the source must fall from that peak (v6 s09 braam 9.0 ok; "Тяните" 4.9 not)
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


def quiet_detail(times, db, t, margin=0.05, drop=25) -> tuple[bool, float, int]:
    """(quiet, threshold, loud windows) for the windows overlapping [t-margin, t+margin].

    Threshold: `drop` dB below the nearby peak, but never below QUIET_ABS (true silence
    in a clip is quiet even next to a click). Quiet: at most one window above it, and that
    one is a click, i.e. its neighbours inside the margin are below the threshold.
    """
    thr = max(peak_near(times, db, t) - drop, QUIET_ABS)
    w = _win(times)
    sel = np.flatnonzero((times < t + margin - EPS) & (times + w > t - margin + EPS))
    loud = db[sel] > thr
    n = int(loud.sum())
    return n == 0 or (n <= CLICK_WINDOWS and not np.any(loud[1:] & loud[:-1])), float(thr), n


def quiet_at(times, db, t, margin=0.05, drop=25) -> bool:
    """See quiet_detail."""
    return quiet_detail(times, db, t, margin, drop)[0]


def onset_before(times, db, t_guess, drop=25, search=1.5) -> float:
    """Real start of the word whisper places at t_guess (whisper lags on shouts).

    Finds the last >= 40 ms gap below the threshold in [t_guess-search, t_guess] (so a
    previous word is skipped), then returns the first window after it that stays above the
    threshold for 40 ms more. No gap in the span, or no such window: returns t_guess.
    """
    thr = peak_near(times, db, t_guess) - drop
    above = db > thr
    span = np.flatnonzero((times >= t_guess - search - EPS) & (times <= t_guess + EPS))
    gap = next((j for j in span[::-1]
                if j > span[0] and not above[j] and not above[j - 1]), None)
    if gap is None:
        return float(t_guess)
    for i in range(gap, span[-1] + 1):
        if above[i] and i + 2 < len(db) and above[i + 1] and above[i + 2]:
            return float(times[i])
    return float(t_guess)


def dip_between(times, db, t0, t1, prominence=2) -> float:
    """The envelope dip closest to t1 in [t0, t1]: the last local minimum at least
    `prominence` dB deep, or the quietest window if there is none.

    The last dip, not the deepest: in a shout the previous word's own syllable
    gaps can be deeper than the short dip right before the next word.
    """
    sel = np.flatnonzero((times >= t0 - EPS) & (times <= t1 + EPS))
    if len(sel) == 0:
        return float(t0)
    idx, _ = find_peaks(-db[sel], prominence=prominence)
    i = idx[-1] if len(idx) else int(np.argmin(db[sel]))
    return float(times[sel[i]])


def smooth(db, n=SMOOTH) -> np.ndarray:
    """Centered moving average over n windows (edges averaged over what exists)."""
    k = np.ones(n)
    return np.convolve(db, k, mode="same") / np.convolve(np.ones(len(db)), k, mode="same")


def syllable_peaks(times, db, t0, t1, prominence=6, below=SYL_BELOW) -> list[float]:
    """Syllable peaks in [t0, t1] on the smoothed envelope: at least `prominence` dB
    prominent and at most `below` dB under the loudest one (quieter peaks are noise)."""
    sel = np.flatnonzero((times >= t0 - EPS) & (times <= t1 + EPS))
    if len(sel) == 0:
        return []
    s = smooth(db[sel])
    idx, _ = find_peaks(s, prominence=prominence)
    if len(idx) == 0:
        return []
    top = s[idx].max()
    return [float(times[sel[i]]) for i in idx if s[i] >= top - below]


def hit_on_syllable(times, db, last, t_hit, end) -> tuple[bool, float]:
    """Does a hit at t_hit (clip time) land on the syllable peaking at `last`?

    The source is read at the hit, or at `end` (the cut) when the hit comes after it.
    Clear only if the hit is HIT_GAP after the peak and the source there is HIT_DROP dB
    under the syllable's top. Returns (on syllable, drop in dB).
    """
    top = db[(times >= last - 0.04 - EPS) & (times <= last + 0.04 + EPS)].max()
    t = min(t_hit, end)
    i = np.flatnonzero(times <= t + EPS)
    drop = float(top - db[i[-1]]) if len(i) else 0.0
    return t_hit < last + HIT_GAP or drop < HIT_DROP, drop
