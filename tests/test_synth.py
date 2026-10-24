import numpy as np
import pytest
from scipy import signal

from montagelib import synth
from montagelib.synth import EVENTS, INSTRUMENTS, SR


@pytest.mark.parametrize("name", sorted(INSTRUMENTS))
def test_instrument_length_and_finite(name):
    x = INSTRUMENTS[name](2.0)
    assert len(x) == 96000
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) <= 5


@pytest.mark.parametrize("name", sorted(EVENTS))
def test_events_finite(name):
    x = EVENTS[name]()
    assert len(x) > 0
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) <= 5


@pytest.mark.parametrize("bpm", [None, 60])
def test_heartbeat_bpm(bpm):
    x = synth.heartbeat(4.0) if bpm is None else synth.heartbeat(4.0, bpm=bpm)
    period = 60 / (58 if bpm is None else bpm)
    env = signal.sosfilt(signal.butter(2, 20 / (SR / 2), output="sos"), np.abs(x))
    rate = 1000
    e = env[::SR // rate]
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (len(e) - np.arange(len(e)))  # unbiased: undo the shrinking overlap
    lo, hi = int(0.4 * rate), int(1.6 * rate)  # past the 0.19 s double beat
    lag = (lo + int(np.argmax(ac[lo:hi]))) / rate
    assert abs(lag - period) <= 0.05 * period
