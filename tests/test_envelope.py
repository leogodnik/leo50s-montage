import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from montagelib import ffm
from montagelib.envelope import (bars, dip_between, onset_before, quiet_at, rms_db,
                                 syllable_peaks)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не найден в PATH")
    tmp_path = tmp_path_factory.mktemp("env")
    clip = ffm.make_test_clip(tmp_path / "t.mp4", 2.0, tone_hz=440,
                              bursts=[(0.5, 0.8), (1.2, 1.5)])
    wav = ffm.extract_wav(clip, tmp_path / "t.wav")
    sr, x = wavfile.read(wav)
    return rms_db(x.astype(float).mean(1) / 32768, sr)  # mono average, as callers do


def test_quiet_between_bursts(env):
    times, db = env
    assert quiet_at(times, db, 1.0)
    assert not quiet_at(times, db, 0.6)


def test_onset_before_finds_real_start(env):
    times, db = env
    assert abs(onset_before(times, db, t_guess=0.75) - 0.5) <= 0.03
    # the previous word inside the search span is not taken for this word's onset
    assert abs(onset_before(times, db, t_guess=1.25) - 1.2) <= 0.03


def test_syllable_peaks_count(env):
    times, db = env
    assert len(syllable_peaks(times, db, 0.3, 1.7)) == 2


def test_bars_has_one_line_per_window(env):
    times, db = env
    lines = bars(times, db).splitlines()
    assert len(lines) == len(times)
    assert lines[0].split()[0] == "0.00"
    assert np.isclose(times[1] - times[0], 0.02)


def test_onset_before_long_word(tmp_path, ffmpeg_ok):
    # the real start lies more than 0.5 s before whisper's guess
    clip = ffm.make_test_clip(tmp_path / "long.mp4", 2.0, tone_hz=440, bursts=[(0.5, 1.6)])
    sr, x = wavfile.read(ffm.extract_wav(clip, tmp_path / "long.wav"))
    times, db = rms_db(x.astype(float).mean(1) / 32768, sr)
    assert abs(onset_before(times, db, t_guess=1.2) - 0.5) <= 0.03


def test_dip_between_finds_local_minimum(env):
    times, db = env
    assert 0.8 <= dip_between(times, db, 0.6, 1.3) < 1.2


def test_dip_between_prefers_last_dip():
    # a deep syllable gap at 0.40 and a shallower dip right before the next word at 0.80
    times = np.arange(0, 1.0, 0.02)
    db = np.full(len(times), -12.0)
    db[20] = -29.0
    db[40] = -20.0
    assert abs(dip_between(times, db, 0.2, 0.9) - 0.80) < 1e-9
    assert abs(dip_between(times, db, 0.2, 0.7) - 0.40) < 1e-9
