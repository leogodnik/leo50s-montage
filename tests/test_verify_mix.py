import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from montagelib import ffm
from montagelib.assemble import silent_score
from montagelib.project import load_project
from montagelib.timeline import timeline

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "skills" / "speech-safe-cuts" / "scripts" / "verify_mix.py"


def _mix(mini_project, ss, d=1.9, bursts=((0.5, 0.8), (1.2, 1.5)), amp=None, **seg_fields):
    """Speech clip a (bursts 0.5-0.8, 1.2-1.5, or a 440 Hz tone under amp(t)) from `ss`
    laid into out/score.wav at its start."""
    p = mini_project
    if amp is None:
        clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", 2.0, tone_hz=440,
                                  bursts=list(bursts))
        ffm.extract_wav(clip, p.wav_dir / "a.wav")
    else:
        sr = 48000
        t = np.arange(int(2.5 * sr)) / sr
        x = (amp(t) * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wavfile.write(p.wav_dir / "a.wav", sr, np.stack([x, x], 1))
    data = json.loads(p.path.read_text(encoding="utf-8"))
    data["edl"][2].update(ss=ss, d=d)
    for e in data["edl"]:
        e.update(seg_fields.get(e["id"], {}))
    p.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    p = load_project(p.path)
    tl, _ = timeline(p.edl, p.fps)
    seg, start = next((s, st) for s, st in tl if s.id == "s02")
    sr, mix = wavfile.read(silent_score(p))
    _, src = wavfile.read(p.wav_dir / "a.wav")
    i, n = int(round(start * sr)), int(round(seg.d * sr))
    mix[i:i + n] = src[int(round(seg.ss * sr)):][:n]
    wavfile.write(p.out_dir / "score.wav", sr, mix)
    return p, start, sr


@pytest.fixture
def mixed(mini_project):
    return _mix(mini_project, 0)


def _verify(p):
    return subprocess.run([sys.executable, str(VERIFY), str(p.path)],
                          capture_output=True, text=True)


def test_all_syllables_heard_passes(mixed):
    p, _, _ = mixed
    r = _verify(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "peaks src/matched 2/2" in r.stdout
    assert "problems: 0" in r.stdout


def test_masked_syllable_fails(mixed):
    p, start, sr = mixed
    score = p.out_dir / "score.wav"
    _, mix = wavfile.read(score)
    mix[int((start + 1.15) * sr):int((start + 1.55) * sr)] = 0
    wavfile.write(score, sr, mix)
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "syllable lost" in r.stdout


def test_off_grid_segment_matches(mini_project):
    # ss is not on the 20 ms grid: source and mix windows must still share the phase
    p, _, _ = _mix(mini_project, 0.013)
    r = _verify(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "peaks src/matched 2/2" in r.stdout


def _add(p, t0, t1, sig):
    """Add sig(t) (float, 1.0 = full scale) to out/score.wav on the timeline span [t0, t1)."""
    score = p.out_dir / "score.wav"
    sr, mix = wavfile.read(score)
    i0, i1 = int(t0 * sr), int(t1 * sr)
    t = np.arange(i1 - i0) / sr
    x = mix[i0:i1].astype(float) + (32767 * sig(t))[:, None]
    mix[i0:i1] = np.clip(x, -32768, 32767).astype(mix.dtype)
    wavfile.write(score, sr, mix)


def _speech_amp(p):
    """Peak amplitude of the speech (the mix carries the source 1:1)."""
    _, src = wavfile.read(p.wav_dir / "a.wav")
    return float(np.abs(src).max()) / 32768


def test_drone_filling_syllable_gaps_passes(mixed):
    # v6 at duck 0.5: a drone 4.4 dB under the speech fills the gaps between syllables
    # (their contrast drops under 6 dB); every syllable is still there at its own level
    p, start, sr = mixed
    a = 0.6 * _speech_amp(p)
    _add(p, start, start + 1.9, lambda t: a * np.sin(2 * np.pi * 110 * t))
    r = _verify(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "problems: 0" in r.stdout


def test_attenuated_syllable_fails(mixed):
    # the second syllable is still a peak in the mix, but 12 dB under the source
    p, start, sr = mixed
    score = p.out_dir / "score.wav"
    _, mix = wavfile.read(score)
    i0, i1 = int((start + 1.15) * sr), int((start + 1.55) * sr)
    mix[i0:i1] = (mix[i0:i1] / 4).astype(mix.dtype)
    wavfile.write(score, sr, mix)
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "syllable lost at 1." in r.stdout


def _noise(a):
    rng = np.random.default_rng(1)
    return lambda t: a * rng.uniform(-1, 1, len(t))


THREE = ((0.3, 0.5), (0.8, 1.0), (1.3, 1.5))


def test_syllable_under_loud_noise_fails(mini_project):
    # a noise burst 12 dB over the second of three syllables masks it
    # (with two syllables a tie would be read as the other one lost: still a problem)
    p, start, sr = _mix(mini_project, 0, bursts=THREE)
    _add(p, start + 0.75, start + 1.05, _noise(4 * _speech_amp(p)))
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "syllable masked at 0.8" in r.stdout


def test_masked_first_syllable_fails_even_under_own_hit(mini_project):
    # no exemption for the segment's own incoming hit: +12 dB on the first syllable of a
    # short insert buries it
    p, start, sr = _mix(mini_project, 0.2, d=1.6, bursts=THREE, s02={"hit": "hit"})
    vs = start + 0.125
    _add(p, vs, vs + 0.3, _noise(4 * _speech_amp(p)))
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "syllable masked" in r.stdout


def test_next_hit_on_last_syllable_fails(mini_project):
    # d 1.55: the next segment's hit (hard cut, 1.50 s in clip time) lands on the last
    # syllable peak (1.46 s)
    p, _, _ = _mix(mini_project, 0, d=1.55)
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "score event on last syllable" in r.stdout and "s03@" in r.stdout


def _sustained(t):
    # last syllable peaks at 1.2-1.3, dips 8 dB, then keeps sounding 2 dB under the peak
    amp = np.where((t >= 0.5) & (t < 0.8), 0.5, 0.0)
    amp = np.where((t >= 1.2) & (t < 1.3), 0.5, amp)
    amp = np.where((t >= 1.3) & (t < 1.4), 0.2, amp)
    return np.where((t >= 1.4) & (t < 2.0), 0.4, amp)


def test_next_hit_on_still_sounding_last_syllable_fails(mini_project):
    # the next segment's hit (hard cut, 1.875 s) is 0.6 s after the last syllable peak,
    # but the source there is only 2 dB under it
    p, _, _ = _mix(mini_project, 0, d=1.9, amp=_sustained)
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "score event on last syllable" in r.stdout and "s03@" in r.stdout


def test_segment_buried_under_score_fails(mixed):
    # noise 10 dB over the speech for the whole segment: every syllable rises together,
    # so only the segment gain shows it
    p, start, sr = mixed
    _add(p, start, start + 1.9, _noise(3.2 * _speech_amp(p)))
    r = _verify(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "segment buried under score (gain +" in r.stdout
