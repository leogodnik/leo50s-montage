import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from montagelib import ffm

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "skills" / "speech-safe-cuts" / "scripts" / "check_cuts.py"
MINI_WORDS = Path(__file__).parent / "fixtures" / "words-mini.json"


@pytest.fixture
def speech_project(mini_project):
    p = mini_project
    clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", 2.0, tone_hz=440,
                              bursts=[(0.5, 0.8), (1.2, 1.5)])
    ffm.extract_wav(clip, p.wav_dir / "a.wav")
    shutil.copy(MINI_WORDS, p.build_dir / "words.json")
    return p


def _edit(p, **seg_fields):
    data = json.loads(p.path.read_text(encoding="utf-8"))
    for e in data["edl"]:
        e.update(seg_fields.get(e["id"], {}))
    p.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _check(p):
    return subprocess.run([sys.executable, str(CHECK), str(p.path)],
                          capture_output=True, text=True)


def test_clean_project_passes(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.9})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "один" in r.stdout and "два" in r.stdout
    assert "problems: 0" in r.stdout


def test_cut_word_fails(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0.6, "d": 1.3})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "CUT WORD" in r.stdout


def test_tail_word_with_hit_on_last_syllable_fails(speech_project):
    # "два" runs into the end; the next segment's hit (mid-crossfade, 1.29 s in clip time)
    # lands on its last syllable (plateau middle ~1.34 s)
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "fade", "ov": 0.5})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "hit on last syllable: next s03" in r.stdout
    # the waiver covers only the END: a start inside a word is still a problem
    _edit(p, s02={"ss": 0.6, "d": 0.95}, s03={"trans": "cut", "hit": None})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "tail: hard cut ok" in r.stdout
    assert "start not quiet" in r.stdout


def test_tail_word_with_fade_and_no_hit_passes(speech_project):
    # v6 s07 -> s08: the last word runs into a dissolve without a hit. The renderer does
    # not fade the outgoing speech on a visual transition, so nothing masks the word.
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "fade", "ov": 0.5, "hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: next s03 fade, no hit" in r.stdout


def _decaying(p):
    # "один" 0.5-0.8; "два" peaks at 1.25 and decays (still -21 dB at 1.49), like "паз!"
    import numpy as np
    from scipy.io import wavfile
    sr = 48000
    t = np.arange(int(2.0 * sr)) / sr
    amp = np.where((t >= 0.5) & (t < 0.8), 0.5, 0.0)
    amp += np.where(t >= 1.2, 0.5 * np.exp(-np.clip(t - 1.25, 0, None) / 0.1), 0.0)
    amp[(t >= 1.2) & (t < 1.25)] = 0.5
    x = amp * np.sin(2 * np.pi * 440 * t)
    wavfile.write(p.wav_dir / "a.wav", sr, (x * 32767).astype(np.int16))


def test_tail_word_with_hit_after_last_syllable_passes(speech_project):
    # v6 s08 -> s09: a hard cut whose hit lands 0.2 s after the last syllable peak
    p = speech_project
    _decaying(p)
    _edit(p, s02={"ss": 0, "d": 1.55})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: next s03 hit" in r.stdout and "after last syllable" in r.stdout


def test_sound_running_on_to_the_end_is_a_tail(speech_project):
    # v6 s11: whisper ends "два" at 1.30, but the sound never goes quiet before the end
    # (no hit after it: a hit there would land on sound still at full level)
    p = speech_project
    _clip(p, 2.5, [(0.5, 0.8), (1.2, 2.0)])
    _words(p, [("один", 0.5, 0.8), ("два", 1.2, 1.3)])
    _edit(p, s02={"ss": 0, "d": 1.9}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "end not quiet" not in r.stdout
    assert "tail: sound runs on to the end" in r.stdout


def test_end_is_checked_on_the_frame_grid(speech_project):
    # d 1.48 renders as 36 frames = 1.50 s; whisper's "два" ends at 1.52, so the raw end
    # 1.48 would look like a cut word while the rendered end is after it
    p = speech_project
    _words(p, [("один", 0.5, 0.8), ("два", 1.2, 1.52)])
    _edit(p, s02={"ss": 0, "d": 1.48}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0.00- 1.50" in r.stdout
    assert "CUT WORD" not in r.stdout


def test_missing_wav_is_loud_failure(speech_project):
    p = speech_project
    (p.wav_dir / "a.wav").unlink()
    r = _check(p)
    assert r.returncode == 2
    assert "wav/a.wav" in r.stdout + r.stderr


def test_tail_word_with_hard_cut_passes(speech_project):
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "cut", "hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: hard cut ok" in r.stdout
    assert "problems: 0" in r.stdout


def _words(p, ws):
    (p.build_dir / "words.json").write_text(
        json.dumps({"a": [{"w": w, "s": s, "e": e} for w, s, e in ws]}, ensure_ascii=False),
        encoding="utf-8")


def _clip(p, seconds, bursts):
    clip = ffm.make_test_clip(p.build_dir / "clips" / "a.mp4", seconds, tone_hz=440,
                              bursts=bursts)
    ffm.extract_wav(clip, p.wav_dir / "a.wav")


def _continuous(p):
    # one shout from 0.0 to 1.5 s, a 14 dB dip (no silence) at 1.18-1.20, then silence
    import numpy as np
    from scipy.io import wavfile
    sr = 48000
    t = np.arange(int(3.0 * sr)) / sr
    amp = np.where(t < 1.5, 0.5, 0.0)
    amp[(t >= 1.18) & (t < 1.20)] = 0.1
    wavfile.write(p.wav_dir / "a.wav", sr, (amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))


def test_start_after_previous_word_passes(speech_project):
    # "два" follows "один" with no pause: the start cannot be quiet, but it sits
    # in the envelope dip before the word, so only a note is printed
    p = speech_project
    _continuous(p)
    _words(p, [("один", 0.0, 1.2), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 1.18, "d": 0.8})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "start: not quiet (previous sound tail, dip 1.18)" in r.stdout
    assert "problems: 0" in r.stdout


def test_start_mid_word_in_continuous_speech_fails(speech_project):
    # whisper lags and puts "два" at 1.30; no pause before it, so onset_before
    # falls back to whisper; the dip at 1.18 shows the start is mid-word
    p = speech_project
    _continuous(p)
    _words(p, [("один", 0.0, 1.30), ("два", 1.30, 1.5)])
    _edit(p, s02={"ss": 1.30, "d": 0.7})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "start cuts into a word (dip 1.18)" in r.stdout


def test_start_cutting_into_word_fails(speech_project):
    # whisper puts "один" at 0.7, the sound starts at 0.5; a start at 0.6 cuts it
    p = speech_project
    _words(p, [("один", 0.7, 0.8), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 0.6, "d": 1.3})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "start cuts into a word" in r.stdout


def _shape(p, amp_of_t, seconds=2.5):
    """Write wav/a.wav: a 440 Hz tone under the amplitude envelope amp_of_t(t)."""
    import numpy as np
    from scipy.io import wavfile
    sr = 48000
    t = np.arange(int(seconds * sr)) / sr
    x = amp_of_t(t) * np.sin(2 * np.pi * 440 * t)
    wavfile.write(p.wav_dir / "a.wav", sr, (x * 32767).astype(np.int16))


def _sustained(t):
    # "один" 0.5-0.8; "два" peaks at 1.2-1.3, dips 8 dB at 1.35-1.40, then "-те" keeps
    # sounding 2 dB under the peak until 2.0 (v6 "Тяните" at 1.30-2.13)
    import numpy as np
    amp = np.where((t >= 0.5) & (t < 0.8), 0.5, 0.0)
    amp = np.where((t >= 1.2) & (t < 1.3), 0.5, amp)
    amp = np.where((t >= 1.3) & (t < 1.4), 0.2, amp)
    return np.where((t >= 1.4) & (t < 2.0), 0.4, amp)


def test_hit_on_a_still_sounding_last_syllable_fails(speech_project):
    # the hard cut's hit (1.875) is 0.6 s after the last syllable peak, but the source
    # there is only 2 dB under that peak: the hit lands on a sounding syllable
    p = speech_project
    _shape(p, _sustained)
    _words(p, [("один", 0.5, 0.8), ("два", 1.2, 1.9)])
    _edit(p, s02={"ss": 0, "d": 1.9})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "hit on last syllable: next s03" in r.stdout


def test_hit_after_a_decayed_last_syllable_passes(speech_project):
    # "два" decays from 1.25 (tau 0.1 s); the hit 0.29 s later finds it 25 dB down
    p = speech_project
    _decaying(p)
    _edit(p, s02={"ss": 0, "d": 1.5833})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: next s03 hit" in r.stdout


def test_fadewhite_after_tail_word_fails(speech_project):
    # the user's correction: no white flash after a word that runs into the cut
    p = speech_project
    _edit(p, s02={"ss": 0, "d": 1.55}, s03={"trans": "fadewhite", "ov": 0.5, "hit": None})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "needs cut or fade (tail word): next s03 fadewhite" in r.stdout


def _run_on(dip_at):
    # one unbroken sound 0.5-1.9 with a 14 dB dip (20 ms) where "два" really starts
    def amp(t):
        import numpy as np
        a = np.where((t >= 0.5) & (t < 1.9), 0.5, 0.0)
        a[(t >= dip_at) & (t < dip_at + 0.02)] = 0.1
        return a
    return amp


def test_end_cutting_into_next_word_fails(speech_project):
    # whisper places "два" at 1.70, 0.2 s after the end; the envelope shows it
    # starting at the 1.38 dip, 0.1 s before the end: the run-on waiver must not apply
    p = speech_project
    _shape(p, _run_on(1.38))
    _words(p, [("один", 0.5, 1.2), ("два", 1.7, 1.9)])
    _edit(p, s02={"ss": 0, "d": 1.5}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "end cuts into next word (onset at 1.38)" in r.stdout


def test_end_before_next_word_onset_passes(speech_project):
    # the same, but "два" really starts at 1.58, after the end
    p = speech_project
    _shape(p, _run_on(1.58))
    _words(p, [("один", 0.5, 1.2), ("два", 1.7, 1.9)])
    _edit(p, s02={"ss": 0, "d": 1.5}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tail: sound runs on to the end" in r.stdout


def test_whisper_end_late_in_silence_is_a_note(speech_project):
    # the burst of "один" ends at 0.8, whisper stretches it to 1.1; the end 0.95 (rendered
    # 0.958) is in silence: whisper's late end is not a cut word
    p = speech_project
    _words(p, [("один", 0.5, 1.1), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 0, "d": 0.95}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CUT WORD" not in r.stdout
    assert "end: whisper end late (один 0.5-1.1)" in r.stdout
    assert "problems: 0" in r.stdout


def test_whisper_end_late_inside_the_burst_is_a_cut_word(speech_project):
    # the same words, but the end 0.7 is inside the burst: the word is really cut
    p = speech_project
    _words(p, [("один", 0.5, 1.1), ("два", 1.2, 1.5)])
    _edit(p, s02={"ss": 0, "d": 0.7}, s03={"hit": None})
    r = _check(p)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "CUT WORD" in r.stdout
    assert "whisper end late" not in r.stdout
