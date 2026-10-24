from pathlib import Path

from montagelib.words import Word, inside, load_words, straddling

MINI = Path(__file__).parent / "fixtures" / "words-mini.json"


def test_straddling_detects_cut_word():
    ws = load_words(MINI)["a"]
    assert ws == [Word("один", 0.5, 0.8), Word("два", 1.2, 1.5)]
    assert [w.w for w in straddling(ws, 0.6, 1.9)] == ["один"]
    assert [w.w for w in straddling(ws, 0.0, 1.35)] == ["два"]
    assert straddling(ws, 0.0, 1.9) == []
    assert [w.w for w in inside(ws, 0.6, 1.9)] == ["два"]


def test_tolerance_30ms():
    ws = load_words(MINI)["a"]
    # boundary 20 ms inside the word: within tolerance, the word counts as inside
    assert straddling(ws, 0.52, 1.48) == []
    assert [w.w for w in inside(ws, 0.52, 1.48)] == ["один", "два"]
    # 40 ms inside: the word is cut
    assert [w.w for w in straddling(ws, 0.54, 1.46)] == ["один", "два"]
    assert inside(ws, 0.54, 1.46) == []
