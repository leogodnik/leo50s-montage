import json
from pathlib import Path

import pytest

from montagelib.project import ProjectError, load_project

FIXTURE = Path(__file__).parent / "fixtures" / "mini-trailer.json"


def _write(tmp_path, mutate):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate(data)
    p = tmp_path / "trailer.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_loads_mini():
    pr = load_project(FIXTURE)
    assert len(pr.edl) == 6
    assert pr.clips["a"].is_absolute()
    assert pr.clips["a"] == FIXTURE.parent / "clips" / "a.mp4"
    assert pr.fps == 24
    assert pr.build_dir == FIXTURE.parent
    assert pr.seg_dir == FIXTURE.parent / "seg"
    assert pr.by_id("s02").speech is True


def test_hit_normalised():
    pr = load_project(FIXTURE)
    assert pr.by_id("s03").hit == {
        "type": "boom", "gain": None, "offset": 0, "whistle_from": None}
    assert pr.by_id("s02").hit is None


def test_hit_object_form(tmp_path):
    def m(d):
        d["edl"][3]["hit"] = {"type": "riser", "gain": 0.4, "offset": -0.5, "whistle_from": "s02"}
    pr = load_project(_write(tmp_path, m))
    assert pr.by_id("s03").hit == {
        "type": "riser", "gain": 0.4, "offset": -0.5, "whistle_from": "s02"}


def test_collects_all_problems(tmp_path):
    def m(d):
        d["edl"][2]["src"] = "zzz"
        d["edl"][3]["hit"] = "kaboom"
        d["score"]["layers"][0]["from"] = "nope"
        d["fonts"] = {"card": "/no/such/font.ttf"}
    with pytest.raises(ProjectError) as ei:
        load_project(_write(tmp_path, m))
    problems = ei.value.problems
    assert len(problems) == 4
    msg = str(ei.value)
    for needle in ("zzz", "kaboom", "nope", "/no/such/font.ttf"):
        assert needle in msg


def test_layer_to_end_is_valid(tmp_path):
    def m(d):
        d["score"]["layers"][0]["to"] = "end"
    assert load_project(_write(tmp_path, m)).score["layers"][0]["to"] == "end"


def test_from_end_is_invalid(tmp_path):
    def m(d):
        d["score"]["layers"][0]["from"] = "end"
    with pytest.raises(ProjectError):
        load_project(_write(tmp_path, m))


def test_grade_file(tmp_path):
    (tmp_path / "g.txt").write_text("a 0.9 1.1\nb 1.2\n", encoding="utf-8")
    pr = load_project(_write(tmp_path, lambda d: d.update(grade="g.txt")))
    assert pr.grade == {"a": (0.9, 1.1), "b": (1.2, 1.0)}


def test_defaults():
    pr = load_project(FIXTURE)
    assert pr.letterbox is None
    assert pr.grade == {}
    assert pr.output == {"name": "trailer", "crf": 16, "preview_height": 720}
    assert pr.fonts["card"] == "/System/Library/Fonts/HelveticaNeue.ttc#1"
    assert (pr.width, pr.height) == (1920, 1080)
    assert pr.by_id("s03").fade_in == 0


def test_malformed_types_become_problems(tmp_path):
    def m(d):
        d["edl"][3]["hit"] = {"type": "boom", "gain": "loud-gain"}
        d["edl"][2]["ov"] = "slow-ov"
        d["clips"]["b"] = 42
        d["fonts"] = ["font-list"]
        d["output"] = "output-str"
        d["score"]["layers"] = "layers-str"
    with pytest.raises(ProjectError) as ei:
        load_project(_write(tmp_path, m))
    problems = ei.value.problems
    assert len(problems) == 6
    msg = str(ei.value)
    for needle in ("loud-gain", "slow-ov", "42", "font-list", "output-str", "layers-str"):
        assert needle in msg


def test_ov_bounds(tmp_path):
    def m(d):
        d["edl"][1]["ov"] = -0.1
        d["edl"][2]["ov"] = 1.5  # d == 1.5
    with pytest.raises(ProjectError) as ei:
        load_project(_write(tmp_path, m))
    assert len(ei.value.problems) == 2
    assert all("ov" in p for p in ei.value.problems)
