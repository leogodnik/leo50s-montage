from montagelib.project import Segment
from montagelib.timeline import snap, frames, timeline, visual_start

FPS = 24


def seg(id, d, ov=0.0, trans="fade", kind="black"):
    return Segment(id=id, kind=kind, d=d, ov=ov, trans=trans)


def test_snap_rounds_to_frame():
    assert snap(0.84, 24) == 20 / 24
    assert frames(0.84, 24) == 20


def test_cut_is_one_frame_overlap():
    edl = [seg("a", 1.0), seg("b", 1.0, ov=0.9, trans="cut")]
    timeline(edl, FPS)
    assert edl[1].ov == 1 / FPS


def test_first_segment_no_overlap():
    edl = [seg("a", 1.0, ov=0.5), seg("b", 1.0)]
    res, _ = timeline(edl, FPS)
    assert edl[0].ov == 0.0
    assert res[0][1] == 0.0


def test_total_matches_zemlyanka_grid():
    edl = [seg("a", 1.0), seg("b", 4.0, ov=0.5), seg("c", 5.0, ov=0.6)]
    res, total = timeline(edl, FPS)
    assert total == snap(1 + 3.5 + 4.4, FPS)
    assert [s.id for s, _ in res] == ["a", "b", "c"]


def test_visual_start_cut_vs_fade():
    edl = [seg("a", 1.0), seg("b", 1.0, ov=0.5), seg("c", 1.0, trans="cut")]
    res, _ = timeline(edl, FPS)
    (_, sb), (c, sc) = res[1], res[2]
    assert visual_start(edl[1], sb) == sb + edl[1].ov / 2
    assert visual_start(c, sc) == sc
