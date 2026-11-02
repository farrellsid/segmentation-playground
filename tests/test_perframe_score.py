import numpy as np
from eval.perframe_score import score_frame


def _disk(cx, cy, r, shape=(60, 60)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_score_frame_own_foreign_and_overlap():
    node_index = [(15, 15, "AVAL", "a"), (40, 40, "AVAR", "b")]
    masks = {"AVAL": _disk(15, 15, 8), "AVAR": _disk(40, 40, 8)}   # disjoint, each own node
    s = score_frame(masks, node_index, membrane_map=None)
    assert s["own_coverage"] == 1.0
    assert s["total_foreign"] == 0
    assert s["overlap_fraction"] == 0.0
    # now make AVAL swallow AVAR's node -> a foreign hit and overlap
    masks2 = {"AVAL": _disk(27, 27, 22), "AVAR": _disk(40, 40, 8)}
    s2 = score_frame(masks2, node_index, membrane_map=None)
    assert s2["total_foreign"] >= 1
    assert s2["overlap_fraction"] > 0.0
