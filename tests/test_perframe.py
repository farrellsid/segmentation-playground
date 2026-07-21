import numpy as np
import pandas as pd
from sam2_utils import perframe as pf


def test_nodes_in_frame_filters_z_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"], "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401], "x_tif": [800.0, 1600.0, 240.0], "y_tif": [80.0, 160.0, 800.0],
    })
    got = pf.nodes_in_frame(df, 1400, scale=8)
    assert sorted(got) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert pf.nodes_in_frame(df, 1401, scale=8) == [(30.0, 100.0, "AVAL", "c")]


def test_argmax_resolves_overlap_to_nearest_node():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = np.zeros((20, 20), bool); b[8:18, 8:18] = True   # overlaps a in [8:12, 8:12]
    lab = pf.resolve_overlaps_argmax([a, b], [(6, 6), (13, 13)])
    # contested pixel (9,9): nearer node b(13,13)? dist to (6,6)=~4.2, to (13,13)=~5.6 -> a
    assert lab[9, 9] == 1
    # (11,11): to (6,6)=~7.1, to (13,13)=~2.8 -> b
    assert lab[11, 11] == 2
    # uncontested
    assert lab[3, 3] == 1 and lab[16, 16] == 2 and lab[0, 0] == 0

    # Off-diagonal case: catches an x/y transposition in the distance computation.
    # node_xy is (x, y). Seeds are mirror images of each other across the diagonal:
    # seedC=(x=1, y=19), seedD=(x=19, y=1). Contested pixel is (row=5, col=15), i.e.
    # (x=15, y=5). Correct distance: d(C)=(1-15)^2+(19-5)^2=392, d(D)=(19-15)^2+(1-5)^2=32
    # -> nearest is D (label 2). If x and y were swapped anywhere in the distance formula
    # (or in unpacking the contested-pixel coordinates), the squared terms pair up with
    # the wrong seed axis: d(C)'=(1-5)^2+(19-15)^2=32, d(D)'=(19-5)^2+(1-15)^2=392 -> nearest
    # would flip to C (label 1). So this pixel is guaranteed to fail under an x/y swap.
    c = np.zeros((20, 20), bool); c[:, 0:16] = True
    d = np.zeros((20, 20), bool); d[0:11, 10:20] = True   # overlaps c in rows0:11, cols10:16
    lab2 = pf.resolve_overlaps_argmax([c, d], [(1, 19), (19, 1)])
    assert lab2[5, 15] == 2
    # uncontested, off-diagonal sanity checks
    assert lab2[19, 1] == 1   # only c claims (row != col)
    assert lab2[1, 19] == 2   # only d claims (row != col)


def test_watershed_labels_are_disjoint_and_seeded():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = np.zeros((20, 20), bool); b[8:18, 8:18] = True
    mem = np.zeros((20, 20), np.float32)
    lab = pf.resolve_overlaps_watershed([a, b], [(6, 6), (13, 13)], mem)
    assert lab[6, 6] == 1 and lab[13, 13] == 2          # seeds keep their label
    assert set(np.unique(lab)) <= {0, 1, 2}
    # Real partition check: every pixel in the union of the two masks must get exactly
    # one of the two labels (the old `(lab == 1) & (lab == 2)` check was tautological,
    # since a single int array can never equal both 1 and 2 at the same cell).
    union = a | b
    n1, n2 = int((lab == 1).sum()), int((lab == 2).sum())
    assert n1 + n2 == int(union.sum())


def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_select_by_metric_prefers_node_containing_membrane_aligned():
    node = (20, 20)
    small = _disk(20, 20, 4)          # contains node, tight
    big = _disk(20, 20, 15)           # contains node, but engulfs a foreign node
    off = _disk(35, 35, 4)            # does not contain node
    mem = np.zeros((40, 40), np.float32)
    # membrane ridge on the small disk's rim -> high boundary_on_membrane for `small`
    from sam2_utils.perframe import _rim
    mem[_rim(small)] = 1.0
    idx = pf.select_by_metric([off, big, small], node, foreign_xy=[(20, 30)], membrane_map=mem)
    assert idx == 2                   # `small`: contains node, no foreign, best boundary


def test_match_amg_assigns_nodes_and_keeps_leftover():
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    m_a = _disk(10, 10, 5); m_b = _disk(30, 30, 5); junk = _disk(20, 5, 3)
    mem = np.zeros((40, 40), np.float32)
    labels, leftover = pf.match_amg_to_nodes([junk, m_a, m_b], node_index, mem)
    assert set(labels) == {"AVAL", "AVAR"}
    assert int(labels["AVAL"].sum()) == int(m_a.sum())
    assert len(leftover) == 1 and int(leftover[0].sum()) == int(junk.sum())
