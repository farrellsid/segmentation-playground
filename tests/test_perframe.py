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


def test_watershed_labels_are_disjoint_and_seeded():
    a = np.zeros((20, 20), bool); a[2:12, 2:12] = True
    b = np.zeros((20, 20), bool); b[8:18, 8:18] = True
    mem = np.zeros((20, 20), np.float32)
    lab = pf.resolve_overlaps_watershed([a, b], [(6, 6), (13, 13)], mem)
    assert lab[6, 6] == 1 and lab[13, 13] == 2          # seeds keep their label
    assert set(np.unique(lab)) <= {0, 1, 2}
    assert not ((lab == 1) & (lab == 2)).any()          # disjoint by construction
