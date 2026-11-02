import numpy as np
from eval import merge_metric as mm


def _rect(h=10, w=10, y0=3, y1=7, x0=3, x1=7):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks_perfect_agreement():
    m = _rect()
    fwd = {5: (m, 0, 0)}
    back = {5: (m.copy(), 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["z"] == 5
    assert r["iou"] == 1.0
    assert r["centroid_drift_px"] == 0.0


def test_shifted_mask_hand_computed_iou_and_drift():
    m_a = _rect()                      # x[3:7), y[3:7)
    m_b = _rect(x0=5, x1=9)             # x[5:9), y[3:7), shifted +2 in x
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_b, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    # intersection x[5:7),y[3:7) = 2*4=8; each area 16; union = 16+16-8=24
    assert abs(r["iou"] - 8 / 24) < 1e-9
    assert abs(r["centroid_drift_px"] - 2.0) < 1e-9


def test_offset_and_shape_difference_same_region_gives_perfect_agreement():
    mask_a = np.ones((4, 4), dtype=bool)               # 4x4 window at (10, 10)
    mask_b = np.zeros((6, 6), dtype=bool)               # 6x6 window at (8, 8)
    mask_b[2:6, 2:6] = True                             # same absolute region
    fwd = {5: (mask_a, 10, 10)}
    back = {5: (mask_b, 8, 8)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] == 1.0
    assert r["centroid_drift_px"] == 0.0


def test_disjoint_masks_zero_iou():
    m_a = np.zeros((10, 10), dtype=bool); m_a[0:2, 0:2] = True
    m_b = np.zeros((10, 10), dtype=bool); m_b[8:10, 8:10] = True
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_b, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] == 0.0


def test_one_side_empty_is_dropout_not_low_agreement():
    m_a = _rect()
    m_empty = np.zeros((10, 10), dtype=bool)
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_empty, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] is None
    assert r["centroid_drift_px"] is None


def test_only_shared_z_is_scored():
    m = _rect()
    fwd = {5: (m, 0, 0), 6: (m.copy(), 0, 0)}      # 6 is fwd-only (near the back seed end)
    back = {5: (m.copy(), 0, 0), 4: (m.copy(), 0, 0)}  # 4 is back-only (near the fwd seed end)
    r = mm.directional_disagreement(fwd, back)
    assert [rec["z"] for rec in r] == [5]


def test_summarize_empty_input():
    s = mm.summarize_directional_disagreement([])
    assert s["n_z"] == 0
    assert s["mean_disagreement_iou"] is None
    assert s["mean_centroid_drift_px"] is None
    assert s["frac_low_agreement"] is None


def test_summarize_aggregates_and_excludes_dropout():
    records = [
        {"z": 1, "iou": 0.9, "centroid_drift_px": 1.0},
        {"z": 2, "iou": 0.3, "centroid_drift_px": 5.0},
        {"z": 3, "iou": None, "centroid_drift_px": None},   # one-sided dropout
    ]
    s = mm.summarize_directional_disagreement(records)
    assert s["n_z"] == 3
    assert s["n_dropout_z"] == 1
    assert abs(s["mean_disagreement_iou"] - (0.9 + 0.3) / 2) < 1e-9
    # frac_low_agreement (threshold 0.5) over scored z's only: 1 of 2 below 0.5
    assert abs(s["frac_low_agreement"] - 0.5) < 1e-9
