import numpy as np
from eval import merge_metric as mm


def _rect(h=10, w=10, y0=3, y1=7, x0=3, x1=7):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks_perfect_consistency():
    m = _rect()
    masks = {5: (m, 0, 0), 6: (m.copy(), 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["z_from"] == 5 and t["z_to"] == 6 and t["gap"] == 1
    assert t["iou"] == 1.0
    assert t["centroid_drift_px"] == 0.0


def test_shifted_mask_hand_computed_iou_and_drift():
    m_a = _rect()  # x[3:7), y[3:7)
    m_b = _rect(x0=5, x1=9)  # x[5:9), y[3:7), shifted +2 in x
    masks = {5: (m_a, 0, 0), 6: (m_b, 0, 0)}
    t = mm.z_transitions(masks)[0]
    # intersection x[5:7),y[3:7) = 2*4=8; each mask area 4*4=16; union = 16+16-8 = 24
    assert abs(t["iou"] - 8 / 24) < 1e-9
    assert abs(t["centroid_drift_px"] - 2.0) < 1e-9


def test_disjoint_masks_zero_iou():
    m_a = np.zeros((10, 10), dtype=bool); m_a[0:2, 0:2] = True
    m_b = np.zeros((10, 10), dtype=bool); m_b[8:10, 8:10] = True
    masks = {5: (m_a, 0, 0), 6: (m_b, 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["iou"] == 0.0


def test_empty_mask_is_dropout_not_low_consistency():
    m_a = _rect()
    m_empty = np.zeros((10, 10), dtype=bool)
    masks = {5: (m_a, 0, 0), 6: (m_empty, 0, 0)}
    t = mm.z_transitions(masks)[0]
    assert t["iou"] is None
    assert t["centroid_drift_px"] is None


def test_z_gap_recorded_correctly():
    m = _rect()
    masks = {10: (m, 0, 0), 11: (m.copy(), 0, 0), 13: (m.copy(), 0, 0)}
    ts = mm.z_transitions(masks)
    assert [t["gap"] for t in ts] == [1, 2]


def test_summarize_empty_input():
    s = mm.summarize_z_consistency([])
    assert s["n_transitions"] == 0
    assert s["mean_z2z_iou"] is None
    assert s["mean_centroid_drift_px"] is None
    assert s["frac_gap1_transitions"] is None
    assert s["frac_low_iou"] is None


def test_summarize_aggregates_and_excludes_dropout_and_gaps_correctly():
    transitions = [
        {"gap": 1, "iou": 0.9, "centroid_drift_px": 1.0},
        {"gap": 1, "iou": 0.3, "centroid_drift_px": 5.0},
        {"gap": 1, "iou": None, "centroid_drift_px": None},  # dropout
        {"gap": 2, "iou": 0.4, "centroid_drift_px": 3.0},  # gap != 1
    ]
    s = mm.summarize_z_consistency(transitions)
    assert s["n_transitions"] == 4
    assert s["n_dropout_transitions"] == 1
    assert abs(s["mean_z2z_iou"] - (0.9 + 0.3 + 0.4) / 3) < 1e-9
    assert abs(s["frac_gap1_transitions"] - 3 / 4) < 1e-9
    # frac_low_iou only counts gap==1, scored transitions: [0.9, 0.3], 1 of 2 below 0.5
    assert abs(s["frac_low_iou"] - 0.5) < 1e-9
