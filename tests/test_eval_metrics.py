"""Unit tests for eval.metrics, region overlap + split/merge VOI.

Pure numpy on synthetic arrays with hand-checkable answers, like
test_alignment / test_labels. Run either way:
    py -3 -m pytest tests/test_eval_metrics.py
    py -3 tests/test_eval_metrics.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from eval import metrics as M


def test_binary_perfect_overlap():
    a = np.zeros((4, 4), bool)
    a[1:3, 1:3] = True
    m = M.binary_metrics(a, a)
    assert m["iou"] == 1.0 and m["dice"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["tp"] == 4 and m["fp"] == 0 and m["fn"] == 0


def test_binary_half_overlap():
    pred = np.zeros((1, 4), bool); pred[0, 0:2] = True   # {0,1}
    gt = np.zeros((1, 4), bool);   gt[0, 1:3] = True      # {1,2}
    m = M.binary_metrics(pred, gt)
    # intersection 1, union 3
    assert math.isclose(m["iou"], 1 / 3)
    assert math.isclose(m["dice"], 2 / 4)
    assert math.isclose(m["precision"], 1 / 2)
    assert math.isclose(m["recall"], 1 / 2)


def test_binary_empty_empty_is_perfect():
    z = np.zeros((3, 3), bool)
    m = M.binary_metrics(z, z)
    assert m["iou"] == 1.0 and m["dice"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_binary_gt_empty_recall_nan():
    pred = np.ones((2, 2), bool)
    gt = np.zeros((2, 2), bool)
    m = M.binary_metrics(pred, gt)
    assert m["iou"] == 0.0 and m["dice"] == 0.0
    assert m["precision"] == 0.0          # tp 0 / (tp+fp 4)
    assert math.isnan(m["recall"])        # no GT positives


def test_metrics_from_counts_matches_pixels():
    pred = np.zeros((1, 8), bool); pred[0, 0:5] = True
    gt = np.zeros((1, 8), bool);   gt[0, 3:8] = True
    per = M.binary_metrics(pred, gt)
    agg = M.metrics_from_counts(per["tp"], per["fp"], per["fn"])
    assert math.isclose(agg["iou"], per["iou"])
    assert math.isclose(agg["dice"], per["dice"])


def test_voi_identical_is_zero():
    lab = np.array([[1, 1, 2], [2, 3, 3]])
    v = M.variation_of_information(lab, lab)
    assert math.isclose(v["voi"], 0.0, abs_tol=1e-12)
    assert v["voi_split"] == 0.0 and v["voi_merge"] == 0.0


def test_voi_pure_split():
    # GT: one object (all label 1). SEG cuts it into two equal halves.
    gt = np.ones(4, int)
    seg = np.array([1, 1, 2, 2])
    v = M.variation_of_information(seg, gt, base=2.0)
    # H(seg|gt)=1 bit (gt gives no info about the 50/50 seg split); H(gt|seg)=0.
    assert math.isclose(v["voi_split"], 1.0, abs_tol=1e-9)
    assert math.isclose(v["voi_merge"], 0.0, abs_tol=1e-9)


def test_voi_pure_merge():
    # SEG: one object. GT has two equal halves -> a merge.
    seg = np.ones(4, int)
    gt = np.array([1, 1, 2, 2])
    v = M.variation_of_information(seg, gt, base=2.0)
    assert math.isclose(v["voi_split"], 0.0, abs_tol=1e-9)
    assert math.isclose(v["voi_merge"], 1.0, abs_tol=1e-9)


def test_voi_ignore_gt_background():
    # Half the field is GT background (0); ignoring it leaves a perfect match.
    gt = np.array([0, 0, 1, 1])
    seg = np.array([7, 9, 5, 5])     # background pixels disagree, fg agrees
    v = M.variation_of_information(seg, gt, ignore_gt=(0,))
    assert math.isclose(v["voi"], 0.0, abs_tol=1e-9)
    assert v["n"] == 2


def test_weighted_voi_ratio():
    v = {"voi_split": 1.0, "voi_merge": 2.0}
    assert math.isclose(M.weighted_voi(v, merge_split_ratio=5.0), 11.0)


# --- adapted Rand (SNEMI3D) ---------------------------------------------------

def test_arand_identical_is_perfect():
    lab = np.array([[1, 1, 2], [2, 3, 3]])
    a = M.adapted_rand(lab, lab)
    assert math.isclose(a["are"], 0.0, abs_tol=1e-12)
    assert a["precision"] == 1.0 and a["recall"] == 1.0


def test_arand_pure_split():
    # GT one 4-px object; SEG cuts it into two pairs. Pair counting: GT has 4·3=12
    # within-pairs, SEG keeps 2·(1)+2·(1)=4 of them -> recall 4/12. No merges -> prec 1.
    gt = np.ones(4, int)
    seg = np.array([1, 1, 2, 2])
    a = M.adapted_rand(seg, gt)
    assert math.isclose(a["precision"], 1.0, abs_tol=1e-12)   # no merges
    assert math.isclose(a["recall"], 1 / 3, abs_tol=1e-12)    # gt split in two
    assert math.isclose(a["split_error"], 2 / 3, abs_tol=1e-12)
    assert a["merge_error"] == 0.0


def test_arand_pure_merge():
    # SEG one object; GT is two halves. Symmetric to the split case (pair counting).
    seg = np.ones(4, int)
    gt = np.array([1, 1, 2, 2])
    a = M.adapted_rand(seg, gt)
    assert math.isclose(a["precision"], 1 / 3, abs_tol=1e-12)
    assert math.isclose(a["recall"], 1.0, abs_tol=1e-12)
    assert math.isclose(a["merge_error"], 2 / 3, abs_tol=1e-12)
    assert a["split_error"] == 0.0


def test_arand_ignore_gt_background():
    gt = np.array([0, 0, 1, 1])
    seg = np.array([7, 9, 5, 5])           # bg disagrees, fg perfect
    a = M.adapted_rand(seg, gt, ignore_gt=(0,))
    assert math.isclose(a["are"], 0.0, abs_tol=1e-12)
    assert a["n"] == 2


def test_arand_matches_skimage():
    # Cross-check against skimage's reference implementation. The ARE (the headline
    # error) must match EXACTLY, it's the symmetric, unambiguous number. We label
    # precision by the prediction's groupings (precision = correct / pred pairs, the
    # standard TP/(TP+FP) sense); skimage labels precision against the *truth*
    # marginals, so our precision/recall are skimage's recall/precision (the swap is
    # cosmetic, ARE is identical and our merge/split tests above pin the semantics).
    import pytest
    skm = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    gt = rng.integers(1, 5, size=400)
    seg = rng.integers(1, 6, size=400)
    a = M.adapted_rand(seg, gt)
    are, sk_prec, sk_rec = skm.adapted_rand_error(gt, seg, ignore_labels=())
    assert math.isclose(a["are"], are, abs_tol=1e-9)
    assert math.isclose(a["precision"], sk_rec, abs_tol=1e-9)
    assert math.isclose(a["recall"], sk_prec, abs_tol=1e-9)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
