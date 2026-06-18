"""Unit tests for the multimask anchor auto-select (pipeline._select_anchor_mask)
and its shared geometry helpers (_point_in_mask / _largest_cc_frac).

Torch-free and data-free, like test_alignment: pipeline.py imports torch only
lazily (inside the predictor-touching functions), so importing the module and
exercising the pure selection/geometry logic needs no GPU and no EM stack. This
guards the anchor ranking: node-containment ->
plausible-area -> single-CC -> SAM IoU, graceful (always returns a candidate).

Run either way:
    py -3 -m pytest tests/test_anchor_select.py
    py -3 tests/test_anchor_select.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import pipeline
from pipeline import Prompts


def _mask(hw, boxes):
    """Build a bool mask of shape hw with the given [y0,y1,x0,x1] filled rectangles."""
    m = np.zeros(hw, dtype=bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------------------
# _point_in_mask
# ---------------------------------------------------------------------------

def test_point_in_mask_hit_miss_and_radius():
    m = _mask((20, 20), [(10, 15, 10, 15)])      # a 5x5 blob
    assert pipeline._point_in_mask(m, 12, 12, 0) is True        # point inside
    assert pipeline._point_in_mask(m, 5, 5, 0) is False         # far away, r=0
    assert pipeline._point_in_mask(m, 9, 12, 1) is True         # 1px outside, r=1 reaches it
    assert pipeline._point_in_mask(m, 5, 5, 0) is False


def test_point_in_mask_out_of_frame_is_false():
    m = _mask((20, 20), [(0, 20, 0, 20)])        # entirely foreground
    assert pipeline._point_in_mask(m, -3, 5, 1) is False        # x out of frame
    assert pipeline._point_in_mask(m, 5, 99, 1) is False        # y out of frame


# ---------------------------------------------------------------------------
# _largest_cc_frac
# ---------------------------------------------------------------------------

def test_largest_cc_frac_empty():
    assert pipeline._largest_cc_frac(np.zeros((10, 10), bool)) == (0, 0.0)


def test_largest_cc_frac_single_blob():
    m = _mask((20, 20), [(2, 8, 2, 8)])          # one blob
    n, frac = pipeline._largest_cc_frac(m)
    assert n == 1 and frac == 1.0


def test_largest_cc_frac_two_blobs():
    # 36px blob + 4px blob -> 2 components, largest = 36/40 = 0.9
    m = _mask((30, 30), [(2, 8, 2, 8), (20, 22, 20, 22)])
    n, frac = pipeline._largest_cc_frac(m)
    assert n == 2
    assert abs(frac - 36 / 40) < 1e-9


# ---------------------------------------------------------------------------
# _select_anchor_mask, the ranking
# ---------------------------------------------------------------------------

def _prompts_at(x, y):
    return Prompts(points_sam=np.array([[x, y]], dtype=float), labels=np.array([1]))


def _prompts_pos_neg(pos_xy, *neg_xy):
    """A positive point plus N negative (neighbour) points."""
    pts = np.array([pos_xy, *neg_xy], dtype=float)
    labels = np.array([1] + [0] * len(neg_xy), dtype=int)
    return Prompts(points_sam=pts, labels=labels)


def test_select_prefers_containment_over_iou():
    hw = (100, 100)
    # cand 0: high IoU but does NOT contain the node; cand 1: contains the node, lower IoU
    cand0 = _mask(hw, [(0, 10, 0, 10)])
    cand1 = _mask(hw, [(48, 53, 48, 53)])        # sits on the node at (50,50)
    masks = np.stack([cand0, cand1])
    scores = np.array([0.99, 0.40])
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(1e-5, 0.9))
    assert idx == 1                              # containment beats raw IoU


def test_select_rejects_runaway_grab_on_area_before_cc():
    hw = (100, 100)
    # both contain the node. cand 0 = runaway background grab (90% of frame, one clean
    # blob, lcc~1.0); cand 1 = a tidy small blob within plausible area. Area plausibility
    # must rank cand 1 above cand 0 EVEN though cand 0 wins on single-CC.
    runaway = _mask(hw, [(5, 95, 5, 95)])        # 8100/10000 = 0.81 area_frac
    tidy = _mask(hw, [(45, 56, 45, 56)])         # contains (50,50), small
    masks = np.stack([runaway, tidy])
    scores = np.array([0.95, 0.5])
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(1e-5, 0.4))
    assert idx == 1                              # plausible-area beats single-CC


def test_select_uses_cc_then_iou_when_containment_and_area_tie():
    hw = (100, 100)
    # both contain node + plausible area. cand 0 fragmented (two blobs), cand 1 single blob.
    frag = _mask(hw, [(48, 53, 48, 53), (10, 14, 10, 14)])   # node blob + a stray fragment
    clean = _mask(hw, [(46, 55, 46, 55)])                    # single blob on node
    masks = np.stack([frag, clean])
    scores = np.array([0.8, 0.7])
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(1e-5, 0.9))
    assert idx == 1                              # single-CC breaks the tie over higher IoU


def test_select_is_graceful_when_nothing_contains():
    hw = (100, 100)
    # no candidate contains the node -> falls through to area/CC/IoU, still returns one.
    a = _mask(hw, [(0, 5, 0, 5)])
    b = _mask(hw, [(0, 8, 0, 8)])                # bigger single blob, plausible, higher IoU
    masks = np.stack([a, b])
    scores = np.array([0.3, 0.6])
    idx, mask_b, score = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(1e-5, 0.9))
    assert idx == 1 and score == 0.6
    assert mask_b.dtype == bool


def test_select_no_positive_point_ignores_containment():
    hw = (100, 100)
    # prompts with no positive label -> containment is always False for all; ranking
    # falls to area/CC/IoU. Should not crash, should pick the plausible single blob.
    prompts = Prompts(points_sam=np.array([[10, 10]], dtype=float), labels=np.array([0]))
    a = _mask(hw, [(0, 5, 0, 5)])
    b = _mask(hw, [(40, 50, 40, 50)])
    masks = np.stack([a, b])
    scores = np.array([0.4, 0.5])
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(1e-5, 0.9))
    assert idx == 1


# ---------------------------------------------------------------------------
# _select_anchor_mask, the exclude_neg (anti-bleed) refinement
# ---------------------------------------------------------------------------

def test_exclude_neg_prefers_mask_without_negative_over_higher_iou():
    hw = (100, 100)
    # positive node at (50,50), a negative neighbour at (70,70).
    # cand 0 contains BOTH the positive and the negative (a bleeding mask), higher IoU;
    # cand 1 contains only the positive. With exclude_neg, cand 1 wins.
    bleed = _mask(hw, [(45, 75, 45, 75)])        # covers (50,50) AND (70,70)
    tight = _mask(hw, [(46, 55, 46, 55)])        # covers (50,50) only
    masks = np.stack([bleed, tight])
    scores = np.array([0.95, 0.50])
    prompts = _prompts_pos_neg((50, 50), (70, 70))
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(1e-5, 0.9),
        exclude_neg=True)
    assert idx == 1                              # excluding the negative beats raw IoU


def test_exclude_neg_off_keeps_original_ranking():
    hw = (100, 100)
    # Same masks as above; with exclude_neg OFF the negative is ignored, so both contain
    # the positive + are plausible + single-CC, and the higher-IoU bleeding mask wins.
    bleed = _mask(hw, [(45, 75, 45, 75)])
    tight = _mask(hw, [(46, 55, 46, 55)])
    masks = np.stack([bleed, tight])
    scores = np.array([0.95, 0.50])
    prompts = _prompts_pos_neg((50, 50), (70, 70))
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(1e-5, 0.9),
        exclude_neg=False)
    assert idx == 0                              # original behaviour: IoU tiebreak wins


def test_exclude_neg_graceful_when_all_contain_a_negative():
    hw = (100, 100)
    # both candidates contain the positive AND the negative -> no_neg ties at 0; ranking
    # falls through to area/CC/IoU, still returns a candidate.
    a = _mask(hw, [(45, 75, 45, 75)])            # contains (50,50)+(70,70), higher IoU
    b = _mask(hw, [(40, 80, 40, 80)])            # also contains both, lower IoU, bigger
    masks = np.stack([a, b])
    scores = np.array([0.8, 0.6])
    prompts = _prompts_pos_neg((50, 50), (70, 70))
    idx, mask_b, _ = pipeline._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(1e-5, 0.9),
        exclude_neg=True)
    assert idx == 0 and mask_b.dtype == bool     # graceful: IoU tiebreak, never crashes


def test_exclude_neg_containment_still_dominates():
    hw = (100, 100)
    # cand 0 misses the positive entirely but contains no negative; cand 1 contains the
    # positive AND a negative. Positive-containment outranks negative-exclusion, so cand 1.
    miss = _mask(hw, [(5, 12, 5, 12)])           # no positive, no negative
    onpos = _mask(hw, [(45, 75, 45, 75)])        # positive (50,50) + negative (70,70)
    masks = np.stack([miss, onpos])
    scores = np.array([0.9, 0.5])
    prompts = _prompts_pos_neg((50, 50), (70, 70))
    idx, _, _ = pipeline._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(1e-5, 0.9),
        exclude_neg=True)
    assert idx == 1                              # containing the positive wins over excluding a negative


# ---------------------------------------------------------------------------
# Plain runner (no pytest required)
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                      # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
