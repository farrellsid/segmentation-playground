import numpy as np
import pytest

import run_perframe


class FakeAMG:
    """Stand-in for SAM2AutomaticMaskGenerator.generate: returns AMG-style dicts."""
    def __init__(self, masks): self._masks = masks
    def generate(self, image):
        return [{"segmentation": m, "area": int(m.sum()), "predicted_iou": 0.9,
                 "stability_score": 0.95} for m in self._masks]


def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_segment_frame_amg_labels_and_keeps_competitors():
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    junk = _disk(20, 4, 3)
    amg = FakeAMG([_disk(10, 10, 5), _disk(30, 30, 5), junk])  # 2 cells + 1 junk
    cell_masks, lab, score = run_perframe.segment_frame_amg(
        amg, frame, node_index, mem, match="metric", resolver="argmax",
        cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score
    # both cells actually carry the pixels of their matched AMG disk
    assert cell_masks["AVAL"][10, 10]
    assert cell_masks["AVAR"][30, 30]
    # the junk competitor is resolved into the label map (it can push back on the
    # cell masks) but never survives as a labelled cell: its footprint is background
    # in the cell-only label map once competitor labels are dropped.
    assert not lab[junk].any()


def test_segment_frame_amg_competitor_pushback_shrinks_target():
    """The disjoint-junk test above never exercises the actual pushback: a competitor
    that OVERLAPS a target's mask and contests some of its pixels during overlap
    resolution. Here the competitor's disk overlaps the target's eastern edge, and sits
    far enough from the node that every contested pixel in the overlap lens is nearer the
    competitor's own centroid than the node, so the competitor wins the whole lens. This
    is the regression coverage for Approach 2's central claim: competitors take part in
    resolution and can push bleed off a cell, even though they never survive as a named
    cell afterwards.
    """
    shape = (50, 50)
    frame = np.full(shape + (3,), 128, np.uint8)
    node_index = [(11, 20, "TGT", "n1")]
    mem = np.zeros(shape, np.float32)
    target = _disk(20, 20, 10, shape=shape)
    competitor = _disk(35, 20, 10, shape=shape)  # overlaps target's east edge (x in [25, 30])
    cfg = run_perframe.PerframeCfg(scale=8)

    cells_with, lab_with, _ = run_perframe.segment_frame_amg(
        FakeAMG([target, competitor]), frame, node_index, mem,
        match="metric", resolver="argmax", cfg=cfg)
    cells_without, _lab_without, _ = run_perframe.segment_frame_amg(
        FakeAMG([target]), frame, node_index, mem,
        match="metric", resolver="argmax", cfg=cfg)

    # the competitor claimed some of the contested overlap, so the target keeps less
    # than it would have on its own.
    assert int(cells_with["TGT"].sum()) < int(cells_without["TGT"].sum())
    # the competitor's whole footprint (its exclusive area plus the overlap lens it won)
    # never survives as a named cell: it is background in the returned cell-only label map.
    assert not lab_with[competitor].any()


def test_segment_frame_amg_unmatched_cell_counts_as_uncovered():
    """Fairness regression: a cell WITH a node in this frame that AMG fails to match must
    still appear in cell_masks as an empty mask, so score_frame's own_coverage counts it as
    uncovered. Before this fix, an unmatched cell was simply absent from cell_masks and
    score_frame's mean over cell_masks.keys() silently skipped it, inflating own_coverage
    (here it would have read 1.0 instead of the correct 0.5).
    """
    shape = (40, 40)
    frame = np.full(shape + (3,), 128, np.uint8)
    # B has a node in the frame, but no AMG mask anywhere near it: unmatchable.
    node_index = [(10, 10, "A", "a"), (30, 30, "B", "b")]
    mem = np.zeros(shape, np.float32)
    amg = FakeAMG([_disk(10, 10, 5, shape=shape)])  # only covers A's node

    cell_masks, _lab, score = run_perframe.segment_frame_amg(
        amg, frame, node_index, mem, match="metric", resolver="argmax",
        cfg=run_perframe.PerframeCfg(scale=8))

    assert set(cell_masks) == {"A", "B"}
    assert cell_masks["A"].sum() > 0
    assert cell_masks["B"].sum() == 0
    assert score["own_coverage"] == 0.5


def test_segment_frame_amg_overlap_fraction_is_pre_resolution():
    """The other half of the unify-scoring fix: Approach 2 used to build cell_masks from
    the already-resolved (disjoint by construction) label map and score overlap_fraction
    on THAT, so it read ~0 no matter how much the matched masks originally fought over
    pixels. Here two matched masks (one per node, each containing only its own node) share
    a 10-column band before resolution; overlap_fraction must reflect that pre-resolution
    fight (200 shared px / 800 total px = 0.25), even though the returned cell_masks are
    disjoint.
    """
    shape = (20, 30)
    frame = np.full(shape + (3,), 128, np.uint8)
    node_index = [(5, 10, "A", "a"), (25, 10, "B", "b")]
    mem = np.zeros(shape, np.float32)
    mask_a = np.zeros(shape, bool); mask_a[:, 0:20] = True     # contains A, not B
    mask_b = np.zeros(shape, bool); mask_b[:, 10:30] = True    # contains B, not A
    amg = FakeAMG([mask_a, mask_b])

    cell_masks, _lab, score = run_perframe.segment_frame_amg(
        amg, frame, node_index, mem, match="metric", resolver="argmax",
        cfg=run_perframe.PerframeCfg(scale=8))

    assert score["overlap_fraction"] == pytest.approx(0.25)
    assert not (cell_masks["A"] & cell_masks["B"]).any()
