import numpy as np
import pytest
torch = pytest.importorskip("torch")   # smoke only runs where torch is present
import run_perframe


def test_segment_frame_prompt_shapes(monkeypatch):
    # a fake image predictor: returns one candidate mask per set_image/predict, torch-free
    class FakePred:
        def set_image(self, img): self._hw = img.shape[:2]
        def predict(self, **kw):
            h, w = self._hw
            m = np.zeros((1, h, w), bool); m[0, :h // 2, :w // 2] = True
            return m, np.array([0.9]), np.zeros((1, 256, 256), np.float32)
        def reset_predictor(self): pass
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    cell_masks, lab, score = run_perframe.segment_frame_prompt(
        FakePred(), frame, node_index, mem, negatives=True, selection="pred_iou",
        resolver="argmax", cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score


class _FullFrameFakePred:
    """Every predict() call returns the whole frame as its one candidate, regardless of
    which node it was called for, so two nodes' raw masks come out fully overlapping and
    the resolver is the only thing left that can split them apart."""
    def set_image(self, img): self._hw = img.shape[:2]
    def predict(self, **kw):
        h, w = self._hw
        m = np.ones((1, h, w), bool)
        return m, np.array([0.9]), np.zeros((1, 256, 256), np.float32)
    def reset_predictor(self): pass


def test_resolver_changes_scored_numbers_for_prompt_approach():
    """The bug this guards against: Approach 1 used to score the raw pre-resolution union
    masks, so --resolver never touched a single scored number (a sweep's argmax and
    watershed rows came out identical). Two nodes here get the IDENTICAL raw full-frame
    mask (via _FullFrameFakePred), so any post-resolution difference can only come from
    the resolver, not from the raw masks. A membrane wall placed off-centre from the
    euclidean bisector between the two nodes makes argmax (nearest-node, membrane-blind)
    and watershed (membrane-as-elevation) split the frame differently, which
    mean_boundary_on_membrane should pick up.
    """
    shape = (20, 30)
    frame = np.zeros(shape + (3,), np.uint8)
    node_index = [(5, 10, "A", "a"), (25, 10, "B", "b")]
    mem = np.zeros(shape, np.float32)
    mem[:, 8] = 1.0   # a wall well off the euclidean bisector (x=15) between A and B
    cfg = run_perframe.PerframeCfg(scale=8)

    _cm_a, _lab_a, score_argmax = run_perframe.segment_frame_prompt(
        _FullFrameFakePred(), frame, node_index, mem, negatives=False,
        selection="pred_iou", resolver="argmax", cfg=cfg)
    _cm_w, _lab_w, score_watershed = run_perframe.segment_frame_prompt(
        _FullFrameFakePred(), frame, node_index, mem, negatives=False,
        selection="pred_iou", resolver="watershed", cfg=cfg)

    assert score_argmax["mean_boundary_on_membrane"] != score_watershed["mean_boundary_on_membrane"]


def test_overlap_fraction_is_pre_resolution_even_though_resolved_masks_are_disjoint():
    """overlap_fraction must read the PRE-resolution fight for pixels (the raw, fully
    overlapping masks below), not the post-resolution (disjoint by construction) masks,
    or it would read ~0 regardless of how contested the raw step was.
    """
    shape = (20, 30)
    frame = np.zeros(shape + (3,), np.uint8)
    node_index = [(5, 10, "A", "a"), (25, 10, "B", "b")]
    mem = np.zeros(shape, np.float32)
    cfg = run_perframe.PerframeCfg(scale=8)

    cell_masks, _lab, score = run_perframe.segment_frame_prompt(
        _FullFrameFakePred(), frame, node_index, mem, negatives=False,
        selection="pred_iou", resolver="argmax", cfg=cfg)

    # the raw masks were identical (full frame, full frame): total overlap == total area.
    assert score["overlap_fraction"] > 0.0
    # yet the returned cell_masks are the RESOLVED, disjoint masks: no shared pixels.
    assert not (cell_masks["A"] & cell_masks["B"]).any()
