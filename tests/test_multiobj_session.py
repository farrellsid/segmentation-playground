"""Unit test for MultiObjectPropagationSession's flag set-and-restore.

Torch-free: we drive the session with a fake predictor that records attribute
writes, so we can assert the non-overlap flags are set on enter and RESTORED on
close (the 'do not perturb a shared predictor' contract). The propagation math
itself is torch-bound and exercised manually in the GUI, not here.

Run either way:
    py -3 -m pytest tests/test_multiobj_session.py -v
    py -3 tests/test_multiobj_session.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline


class _FakePredictor:
    """Records the seed calls and lets us read the non-overlap flags."""
    def __init__(self, non_overlap_masks=False, non_overlap_masks_for_mem_enc=False):
        self.non_overlap_masks = non_overlap_masks
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.seeds = []

    def init_state(self, **kw):
        return {"state": True}

    def reset_state(self, state):
        pass

    def add_new_points_or_box(self, **kw):
        self.seeds.append(kw)

    def add_new_mask(self, **kw):
        self.seeds.append(kw)

    # never called in this test (no propagate), present for completeness
    def propagate_in_video(self, state, **kw):
        return iter(())


def test_flags_set_on_enter_and_restored_on_close():
    vp = _FakePredictor(non_overlap_masks=False, non_overlap_masks_for_mem_enc=False)
    sess = pipeline.MultiObjectPropagationSession(
        vp, "frames", non_overlap=True, non_overlap_mem_enc=True)
    # set while the session is live
    assert vp.non_overlap_masks is True
    assert vp.non_overlap_masks_for_mem_enc is True
    sess.close()
    # restored to the original values
    assert vp.non_overlap_masks is False
    assert vp.non_overlap_masks_for_mem_enc is False


def test_close_is_idempotent_and_restores_once():
    vp = _FakePredictor(non_overlap_masks=True, non_overlap_masks_for_mem_enc=False)
    sess = pipeline.MultiObjectPropagationSession(vp, "frames", non_overlap=False)
    assert vp.non_overlap_masks is False     # forced off while live
    sess.close()
    sess.close()                              # second close is a no-op
    assert vp.non_overlap_masks is True       # restored to original


def test_seed_sends_one_call_per_object_with_obj_id():
    vp = _FakePredictor()
    import numpy as np
    with pipeline.MultiObjectPropagationSession(vp, "frames") as sess:
        p = pipeline.Prompts(points_sam=np.array([[5.0, 5.0]]), labels=np.array([1]),
                             box_sam=np.array([1.0, 1.0, 9.0, 9.0], dtype=np.float32))
        sess.seed(1, p, 0)
        sess.seed(2, p, 0)
    obj_ids = [s["obj_id"] for s in vp.seeds]
    assert obj_ids == [1, 2]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
