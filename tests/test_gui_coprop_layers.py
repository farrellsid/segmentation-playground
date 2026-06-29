"""Unit tests for the co-propagation overlay-layer helpers
(gui.ReviewGUI._neighbor_label_stack / ._diff_stack).

Torch-free / napari-free: gui.py imports torch and napari only lazily, so importing
the module and exercising these pure helpers needs no GPU and no viewer (same tactic as
test_gui_box). The helpers do not touch `self`, so we call them unbound with self=None.

Regression focus: SAM2's per-object video logits come back shaped (1, H, W); the label
builder must squeeze that leading dim (like _label_stack_from_segments) rather than index
a 2D frame with a 3D boolean mask.

Run either way:
    py -3 -m pytest tests/test_gui_coprop_layers.py
    py -3 tests/test_gui_coprop_layers.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import gui


def _mask3d(hw, box):
    """A (1, H, W) bool mask (the SAM2 video-logits shape) with [y0,y1,x0,x1] filled."""
    H, W = hw
    m = np.zeros((1, H, W), dtype=bool)
    y0, y1, x0, x1 = box
    m[0, y0:y1, x0:x1] = True
    return m


def test_neighbor_label_stack_squeezes_leading_dim_and_paints_each_obj():
    H, W, t = 10, 12, 2
    # frame 0: neighbor 2 in a corner, neighbor 3 elsewhere; frame 1: only neighbor 2.
    segments = {
        0: {1: _mask3d((H, W), (0, 2, 0, 2)),      # target obj, ignored here
            2: _mask3d((H, W), (0, 3, 0, 3)),
            3: _mask3d((H, W), (5, 8, 5, 8))},
        1: {2: _mask3d((H, W), (1, 2, 1, 2))},
    }
    out = gui.ReviewGUI._neighbor_label_stack(None, segments, [2, 3], t, (H, W))
    assert out.shape == (t, H, W)                  # no IndexError on the (1,H,W) masks
    assert out[0, 0, 0] == 2                        # neighbor 2 painted with its obj id
    assert out[0, 6, 6] == 3                        # neighbor 3 painted with its obj id
    assert out[0, 9, 9] == 0                        # background stays 0
    assert out[1, 1, 1] == 2
    assert not (out == 1).any()                     # target obj (1) is NOT in this layer


def test_diff_stack_marks_lost_and_gained_target_pixels():
    H, W, t = 4, 4, 1
    obj = 1
    alone = np.zeros((t, H, W), dtype=np.uint8)
    withn = np.zeros((t, H, W), dtype=np.uint8)
    alone[0, 0, 0] = obj          # present alone, absent with neighbors -> lost (1)
    withn[0, 3, 3] = obj          # absent alone, present with neighbors -> gained (2)
    alone[0, 1, 1] = obj
    withn[0, 1, 1] = obj          # present in both -> unchanged (0)
    diff = gui.ReviewGUI._diff_stack(None, alone, withn, obj, t, (H, W))
    assert diff[0, 0, 0] == 1      # lost (carved-out bleed)
    assert diff[0, 3, 3] == 2      # gained
    assert diff[0, 1, 1] == 0      # unchanged


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
