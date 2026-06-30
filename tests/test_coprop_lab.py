"""Unit tests for coprop_lab's pure label/diff helpers.

Torch-free and napari-free: coprop_lab imports torch and napari only lazily, so
importing the module and exercising these helpers needs no GPU and no viewer.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import coprop_lab


def _mask3d(hw, box):
    """A (1, H, W) bool mask (the SAM2 video-logits shape) with [y0,y1,x0,x1] filled."""
    H, W = hw
    m = np.zeros((1, H, W), dtype=bool)
    y0, y1, x0, x1 = box
    m[0, y0:y1, x0:x1] = True
    return m


def test_label_stack_squeezes_and_paints_each_requested_obj():
    H, W, t = 10, 12, 2
    segments = {
        0: {1: _mask3d((H, W), (0, 2, 0, 2)),
            2: _mask3d((H, W), (0, 3, 0, 3)),
            3: _mask3d((H, W), (5, 8, 5, 8))},
        1: {2: _mask3d((H, W), (1, 2, 1, 2))},
    }
    out = coprop_lab.label_stack(segments, [2, 3], t, (H, W))
    assert out.shape == (t, H, W)          # no IndexError on the (1,H,W) masks
    assert out[0, 0, 0] == 2                # neighbor 2 painted with its id
    assert out[0, 6, 6] == 3                # neighbor 3 painted with its id
    assert out[0, 9, 9] == 0                # background stays 0
    assert out[1, 1, 1] == 2
    assert not (out == 1).any()             # obj 1 not requested, so never painted


def test_label_stack_single_target_obj():
    H, W, t = 4, 5, 1
    segments = {0: {1: _mask3d((H, W), (0, 2, 0, 2))}}
    out = coprop_lab.label_stack(segments, [1], t, (H, W))
    assert out[0, 0, 0] == 1
    assert out[0, 3, 3] == 0


def test_build_diff_stack_marks_lost_and_gained():
    H, W, t = 4, 4, 1
    obj = 1
    alone = np.zeros((t, H, W), dtype=np.uint8)
    withn = np.zeros((t, H, W), dtype=np.uint8)
    alone[0, 0, 0] = obj            # present alone, absent with neighbors -> lost
    withn[0, 2, 2] = obj            # absent alone, present with neighbors -> gained
    alone[0, 1, 1] = obj            # present in both -> unchanged
    withn[0, 1, 1] = obj
    diff = coprop_lab.build_diff_stack(alone, withn, obj)
    assert diff[0, 0, 0] == 1        # lost
    assert diff[0, 2, 2] == 2        # gained
    assert diff[0, 1, 1] == 0        # unchanged
    assert diff[0, 3, 3] == 0        # background


def test_load_em_stack_reads_indexed_jpegs(tmp_path):
    import cv2
    # two 4x6 BGR frames named like the chain's 0-indexed jpegs ({idx:05d}.jpg)
    for i in range(2):
        img = np.full((4, 6, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), img)
    stack = coprop_lab.load_em_stack(str(tmp_path), 2)
    assert stack.shape == (2, 4, 6, 3)
    assert stack.dtype == np.uint8


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
