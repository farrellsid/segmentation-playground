"""Unit tests for pipeline.postprocess_mask (torch-free, no GPU).

Regression guard for the (1, H, W) shape bug: SAM2's video predictor yields each mask as
(1, H, W); postprocess_mask must squeeze that channel axis, else a 3D binary_opening erodes the
singleton axis and empties every mask (the silent postproc failure the A/B campaign surfaced). Run:
    py -3 -m pytest tests/test_postprocess_mask.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import pipeline


def _blob(hw=(64, 64)):
    m = np.zeros(hw, dtype=bool)
    m[20:50, 20:50] = True          # a solid 30x30 block, survives a 1px opening
    return m


def test_2d_blob_survives_postproc():
    out = pipeline.postprocess_mask(_blob(), open_px=1, close_px=1,
                                    keep_largest_cc=True, fill_holes=True)
    assert out.any() and out.ndim == 2


def test_3d_singleton_does_not_empty_the_mask():
    m2 = _blob()
    m3 = m2[None, ...]                                   # (1, H, W), as SAM2 yields
    out = pipeline.postprocess_mask(m3, open_px=1, close_px=1,
                                    keep_largest_cc=True, fill_holes=True)
    assert out.any(), "(1,H,W) input must not be emptied by a 3D opening"
    # and it matches the 2D result
    out2 = pipeline.postprocess_mask(m2, open_px=1, close_px=1,
                                     keep_largest_cc=True, fill_holes=True)
    assert np.array_equal(out, out2)


def test_empty_in_empty_out():
    assert not pipeline.postprocess_mask(np.zeros((32, 32), bool)).any()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
