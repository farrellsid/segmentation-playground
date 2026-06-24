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


# ---------------------------------------------------------------------------
# remove_small_islands: drop specks, keep ALL large components
# ---------------------------------------------------------------------------

def test_remove_small_islands_drops_specks_keeps_blob():
    m = np.zeros((64, 64), bool)
    m[10:40, 10:40] = True          # 900 px blob
    m[50:52, 50:52] = True          # 4 px speck
    out = pipeline.remove_small_islands(m, min_size=50)
    assert out[20, 20] and not out[51, 51]


def test_remove_small_islands_keeps_multiple_large():
    m = np.zeros((64, 64), bool)
    m[5:25, 5:25] = True            # 400 px
    m[40:60, 40:60] = True          # 400 px (a legit second component)
    out = pipeline.remove_small_islands(m, min_size=50)
    assert out[10, 10] and out[50, 50]


def test_remove_small_islands_squeeze_and_empty():
    m3 = np.zeros((1, 32, 32), bool)
    m3[0, 10:20, 10:20] = True
    out = pipeline.remove_small_islands(m3, min_size=4)
    assert out.ndim == 2 and out.any()
    assert not pipeline.remove_small_islands(np.zeros((16, 16), bool), min_size=4).any()


# ---------------------------------------------------------------------------
# fill_small_holes: fill small holes, keep a large cavity
# ---------------------------------------------------------------------------

def test_fill_small_holes_fills_small_keeps_large():
    m = np.ones((60, 60), bool)
    m[10:12, 10:12] = False         # 4 px hole
    m[30:50, 30:50] = False         # 400 px hole
    out = pipeline.fill_small_holes(m, area_threshold=10)
    assert out[10, 10]              # small hole filled
    assert not out[40, 40]          # large hole preserved


# ---------------------------------------------------------------------------
# smooth_edges: shave thin protrusions
# ---------------------------------------------------------------------------

def test_smooth_edges_removes_thin_spike_keeps_body():
    m = np.zeros((64, 64), bool)
    m[20:44, 20:44] = True          # solid body
    m[31:33, 44:60] = True          # a thin 2px-tall spike off the side
    out = pipeline.smooth_edges(m, radius=2)
    assert out[30, 30]              # body survives
    assert not out[31, 55]          # thin spike shaved


def test_smooth_edges_empty_and_solid():
    assert not pipeline.smooth_edges(np.zeros((16, 16), bool), radius=2).any()
    solid = np.zeros((64, 64), bool)
    solid[10:54, 10:54] = True
    assert pipeline.smooth_edges(solid, radius=2).any()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
