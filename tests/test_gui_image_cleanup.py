"""Cleanup of the GUI's image-phase re-predict (gui.image_cleanup_kwargs,
gui.ReviewGUI._clean_predicted, pipeline.clean_mask).

SAM2's raw image-mode mask carries detached fuzz outside the cell and a frayed boundary.
The batch already cleans that up; these guard that the GUI now runs the same sequence, that
the reviewer's dock settings actually reach it, and that "off" really means untouched.

Torch-free / napari-free: the helper is a module-level pure function and _clean_predicted
only reads attributes off self, so a stub stands in for the viewer (same tactic as
tests/test_gui_box.py).

    py -3 -m pytest tests/test_gui_image_cleanup.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import gui
import pipeline


def _mask_with_speck_and_second_blob():
    """A cell (30x30), a real second cross-section (10x10 = 100 px), and a 2x2 speck."""
    m = np.zeros((128, 128), dtype=bool)
    m[20:50, 20:50] = True        # the cell
    m[80:90, 80:90] = True        # a genuine second process, 100 px
    m[5:7, 100:102] = True        # detached fuzz, 4 px
    return m


# ---------------------------------------------------------------------------
# image_cleanup_kwargs: the level -> clean_mask arguments mapping
# ---------------------------------------------------------------------------

def test_off_means_no_cleanup_at_all():
    assert gui.image_cleanup_kwargs(gui.CLEANUP_OFF) is None


def test_specks_keeps_every_component_over_the_floor():
    kw = gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS, min_island_px=64)
    assert kw["keep_largest_cc"] is False, "a real second cross-section must survive"
    assert kw["remove_islands_min_size"] == 64


def test_largest_blob_keeps_one_component():
    kw = gui.image_cleanup_kwargs(gui.CLEANUP_LARGEST)
    assert kw["keep_largest_cc"] is True


def test_neither_level_fills_a_large_cavity():
    for level in (gui.CLEANUP_SPECKS, gui.CLEANUP_LARGEST):
        assert gui.image_cleanup_kwargs(level)["fill_holes"] is False


def test_sizes_come_from_the_dock():
    kw = gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS, min_island_px=250, smooth_radius=3)
    assert kw["remove_islands_min_size"] == 250 and kw["smooth_radius"] == 3


def test_negative_sizes_are_clamped_off_not_passed_through():
    kw = gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS, min_island_px=-5, smooth_radius=-1)
    assert kw["remove_islands_min_size"] == 0 and kw["smooth_radius"] == 0


def test_an_unknown_level_raises_rather_than_silently_skipping():
    with pytest.raises(ValueError):
        gui.image_cleanup_kwargs("scrub it")


# ---------------------------------------------------------------------------
# pipeline.clean_mask: the sequence both the batch and the GUI now run
# ---------------------------------------------------------------------------

def test_specks_settings_drop_the_fuzz_and_keep_both_real_blobs():
    from skimage.measure import label as cc_label
    out = pipeline.clean_mask(_mask_with_speck_and_second_blob(),
                              **gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS))
    assert not out[5:7, 100:102].any(), "the 4 px speck should be gone"
    assert out[20:50, 20:50].any() and out[80:90, 80:90].any()
    assert int(cc_label(out, connectivity=2).max()) == 2


def test_largest_blob_settings_leave_only_the_cell():
    out = pipeline.clean_mask(_mask_with_speck_and_second_blob(),
                              **gui.image_cleanup_kwargs(gui.CLEANUP_LARGEST))
    assert out[20:50, 20:50].any()
    assert not out[80:90, 80:90].any(), "keep_largest_cc drops the second process"


def test_a_sam2_shaped_1hw_mask_is_not_emptied():
    """SAM2 hands back (1, H, W); a 3D opening on that axis empties the mask."""
    m = _mask_with_speck_and_second_blob()[None, ...]
    out = pipeline.clean_mask(m, **gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS))
    assert out.ndim == 2 and out.any()


def test_empty_in_empty_out():
    out = pipeline.clean_mask(np.zeros((32, 32), bool),
                              **gui.image_cleanup_kwargs(gui.CLEANUP_SPECKS))
    assert not out.any()


def test_all_ops_off_is_the_postprocess_baseline():
    """With every size-aware op at 0, clean_mask must equal postprocess_mask, which is
    what makes the orchestrator's switch to it a faithful extraction."""
    m = _mask_with_speck_and_second_blob()
    assert np.array_equal(
        pipeline.clean_mask(m, open_px=1, close_px=1, keep_largest_cc=True, fill_holes=True),
        pipeline.postprocess_mask(m, open_px=1, close_px=1, keep_largest_cc=True,
                                  fill_holes=True))


# ---------------------------------------------------------------------------
# _clean_predicted: the dock settings actually reach the cleanup
# ---------------------------------------------------------------------------

def _stub(**widgets):
    return types.SimpleNamespace(**{k: types.SimpleNamespace(value=v)
                                    for k, v in widgets.items()})


def test_clean_predicted_uses_the_docks_level():
    stub = _stub(_clean_mode=gui.CLEANUP_LARGEST, _clean_island_spin=64, _clean_smooth_spin=1)
    out, desc = gui.ReviewGUI._clean_predicted(stub, _mask_with_speck_and_second_blob())
    assert not out[80:90, 80:90].any()
    assert "px" in desc and gui.CLEANUP_LARGEST in desc


def test_clean_predicted_off_returns_the_mask_untouched_and_says_nothing():
    m = _mask_with_speck_and_second_blob()
    stub = _stub(_clean_mode=gui.CLEANUP_OFF, _clean_island_spin=64, _clean_smooth_spin=1)
    out, desc = gui.ReviewGUI._clean_predicted(stub, m)
    assert np.array_equal(out, m) and desc == ""


def test_missing_widgets_mean_off_never_a_silent_transformation():
    m = _mask_with_speck_and_second_blob()
    out, desc = gui.ReviewGUI._clean_predicted(types.SimpleNamespace(), m)
    assert np.array_equal(out, m) and desc == ""


def test_the_description_reports_the_pixels_lost():
    m = _mask_with_speck_and_second_blob()
    stub = _stub(_clean_mode=gui.CLEANUP_SPECKS, _clean_island_spin=64, _clean_smooth_spin=0)
    out, desc = gui.ReviewGUI._clean_predicted(stub, m)
    assert f"{int(m.sum())}->{int(out.sum())} px" in desc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
