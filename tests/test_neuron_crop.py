"""Unit tests for the neuron-review crop helpers (pipeline.neuron_crop_window /
_neuron_skeleton_box_tif / remap_mask_to_window): the per-neuron crop window (_ncrop)
that unifies a neuron's mixed-space branches, and the remap that places a branch mask
into an arbitrary crop grid.

Torch-free / napari-free (only cv2/pandas/numpy + pipeline, which defers torch):
    py -3 -m pytest tests/test_neuron_crop.py
    py -3 tests/test_neuron_crop.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import pipeline
from pipeline import PipelineConfig


def _df():
    # two chains of one neuron; node xy_tif span x:1000..1600, y:1000..1400
    return pd.DataFrame({
        "node_id": ["a", "b", "c", "d"],
        "z": [1000, 1001, 1002, 1003],
        "x_tif": [1000.0, 1200.0, 1400.0, 1600.0],
        "y_tif": [1000.0, 1100.0, 1300.0, 1400.0],
    })


# ---------------------------------------------------------------------------
# _neuron_skeleton_box_tif / neuron_crop_window
# ---------------------------------------------------------------------------

def test_neuron_skeleton_box_unions_all_chains():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b"]}, {"cell_name": "N", "nodes": ["c", "d"]}]
    box = pipeline._neuron_skeleton_box_tif(chains, df)
    assert box == (1000.0, 1000.0, 1600.0, 1400.0)


def test_neuron_crop_window_contains_skeleton_and_pads():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b"]}, {"cell_name": "N", "nodes": ["c", "d"]}]
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop_pad_tif=100, chain_crop_scale=2, chain_crop_max_px=100000)
    cw = pipeline.neuron_crop_window(chains, df, cfg=cfg, image_hw_tif=(8000, 8000))
    # skeleton bbox 600x400 + 100 pad/side -> 800x600, centered on (1300, 1200)
    assert cw.size_tif == (800, 600)
    assert cw.origin_tif == (900.0, 900.0)
    assert cw.crop_scale == 2


def test_neuron_crop_window_bumps_scale_over_max_px():
    df = _df()
    chains = [{"cell_name": "N", "nodes": ["a", "b", "c", "d"]}]
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop_pad_tif=0, chain_crop_scale=2, chain_crop_max_px=300)
    cw = pipeline.neuron_crop_window(chains, df, cfg=cfg, image_hw_tif=(8000, 8000))
    # longest edge 600 _tif / 300 -> ceil 2; max(2, 2) = 2
    assert cw.crop_scale == 2


# ---------------------------------------------------------------------------
# remap_mask_to_window
# ---------------------------------------------------------------------------

def test_remap_mask_to_window_places_and_scales():
    from sam2_utils.alignment import CropWindow
    # dst window: tif origin (1000,1000), 400x400 tif, crop_scale 2 -> 200x200 grid
    dst = CropWindow(origin_tif=(1000.0, 1000.0), size_tif=(400, 400), crop_scale=2, sam_scale=8)
    # source mask: a 100x100 tif footprint at tif (1100,1100), given at crop_scale 1 (100x100 px)
    src = np.zeros((100, 100), dtype=bool)
    src[20:80, 20:80] = True
    out = pipeline.remap_mask_to_window(
        src, src_origin_tif=(1100.0, 1100.0), src_size_tif=(100, 100), dst_cw=dst)
    assert out.shape == (200, 200)              # dst.crop_hw
    # src footprint maps to dst crop coords: origin (1100-1000)/2 = 50, size 100/2 = 50.
    # the inner foreground [20:80] of a 100px source -> ~[60, 90) in the placed 50px region.
    ys, xs = np.where(out)
    assert 50 <= xs.min() < 70 and 80 <= xs.max() < 100
    assert 50 <= ys.min() < 70 and 80 <= ys.max() < 100


def test_remap_mask_to_window_clips_out_of_window():
    from sam2_utils.alignment import CropWindow
    dst = CropWindow(origin_tif=(1000.0, 1000.0), size_tif=(400, 400), crop_scale=2, sam_scale=8)
    src = np.ones((50, 50), dtype=bool)         # footprint entirely left of the window
    out = pipeline.remap_mask_to_window(
        src, src_origin_tif=(0.0, 0.0), src_size_tif=(50, 50), dst_cw=dst)
    assert out.shape == (200, 200) and not out.any()


# ---------------------------------------------------------------------------
# plain runner
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
