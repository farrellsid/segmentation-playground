"""Unit tests for pipeline.chain_masks_in_sam, the aggregation-prep reader that
puts any finished chain's masks back onto the common _sam grid with their placement,
so a per-neuron z-union can paste legacy _sam AND tier-2 _pcrop chains on one grid.

The point being guarded: the mask PNG + filename do NOT encode their space (only
state.json's crop_window does), so this is the single place the _pcrop->_sam remap
lives, and an aggregator must go through it.

Torch-free (cv2/numpy + pipeline):
    py -3 -m pytest tests/test_chain_masks_in_sam.py
    py -3 tests/test_chain_masks_in_sam.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

import pipeline


def _write_mask(masks_dir: pathlib.Path, z: int, arr_bool: np.ndarray) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(masks_dir / f"mask_{z:04d}.png"), (arr_bool.astype(np.uint8) * 255))


def _write_state(chain_dir: pathlib.Path, crop_window=None) -> None:
    d = {"neuron": "TOY", "chain_idx": 0, "frame_to_z": {"0": 1000},
         "crop_window": crop_window}
    (chain_dir / "state.json").write_text(json.dumps(d))


# ---------------------------------------------------------------------------
# legacy _sam chain: full-frame, no offset, mask returned as-is
# ---------------------------------------------------------------------------

def test_legacy_sam_chain_is_passthrough():
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d)
        _write_state(cd, crop_window=None)
        m = np.zeros((50, 60), dtype=bool)
        m[10:20, 15:25] = True
        _write_mask(cd / "masks", 1000, m)
        out = pipeline.chain_masks_in_sam(cd)
        assert set(out) == {1000}
        mask_sam, x0, y0 = out[1000]
        assert (x0, y0) == (0, 0)
        assert mask_sam.shape == (50, 60)
        assert np.array_equal(mask_sam, m)


# ---------------------------------------------------------------------------
# tier-2 _pcrop chain: downscaled to _sam, placed at origin_tif/sam_scale
# ---------------------------------------------------------------------------

def test_tier2_pcrop_chain_downscales_and_offsets():
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d)
        # window: 1024x512 tif at (800,400), crop_scale 2, sam_scale 8
        cw = {"origin_tif": [800.0, 400.0], "size_tif": [1024, 512],
              "crop_scale": 2, "sam_scale": 8}
        _write_state(cd, crop_window=cw)
        # a _pcrop mask is (size_tif / crop_scale) = (512 wide, 256 tall)
        pcrop = np.zeros((256, 512), dtype=bool)
        pcrop[:] = True                                   # fully filled -> easy to check
        _write_mask(cd / "masks", 1000, pcrop)

        out = pipeline.chain_masks_in_sam(cd)
        mask_sam, x0, y0 = out[1000]
        # _sam footprint = size_tif / sam_scale = (128 wide, 64 tall)
        assert mask_sam.shape == (64, 128)
        # offset = origin_tif / sam_scale = (100, 50)
        assert (x0, y0) == (100, 50)
        assert mask_sam.all()                             # fully filled survives downscale


def test_tier2_placement_lands_in_full_sam_frame():
    """Paste the returned local mask into a full _sam canvas at its offset and confirm
    it occupies exactly the window's _sam footprint (the union operation an aggregator does)."""
    with tempfile.TemporaryDirectory() as d:
        cd = pathlib.Path(d)
        cw = {"origin_tif": [800.0, 400.0], "size_tif": [1024, 512],
              "crop_scale": 2, "sam_scale": 8}
        _write_state(cd, crop_window=cw)
        pcrop = np.ones((256, 512), dtype=bool)
        _write_mask(cd / "masks", 1000, pcrop)

        mask_sam, x0, y0 = pipeline.chain_masks_in_sam(cd)[1000]
        canvas = np.zeros((300, 400), dtype=bool)         # a stand-in full _sam frame
        h, w = mask_sam.shape
        canvas[y0:y0 + h, x0:x0 + w] |= mask_sam
        ys, xs = np.where(canvas)
        # foreground bbox == the window's _sam footprint [x0..x0+128), [y0..y0+64)
        assert (xs.min(), xs.max() + 1) == (100, 228)
        assert (ys.min(), ys.max() + 1) == (50, 114)


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
