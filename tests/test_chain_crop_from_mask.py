"""Unit tests for the tier-2 ``chain_crop_from_mask`` sizing path ("Local high-res
cropping"): size the per-chain crop from the already-saved _sam
mask bbox (unioned with the skeleton bbox) so a cell whose membrane bulges past the
centerline nodes is no longer clipped at the window edge.

Covers the two pure helpers + the union plumbing:
  * pipeline.mask_union_box_px   : union foreground bbox over saved PNGs, queued-frame
                                   exclusion, the all-queued fallback, empty handling.
  * pipeline._prior_queued_z     : read the queued z from a prior qc.csv.
  * pipeline.chain_crop_window   : extra_box_tif unions with the skeleton bbox (strict
                                   superset; None reproduces skeleton-only sizing).

Torch-free (only cv2/pandas/numpy + pipeline, which defers torch to call-time):
    py -3 -m pytest tests/test_chain_crop_from_mask.py
    py -3 tests/test_chain_crop_from_mask.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

import pipeline
from pipeline import PipelineConfig


def _write_mask(masks_dir: pathlib.Path, z: int, box_xyxy, hw=(64, 64)) -> None:
    """Write a 0/255 PNG named like pipeline.save_masks with a filled box of foreground."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    m = np.zeros(hw, dtype=np.uint8)
    x0, y0, x1, y1 = box_xyxy
    m[y0:y1 + 1, x0:x1 + 1] = 255
    cv2.imwrite(str(masks_dir / f"mask_{z:04d}.png"), m)


# ---------------------------------------------------------------------------
# mask_union_box_px
# ---------------------------------------------------------------------------

def test_union_box_spans_all_frames():
    with tempfile.TemporaryDirectory() as d:
        md = pathlib.Path(d) / "masks"
        _write_mask(md, 1000, (10, 12, 20, 22))
        _write_mask(md, 1001, (30, 5, 40, 18))
        box = pipeline.mask_union_box_px(md)
        # union: x 10..40, y 5..22
        assert box == (10.0, 5.0, 40.0, 22.0)


def test_union_box_excludes_queued_frames():
    with tempfile.TemporaryDirectory() as d:
        md = pathlib.Path(d) / "masks"
        _write_mask(md, 1000, (10, 10, 20, 20))     # trustworthy
        _write_mask(md, 1001, (50, 50, 60, 60))     # queued (drifted) -> should be excluded
        box = pipeline.mask_union_box_px(md, exclude_z={1001})
        assert box == (10.0, 10.0, 20.0, 20.0)


def test_union_box_all_queued_falls_back_to_all():
    # if EVERY frame is queued, excluding empties the set -> retry over all frames
    with tempfile.TemporaryDirectory() as d:
        md = pathlib.Path(d) / "masks"
        _write_mask(md, 1000, (10, 10, 20, 20))
        _write_mask(md, 1001, (30, 30, 40, 40))
        box = pipeline.mask_union_box_px(md, exclude_z={1000, 1001})
        assert box == (10.0, 10.0, 40.0, 40.0)


def test_union_box_none_when_empty():
    with tempfile.TemporaryDirectory() as d:
        md = pathlib.Path(d) / "masks"
        md.mkdir(parents=True)
        _write_mask(md, 1000, (0, 0, -1, -1))        # all-zero mask (empty box)
        assert pipeline.mask_union_box_px(md) is None


# ---------------------------------------------------------------------------
# _prior_queued_z
# ---------------------------------------------------------------------------

def test_prior_queued_z_reads_queue_column():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "qc.csv"
        pd.DataFrame({"z": [1000, 1001, 1002],
                      "queue": [False, True, True]}).to_csv(p, index=False)
        assert pipeline._prior_queued_z(p) == {1001, 1002}


def test_prior_queued_z_missing_file_is_empty():
    assert pipeline._prior_queued_z(pathlib.Path("does_not_exist.csv")) == set()


# ---------------------------------------------------------------------------
# chain_crop_window union
# ---------------------------------------------------------------------------

def _toy_chain_and_df():
    """A 2-node chain whose skeleton bbox is small and centered, with a big frame so
    nothing clips. Nodes at _tif (1000,1000) and (1100,1100)."""
    df = pd.DataFrame({
        "node_id": ["a", "b"],
        "z": [1000, 1001],
        "x_tif": [1000.0, 1100.0],
        "y_tif": [1000.0, 1100.0],
    })
    chain = {"cell_name": "TOY", "nodes": ["a", "b"]}
    return chain, df


def test_extra_box_grows_window_superset():
    chain, df = _toy_chain_and_df()
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop=True, chain_crop_min_tif=0,  # disable floor to isolate the union
                         chain_crop_pad_tif=0, chain_crop_max_px=100000)  # no cap, no pad
    hw = (4000, 4000)
    base = pipeline.chain_crop_window(chain, df, cfg=cfg, image_hw_tif=hw)
    # a mask bbox extending well to the right/below the skeleton bbox
    extra = (900.0, 900.0, 1500.0, 1400.0)
    grown = pipeline.chain_crop_window(chain, df, cfg=cfg, image_hw_tif=hw,
                                       extra_box_tif=extra)
    bx0, by0 = base.origin_tif
    gx0, gy0 = grown.origin_tif
    # the grown window starts at/above-left of the base and is strictly larger
    assert gx0 <= bx0 and gy0 <= by0
    assert grown.size_tif[0] >= base.size_tif[0]
    assert grown.size_tif[1] >= base.size_tif[1]
    assert grown.size_tif[0] * grown.size_tif[1] > base.size_tif[0] * base.size_tif[1]
    # and it actually CONTAINS the extra (mask) box: extra corners map inside [0, size]
    c0 = grown.tif_to_crop([extra[0], extra[1]])
    c1 = grown.tif_to_crop([extra[2], extra[3]])
    cw_w, cw_h = grown.crop_hw[1], grown.crop_hw[0]
    assert -1 <= c0[0] and -1 <= c0[1]
    assert c1[0] <= cw_w + 1 and c1[1] <= cw_h + 1


def test_none_extra_box_reproduces_skeleton_sizing():
    chain, df = _toy_chain_and_df()
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8, chain_crop=True)
    hw = (4000, 4000)
    a = pipeline.chain_crop_window(chain, df, cfg=cfg, image_hw_tif=hw)
    b = pipeline.chain_crop_window(chain, df, cfg=cfg, image_hw_tif=hw, extra_box_tif=None)
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# node_crop_window (the collapse fallback)
# ---------------------------------------------------------------------------

def test_node_crop_window_centered_full_size():
    cw = pipeline.node_crop_window((2000.0, 2000.0), size_tif=1024, image_hw_tif=(4000, 4000),
                                   crop_scale=2, max_px=1536, sam_scale=8)
    assert cw.size_tif == (1024, 1024)
    assert cw.origin_tif == (1488.0, 1488.0)        # 2000 - 512, centred on the node
    assert cw.crop_scale == 2                         # 1024/1536 < 1, no bump
    assert cw.sam_scale == 8


def test_node_crop_window_slides_inside_at_edge():
    # a node near the corner keeps the full size and slides inside the frame
    cw = pipeline.node_crop_window((100.0, 100.0), size_tif=1024, image_hw_tif=(4000, 4000),
                                   crop_scale=2, max_px=1536, sam_scale=8)
    assert cw.size_tif == (1024, 1024)
    assert cw.origin_tif == (0.0, 0.0)


def test_node_crop_window_bumps_scale_when_over_max_px():
    # size 4000 _tif / max 1536 input -> crop_scale ceil(2.6) = 3
    cw = pipeline.node_crop_window((2500.0, 2500.0), size_tif=4000, image_hw_tif=(6000, 6000),
                                   crop_scale=2, max_px=1536, sam_scale=8)
    assert cw.crop_scale == 3
    assert cw.size_tif == (4000, 4000)


def test_node_crop_window_clamps_to_small_frame():
    cw = pipeline.node_crop_window((250.0, 250.0), size_tif=1024, image_hw_tif=(500, 500),
                                   crop_scale=2, max_px=1536, sam_scale=8)
    assert cw.size_tif == (500, 500)                  # window can't exceed the frame


# ---------------------------------------------------------------------------
# grow_crop_window (the GUI recrop Phase-1 lever)
# ---------------------------------------------------------------------------

def _cw_at(origin=(2000.0, 2000.0), size=(1000, 800), crop_scale=2, sam_scale=8):
    from sam2_utils.alignment import CropWindow
    return CropWindow(origin_tif=origin, size_tif=size, crop_scale=crop_scale, sam_scale=sam_scale)


def test_grow_crop_window_widens_each_side():
    cw = pipeline.grow_crop_window(_cw_at(), grow_tif=500, image_hw_tif=(8000, 8000),
                                   max_px=100000)
    assert cw.origin_tif == (1500.0, 1500.0)          # 2000 - 500 each axis
    assert cw.size_tif == (2000, 1800)                # +1000 each dim
    assert cw.crop_scale == 2 and cw.sam_scale == 8   # scale preserved (no cap hit)


def test_grow_crop_window_clips_to_frame():
    # grow near the top-left corner: origin floors at 0
    cw = pipeline.grow_crop_window(_cw_at(origin=(100.0, 100.0)), grow_tif=500,
                                   image_hw_tif=(2000, 2000), max_px=100000)
    assert cw.origin_tif == (0.0, 0.0)


def test_grow_crop_window_bumps_scale_over_max_px():
    cw = pipeline.grow_crop_window(_cw_at(), grow_tif=2000, image_hw_tif=(20000, 20000),
                                   max_px=1536)
    # longest grown dim = 1000 + 2*2000 = 5000 _tif; 5000/1536 -> ceil 4
    assert cw.crop_scale == 4


# ---------------------------------------------------------------------------
# window_from_sam_box (the GUI recrop picker, Phase 2)
# ---------------------------------------------------------------------------

def test_window_from_sam_box_scales_sam_to_tif():
    # a box in _sam px at sam_scale 8 -> _tif window 8x larger
    cw = pipeline.window_from_sam_box((100.0, 50.0, 300.0, 250.0), sam_scale=8,
                                      image_hw_tif=(8000, 8000), crop_scale=2, max_px=100000)
    assert cw.origin_tif == (800.0, 400.0)            # 100*8, 50*8
    assert cw.size_tif == (1600, 1600)                # (300-100)*8, (250-50)*8
    assert cw.crop_scale == 2 and cw.sam_scale == 8


def test_window_from_sam_box_bumps_scale_and_clips():
    # a wide box near the edge: clipped to the frame, crop_scale bumped under max_px
    cw = pipeline.window_from_sam_box((0.0, 0.0, 400.0, 100.0), sam_scale=8,
                                      image_hw_tif=(1000, 2000), crop_scale=2, max_px=1536)
    assert cw.origin_tif == (0.0, 0.0)
    # box _tif = 3200 x 800; width clips to frame W=2000; longest 3200/1536 -> ceil 3
    assert cw.size_tif[0] == 2000
    assert cw.crop_scale == 3


# ---------------------------------------------------------------------------
# tier-2 fallback diagnostics survive serialization (the observability fix)
# ---------------------------------------------------------------------------

def test_fallback_diagnostics_round_trip():
    """The crop-pass reason/score/gate captured before the _sam recovery overwrites them
    must persist to state.json (else the failing crop-pass values are lost)."""
    st = pipeline.ChainState(neuron="TOY", chain_idx=0)
    st.fell_back_to_sam = True
    st.fellback_reason = "score<0.7"
    st.crop_image_score = 0.516
    st.crop_anchor_score = {"passed": True, "reasons": [], "area_frac": 0.01,
                            "largest_cc_frac": 0.99, "contained": True, "n_components": 1}
    back = pipeline.state_from_dict(pipeline.state_to_dict(st))
    assert back.fell_back_to_sam is True
    assert back.fellback_reason == "score<0.7"
    assert abs(back.crop_image_score - 0.516) < 1e-9
    assert back.crop_anchor_score["area_frac"] == 0.01


def test_fallback_diagnostics_default_none():
    # a chain that did NOT fall back leaves the diagnostics None
    st = pipeline.ChainState(neuron="TOY", chain_idx=0)
    back = pipeline.state_from_dict(pipeline.state_to_dict(st))
    assert back.fell_back_to_sam is False
    assert back.fellback_reason is None
    assert back.crop_image_score is None
    assert back.crop_anchor_score is None


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
