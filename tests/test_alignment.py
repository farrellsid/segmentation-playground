"""Unit tests for sam2_utils.alignment - the one home for coordinate transforms.

These are deliberately torch-free and data-free: they exercise the pure
coordinate math (affine, _tif<->_sam, z section maps, nm->stack-px, CropWindow)
and nothing that needs a GPU or the EM stack. That makes them the runnable
guard for the "centralize transforms" refactor: if the maps still round-trip
here, the inline call sites in pipeline.py / qc.py / catmaid.py that now delegate
to them behave as before.

Run either way:
    py -3 -m pytest tests/test_alignment.py        # if pytest is installed
    py -3 tests/test_alignment.py                  # plain runner, no pytest needed
"""

from __future__ import annotations

import pathlib
import sys

# Make `from sam2_utils import ...` work no matter how this file is invoked
# (pytest from the repo root, or `py -3 tests/test_alignment.py` from anywhere).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from sam2_utils import alignment, config


# ---------------------------------------------------------------------------
# apply_affine / catmaid_to_tif
# ---------------------------------------------------------------------------

def test_apply_affine_single_and_batch_shapes():
    M = np.array([[2.0, 0.0], [0.0, 3.0]])
    t = np.array([10.0, -5.0])
    # single (2,) point -> (2,)
    out = alignment.apply_affine([1.0, 1.0], M, t)
    assert out.shape == (2,)
    assert np.allclose(out, [12.0, -2.0])
    # batch (N, 2) -> (N, 2)
    out = alignment.apply_affine([[1.0, 1.0], [0.0, 0.0]], M, t)
    assert out.shape == (2, 2)
    assert np.allclose(out, [[12.0, -2.0], [10.0, -5.0]])


def test_catmaid_to_tif_matches_stored_affine():
    # catmaid_to_tif must equal apply_affine with the config affine, for both
    # scalar and array input (this is what build_session relies on).
    x, y = 5000.0, 4000.0
    expected = alignment.apply_affine([x, y], config.M_AFFINE, config.T_AFFINE)
    got = alignment.catmaid_to_tif(x, y)
    assert got.shape == (2,)
    assert np.allclose(got, expected)

    xs = np.array([5000.0, 1234.0])
    ys = np.array([4000.0, 8765.0])
    got = alignment.catmaid_to_tif(xs, ys)
    assert got.shape == (2, 2)
    expected = alignment.apply_affine(np.column_stack([xs, ys]),
                                      config.M_AFFINE, config.T_AFFINE)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# tif_to_sam / sam_to_tif
# ---------------------------------------------------------------------------

def test_tif_sam_known_values_and_roundtrip():
    # tif / scale = sam; sam * scale = tif.
    assert np.allclose(alignment.tif_to_sam([800.0, 80.0], 8), [100.0, 10.0])
    assert np.allclose(alignment.sam_to_tif([100.0, 10.0], 8), [800.0, 80.0])
    # round-trip is identity for any scale
    pts = np.array([[8.0, 16.0], [24.0, 32.0], [0.0, 7.0]])
    assert np.allclose(alignment.sam_to_tif(alignment.tif_to_sam(pts, 8), 8), pts)


def test_tif_to_sam_preserves_shape():
    # single (2,) point stays (2,); (N, 2) stays (N, 2). The call sites depend on
    # both (build_prompts passes a (1, 2); qc passes a single (2,)).
    assert alignment.tif_to_sam([16.0, 16.0], 8).shape == (2,)
    assert alignment.tif_to_sam([[8.0, 16.0], [24.0, 32.0]], 8).shape == (2, 2)
    assert np.allclose(alignment.tif_to_sam([[8.0, 16.0], [24.0, 32.0]], 8),
                       [[1.0, 2.0], [3.0, 4.0]])


def test_tif_to_sam_with_save_downscale_equals_mask_space():
    # qc divides the skeleton by save_downscale to reach mask px. Under the
    # canonical save_downscale == scale rule this is the same as _sam.
    scale = save_downscale = 8
    xy = [800.0, 80.0]
    assert np.allclose(alignment.tif_to_sam(xy, save_downscale),
                       alignment.tif_to_sam(xy, scale))


# ---------------------------------------------------------------------------
# catmaid_z <-> file_z
# ---------------------------------------------------------------------------

def test_z_maps_known_value_and_roundtrip():
    # config says CATMAID_z = file_z + FILE_Z_OFFSET, and file "z1300" == catmaid 1293.
    assert alignment.catmaid_z_to_file_z(1293) == 1293 - config.FILE_Z_OFFSET
    assert alignment.file_z_to_catmaid_z(1300) == 1300 + config.FILE_Z_OFFSET
    with_offset = config.FILE_Z_OFFSET
    if with_offset == -7:                       # the fitted value; guard the headline case
        assert alignment.catmaid_z_to_file_z(1293) == 1300
        assert alignment.file_z_to_catmaid_z(1300) == 1293
    # round-trip both directions
    for z in (0, 100, 1293, 5000):
        assert alignment.file_z_to_catmaid_z(alignment.catmaid_z_to_file_z(z)) == z
        assert alignment.catmaid_z_to_file_z(alignment.file_z_to_catmaid_z(z)) == z


# ---------------------------------------------------------------------------
# nm -> stack-pixel
# ---------------------------------------------------------------------------

def test_nm_to_stack_px_scalar_and_array():
    rx, ry, rz = config.STACK_RESOLUTION_NM
    x, y, z = alignment.nm_to_stack_px(2.0 * rx, 4.0 * ry, 3.0 * rz)
    assert np.allclose([x, y, z], [2.0, 4.0, 3.0])
    # arrays, per-axis divide
    xs = alignment.nm_to_stack_px(np.array([rx, 2 * rx]),
                                  np.array([ry, 3 * ry]),
                                  np.array([rz, 4 * rz]))
    assert np.allclose(xs[0], [1.0, 2.0])
    assert np.allclose(xs[1], [1.0, 3.0])
    assert np.allclose(xs[2], [1.0, 4.0])


# ---------------------------------------------------------------------------
# CropWindow - the _crop <-> _tif <-> _sam home
# ---------------------------------------------------------------------------

def _cw():
    # node at (1000, 800) in a 5000x6000 (HxW) frame, 400px window, crop_scale 2,
    # sam_scale 8. Window fits with room to spare, so it centers exactly.
    return alignment.CropWindow.around_node(
        (1000.0, 800.0), size_tif=400, image_hw_tif=(5000, 6000),
        crop_scale=2, sam_scale=8)


def test_cropwindow_centers_and_geometry():
    cw = _cw()
    assert cw.origin_tif == (800.0, 600.0)          # node - size/2
    assert cw.size_tif == (400, 400)
    assert cw.crop_hw == (200, 200)                 # size / crop_scale


def test_cropwindow_slice_is_row_col():
    # slice_tif is the ONLY x/y swap: rows = y, cols = x.
    cw = _cw()
    rs, cs = cw.slice_tif()
    assert (rs.start, rs.stop) == (600, 1000)       # y
    assert (cs.start, cs.stop) == (800, 1200)       # x


def test_cropwindow_point_roundtrip_and_center():
    cw = _cw()
    # the node sits at the crop center
    assert np.allclose(cw.tif_to_crop((1000.0, 800.0)), [100.0, 100.0])
    # tif <-> crop round-trip
    pts_tif = np.array([[1000.0, 800.0], [820.0, 640.0]])
    assert np.allclose(cw.crop_to_tif(cw.tif_to_crop(pts_tif)), pts_tif)
    # crop -> sam goes via tif then / sam_scale
    assert np.allclose(cw.crop_to_sam((100.0, 100.0)), [1000.0 / 8, 800.0 / 8])


def test_cropwindow_box_crop_to_sam():
    cw = _cw()
    box_crop = [90.0, 90.0, 110.0, 110.0]
    # corners: crop->tif then /8
    expected = np.array([(90 * 2 + 800) / 8, (90 * 2 + 600) / 8,
                         (110 * 2 + 800) / 8, (110 * 2 + 600) / 8])
    assert np.allclose(cw.box_crop_to_sam(box_crop), expected)


def test_cropwindow_clips_at_edges():
    # node in the top-left corner: the window slides fully inside, origin clamps to 0.
    cw = alignment.CropWindow.around_node(
        (10.0, 10.0), size_tif=400, image_hw_tif=(5000, 6000),
        crop_scale=2, sam_scale=8)
    assert cw.origin_tif == (0.0, 0.0)
    assert cw.size_tif == (400, 400)
    # node is no longer centered (it's near the corner of the crop)
    assert np.allclose(cw.tif_to_crop((10.0, 10.0)), [5.0, 5.0])


def test_cropwindow_window_cannot_exceed_image():
    # an over-large request is clamped to the image extent.
    cw = alignment.CropWindow.around_node(
        (3000.0, 2500.0), size_tif=10000, image_hw_tif=(5000, 6000),
        crop_scale=1, sam_scale=8)
    assert cw.size_tif == (6000, 5000)              # (w, h) = (W, H)
    assert cw.origin_tif == (0.0, 0.0)


# ---------------------------------------------------------------------------
# CropWindow.around_box + sam_to_crop + (de)serialize - the tier-2 per-chain crop
# ---------------------------------------------------------------------------

def test_cropwindow_around_box_pad_and_intersection():
    # bbox (1000,2000)-(1400,2600) + 50 pad, clipped to a big frame -> exact extent.
    cw = alignment.CropWindow.around_box(
        (1000.0, 2000.0, 1400.0, 2600.0), pad_tif=50,
        image_hw_tif=(9230, 9216), crop_scale=2, sam_scale=8)
    assert cw.origin_tif == (950.0, 1950.0)
    assert cw.size_tif == (500, 700)                # (1450-950, 2650-1950)
    assert cw.crop_hw == (350, 250)                 # (h, w) / crop_scale


def test_cropwindow_around_box_clips_at_corner():
    # a box hugging the top-left: padded origin floors below 0 -> clamps to 0.
    cw = alignment.CropWindow.around_box(
        (10.0, 5.0, 100.0, 90.0), pad_tif=64,
        image_hw_tif=(9230, 9216), crop_scale=1, sam_scale=8)
    assert cw.origin_tif == (0.0, 0.0)


def test_cropwindow_sam_to_crop_matches_tif_path():
    # sam_to_crop == tif_to_crop(xy_sam * sam_scale): the tier-2 prompt/skeleton map.
    cw = alignment.CropWindow.around_box(
        (1000.0, 2000.0, 1400.0, 2600.0), pad_tif=50,
        image_hw_tif=(9230, 9216), crop_scale=2, sam_scale=8)
    p_tif = np.array([1200.0, 2300.0])
    assert np.allclose(cw.sam_to_crop(p_tif / 8.0), cw.tif_to_crop(p_tif))


def test_cropwindow_dict_roundtrip():
    cw = alignment.CropWindow.around_box(
        (1000.0, 2000.0, 1400.0, 2600.0), pad_tif=50,
        image_hw_tif=(9230, 9216), crop_scale=2, sam_scale=8)
    cw2 = alignment.CropWindow.from_dict(cw.to_dict())
    assert (cw2.origin_tif, cw2.size_tif, cw2.crop_scale, cw2.sam_scale) == \
           (cw.origin_tif, cw.size_tif, cw.crop_scale, cw.sam_scale)


# ---------------------------------------------------------------------------
# Plain runner (no pytest required)
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                      # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
