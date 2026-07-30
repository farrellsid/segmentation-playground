import numpy as np
from sam2_utils import membrane as mb


def test_dark_round_blob_fully_detected():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    blob = (yy - 20) ** 2 + (xx - 20) ** 2 <= 3 ** 2
    patch[blob] = 20.0
    mask = mb.detect_organelle_blobs(patch)
    # every true blob pixel is covered (the dilated mask is a superset of the true blob)
    assert bool((blob & mask).sum() == blob.sum())
    # nothing flagged far from the blob
    assert not mask[0:5, 0:5].any()


def test_thin_ridge_not_detected():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    patch[:, 19:21] = 20.0  # a 2px-wide, full-height dark ridge
    mask = mb.detect_organelle_blobs(patch)
    assert mask.sum() == 0


def test_suppress_hard_edged_blob_reaches_background():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    blob = (yy - 20) ** 2 + (xx - 20) ** 2 <= 3 ** 2
    patch[blob] = 20.0
    mask = mb.detect_organelle_blobs(patch)
    cleaned = mb.suppress_organelles(patch, mask)
    assert abs(float(cleaned[20, 20]) - 200.0) < 1e-3


def test_suppress_soft_edged_blob_dilation_helps():
    yy, xx = np.ogrid[:40, :40]
    dist2 = (yy - 20) ** 2 + (xx - 20) ** 2
    soft_blob = (200.0 - 180.0 * np.exp(-dist2 / (2 * 3.0 ** 2))).astype(np.float32)
    mask_undilated = mb.detect_organelle_blobs(soft_blob, dilate_px=0)
    mask_dilated = mb.detect_organelle_blobs(soft_blob, dilate_px=2)
    cleaned_undilated = mb.suppress_organelles(soft_blob, mask_undilated)
    cleaned_dilated = mb.suppress_organelles(soft_blob, mask_dilated)
    # dilation gets meaningfully closer to the true background (200.0) than no dilation
    assert abs(float(cleaned_dilated[20, 20]) - 200.0) < abs(float(cleaned_undilated[20, 20]) - 200.0)


def test_suppress_organelles_empty_mask_is_noop():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    empty_mask = np.zeros((40, 40), dtype=bool)
    result = mb.suppress_organelles(patch, empty_mask)
    assert np.array_equal(result, patch)


def test_detect_organelle_blobs_uniform_patch_no_exception():
    patch = np.full((40, 40), 150.0, dtype=np.float32)
    mask = mb.detect_organelle_blobs(patch)
    assert mask.sum() == 0


def test_suppress_organelles_empty_mask_returns_float32():
    # the no-op (empty-mask) path used to pass em_patch through unchanged, so a
    # non-float32 input dtype leaked into the caller; it should now match the
    # non-empty path's dtype regardless of the input dtype.
    patch = np.full((40, 40), 150.0, dtype=np.uint8)
    empty_mask = np.zeros((40, 40), dtype=bool)
    result = mb.suppress_organelles(patch, empty_mask)
    assert result.dtype == np.float32
    assert np.array_equal(result, patch.astype(np.float32))


def test_suppress_organelles_3d_input_with_channel_axis():
    # detect_organelle_blobs always returns a 2D mask, even for 3D input (it
    # grayscales before detecting), so suppress_organelles must be able to take
    # a 3D em_patch with a 2D mask without inpaint_biharmonic's shape check
    # rejecting the pair.
    patch = np.full((40, 40, 3), 200.0, dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    blob = (yy - 20) ** 2 + (xx - 20) ** 2 <= 3 ** 2
    patch[blob] = 20.0
    mask = mb.detect_organelle_blobs(patch)
    assert mask.ndim == 2
    cleaned = mb.suppress_organelles(patch, mask)
    assert cleaned.shape == patch.shape
    assert cleaned.dtype == np.float32
    assert abs(float(cleaned[20, 20, 0]) - 200.0) < 1.0
