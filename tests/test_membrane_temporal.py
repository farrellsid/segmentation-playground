import numpy as np
import pytest
from sam2_utils import membrane as mb


def test_register_crops_single_element_passthrough():
    crop = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = mb.register_crops([crop])
    assert len(out) == 1
    assert np.array_equal(out[0], crop)


def test_register_crops_aligns_to_center():
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    shifted = np.zeros((24, 24), dtype=np.float32)
    shifted[6:10, 13:17] = 200.0  # same block, moved relative to ref
    aligned_list = mb.register_crops([shifted, ref])  # center = index 1 (ref)
    aligned = aligned_list[0]
    ref_peak = np.unravel_index(np.argmax(ref), ref.shape)
    aligned_peak = np.unravel_index(np.argmax(aligned), aligned.shape)
    assert abs(ref_peak[0] - aligned_peak[0]) <= 1
    assert abs(ref_peak[1] - aligned_peak[1]) <= 1


def test_register_crops_explicit_center_overrides_lenhalf_default():
    # len(crops) // 2 for a 2-element list is 0. grow_all builds crops from whichever z's
    # actually loaded, so a degraded window (a failed frame load near the stack edge) can
    # leave a 2-element list whose true center is index 1, not 0. Passing center=1
    # explicitly must use crops[1] as the reference, not silently fall back to crops[0].
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    other = np.zeros((24, 24), dtype=np.float32)
    other[6:10, 6:10] = 200.0  # mild shift, within max_shift, from a different starting position
    aligned_list = mb.register_crops([other, ref], center=1)
    assert np.array_equal(aligned_list[1], ref)  # the true reference is never modified
    ref_peak = np.array(np.unravel_index(np.argmax(ref), ref.shape))
    aligned_peak = np.array(np.unravel_index(np.argmax(aligned_list[0]), aligned_list[0].shape))
    assert np.all(np.abs(ref_peak - aligned_peak) <= 1)


def test_register_crops_center_arg_fixes_lenhalf_bug():
    # Regression for the register_crops center-index bug (final whole-plan review, Finding 3).
    # For a 3-element list, len(crops) // 2 == 1, so the default treats crops[1] as the
    # reference. If the caller's true center is crops[2] (grow_all's zs.index(center_z) can
    # land anywhere once a window degrades near the stack edge), the default silently aligns
    # to the wrong crop, including "correcting" what should have been the untouched reference.
    true_ref = np.zeros((24, 24), dtype=np.float32)
    true_ref[10:14, 10:14] = 200.0  # crops[2], the real reference
    other_a = np.zeros((24, 24), dtype=np.float32)
    other_a[6:10, 13:17] = 200.0  # crops[0], a mild shift relative to true_ref
    other_b = np.zeros((24, 24), dtype=np.float32)
    other_b[16:20, 2:6] = 200.0  # crops[1], unrelated content: the len // 2 default's pick

    crops = [other_a, other_b, true_ref]

    # Bug scenario: the default center picks index 1, so crops[2] (the true reference) gets
    # "aligned" to crops[1] instead of being left alone.
    default_out = mb.register_crops(crops)
    assert not np.array_equal(default_out[2], true_ref)

    # Fix: an explicit center=2 keeps crops[2] as the untouched reference, and aligns the
    # other crops toward IT, not toward crops[1].
    fixed_out = mb.register_crops(crops, center=2)
    assert np.array_equal(fixed_out[2], true_ref)
    ref_peak = np.array(np.unravel_index(np.argmax(true_ref), true_ref.shape))
    aligned_peak = np.array(np.unravel_index(np.argmax(fixed_out[0]), fixed_out[0].shape))
    assert np.all(np.abs(ref_peak - aligned_peak) <= 1)


def test_register_crops_default_center_unchanged():
    # register_crops's default (center=None) must still behave exactly as before: the
    # existing aligns-to-center test above already exercises this without a center argument,
    # this test just makes the "default is unchanged" guarantee explicit and named.
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    shifted = np.zeros((24, 24), dtype=np.float32)
    shifted[6:10, 13:17] = 200.0
    crops = [shifted, ref]
    default_out = mb.register_crops(crops)
    explicit_out = mb.register_crops(crops, center=len(crops) // 2)
    for a, b in zip(default_out, explicit_out):
        assert np.array_equal(a, b)


def test_register_crops_clamps_large_shift():
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    shifted = np.zeros((24, 24), dtype=np.float32)
    shifted[0:4, 0:4] = 200.0  # far shift, larger than max_shift
    aligned_list = mb.register_crops([shifted, ref], max_shift=2)
    aligned = aligned_list[0]
    ref_peak = np.array(np.unravel_index(np.argmax(ref), ref.shape))
    aligned_peak = np.array(np.unravel_index(np.argmax(aligned), aligned.shape))
    assert np.any(np.abs(ref_peak - aligned_peak) > 2)  # clamp prevented full correction


def test_project_crops_median_suppresses_transient_dark():
    bright = np.full((6, 6), 200.0, dtype=np.float32)
    dark = bright.copy()
    dark[3, 3] = 10.0  # transient dark pixel, present in 1 of 3 crops
    crops = [bright.copy(), dark, bright.copy()]
    proj = mb.project_crops(crops, combine="median")
    assert proj[3, 3] > 150.0  # pulled back toward bright, not stuck dark


def test_project_crops_persistent_dark_stays_dark():
    dark_val = 10.0
    crops = [np.full((6, 6), dark_val, dtype=np.float32) for _ in range(3)]
    proj = mb.project_crops(crops, combine="median")
    assert np.allclose(proj, dark_val)


def test_project_crops_mean_partially_suppresses_less_than_median():
    bright = np.full((6, 6), 200.0, dtype=np.float32)
    dark = bright.copy()
    dark[3, 3] = 10.0
    crops = [bright.copy(), dark, bright.copy()]
    med = mb.project_crops(crops, combine="median")
    mean = mb.project_crops(crops, combine="mean")
    assert mean[3, 3] < med[3, 3]  # mean pulls less far back toward bright than median


def test_project_crops_single_element_identity():
    crop = np.arange(16, dtype=np.float32).reshape(4, 4)
    for combine in ("median", "mean", "max", "min"):
        assert np.array_equal(mb.project_crops([crop], combine=combine), crop)


def test_project_crops_unknown_combine_raises():
    with pytest.raises(ValueError):
        mb.project_crops([np.zeros((4, 4), np.float32)], combine="bogus")


def test_temporal_projection_then_membrane_map_suppresses_blob_response():
    # A flat background plus one straight ridge has almost no 2D texture, so phase
    # correlation locks onto the small blob difference instead of the (correct)
    # near-zero shift between slices that are otherwise identical. Real EM crops are
    # richly textured, so a shared background texture makes this synthetic case
    # representative instead of a degenerate worst case for phase correlation.
    rng = np.random.default_rng(0)
    base_texture = rng.uniform(180.0, 220.0, size=(30, 30)).astype(np.float32)

    def make_slice(with_blob):
        p = base_texture.copy()
        p[:, 14:16] = 20.0  # persistent vertical ridge (a membrane)
        if with_blob:
            p[20:24, 20:24] = 15.0  # transient dark blob (an organelle), center slice only
        return p

    crops = [make_slice(False), make_slice(True), make_slice(False)]
    registered = mb.register_crops(crops)
    projected = mb.project_crops(registered, combine="median")

    mem_single = mb.membrane_map(crops[1])  # single center slice, blob included
    mem_temporal = mb.membrane_map(projected)  # temporal-projected

    blob_region = np.s_[20:24, 20:24]
    ridge_region = np.s_[:, 14:16]
    assert mem_temporal[blob_region].mean() < mem_single[blob_region].mean()
    assert mem_temporal[ridge_region].mean() > 0.3
