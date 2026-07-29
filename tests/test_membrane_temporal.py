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
    def make_slice(with_blob):
        p = np.full((30, 30), 200.0, dtype=np.float32)
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
