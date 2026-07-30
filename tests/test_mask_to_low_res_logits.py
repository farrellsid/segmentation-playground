"""Unit tests for pipeline.predict.mask_to_low_res_logits, the helper that lets a
saved boolean mask stand in for SAM2's mask_input (a low-res logits array SAM2
normally gets from a previous prediction iteration, not a binary mask)."""
import cv2
import numpy as np

from pipeline.predict import mask_to_low_res_logits


def test_output_shape_and_dtype():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:60, 10:40] = True
    out = mask_to_low_res_logits(mask)
    assert out.shape == (1, 256, 256)
    assert out.dtype == np.float32


def test_threshold_at_zero_matches_resized_mask():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:60, 10:40] = True
    out = mask_to_low_res_logits(mask)
    recovered = out[0] > 0.0
    expected = cv2.resize(mask.astype(np.uint8), (256, 256),
                          interpolation=cv2.INTER_NEAREST) > 0
    assert np.array_equal(recovered, expected)


def test_all_false_mask_is_all_negative():
    mask = np.zeros((50, 50), dtype=bool)
    out = mask_to_low_res_logits(mask)
    assert (out < 0).all()


def test_all_true_mask_is_all_positive():
    mask = np.ones((50, 50), dtype=bool)
    out = mask_to_low_res_logits(mask)
    assert (out > 0).all()
