import numpy as np

from sam2_utils import sam3_backend as sb


def test_select_image_masks_multimask_keeps_all():
    masks = np.zeros((3, 4, 4), bool)
    scores = np.array([0.1, 0.9, 0.5])
    logits = np.zeros((3, 2, 2), float)
    m, s, lg = sb.select_image_masks(masks, scores, logits, multimask_output=True)
    assert m.shape == (3, 4, 4) and s.shape == (3,) and lg.shape == (3, 2, 2)


def test_select_image_masks_single_returns_one():
    masks = np.zeros((1, 4, 4), bool)
    scores = np.array([0.7])
    logits = np.zeros((1, 2, 2), float)
    m, s, lg = sb.select_image_masks(masks, scores, logits, multimask_output=False)
    assert m.shape == (1, 4, 4) and s.shape == (1,) and lg.shape == (1, 2, 2)
    assert m.dtype == bool


def test_video_logits_to_mask_thresholds_at_zero():
    lg = np.array([[-1.0, 0.5], [2.0, -0.1]])
    assert sb.video_logits_to_mask(lg).tolist() == [[False, True], [True, False]]
