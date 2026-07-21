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


def test_to_image_prompts_points_only():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    labs = np.array([1, 0])
    out = sb.to_image_prompts(pts, labs, None)
    assert out["input_points"] == [[[[10.0, 20.0], [30.0, 40.0]]]]
    assert out["input_labels"] == [[[1, 0]]]
    assert "input_boxes" not in out


def test_to_image_prompts_box_only():
    box = np.array([1.0, 2.0, 3.0, 4.0])
    out = sb.to_image_prompts(None, None, box)
    assert out["input_boxes"] == [[[1.0, 2.0, 3.0, 4.0]]]
    assert "input_points" not in out


def test_to_image_prompts_points_and_box():
    out = sb.to_image_prompts(np.array([[5.0, 6.0]]), np.array([1]), np.array([0.0, 0.0, 9.0, 9.0]))
    assert out["input_points"] == [[[[5.0, 6.0]]]]
    assert out["input_labels"] == [[[1]]]
    assert out["input_boxes"] == [[[0.0, 0.0, 9.0, 9.0]]]


def test_to_video_prompt_box():
    out = sb.to_video_prompt(4, 1, np.array([1.0, 2.0, 3.0, 4.0]), None, None)
    assert out["frame_idx"] == 4
    assert out["obj_ids"] == 1
    assert out["input_boxes"] == [[[1.0, 2.0, 3.0, 4.0]]]


def test_to_video_prompt_points():
    out = sb.to_video_prompt(0, 2, None, np.array([[7.0, 8.0]]), np.array([1]))
    assert out["frame_idx"] == 0
    assert out["obj_ids"] == 2
    assert out["input_points"] == [[[[7.0, 8.0]]]]
    assert out["input_labels"] == [[[1]]]
