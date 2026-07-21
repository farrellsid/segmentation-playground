"""SAM3 (HuggingFace transformers) adapters presenting the SAM2 predictor interface.

torch and transformers are imported LAZILY inside methods so importing this module
stays CPU-only and does not violate the library import-direction rule.
"""
from __future__ import annotations

import numpy as np


def select_image_masks(masks, scores, low_res_logits, multimask_output):
    """Select and reshape image masks and logits for SAM2 compatibility.

    Args:
        masks: array of shape (num_masks, H, W) or (H, W)
        scores: array of shape (num_masks,)
        low_res_logits: array of shape (num_masks, h, w) or (h, w)
        multimask_output: bool, whether to keep all masks

    Returns:
        tuple of (masks, scores, logits) with shapes:
        - masks: (num_masks, H, W) as bool
        - scores: (num_masks,) as float
        - logits: (num_masks, h, w) as float
    """
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float).ravel()
    logits = np.asarray(low_res_logits, dtype=float)
    if masks.ndim == 2:
        masks = masks[None]
    if logits.ndim == 2:
        logits = logits[None]
    return masks, scores, logits


def video_logits_to_mask(logit_hw):
    """Convert video logits to binary mask using zero threshold.

    Args:
        logit_hw: array of shape (H, W) as float

    Returns:
        array of shape (H, W) as bool, True where logit_hw > 0.0
    """
    return np.asarray(logit_hw, dtype=float) > 0.0


def to_image_prompts(point_coords, point_labels, box):
    """Shape pipeline prompt arrays into SAM3 image-processor kwargs (one image, one object)."""
    out: dict = {}
    if point_coords is not None and len(point_coords):
        pts = np.asarray(point_coords, dtype=float).tolist()
        labs = np.asarray(point_labels, dtype=int).tolist()
        out["input_points"] = [[pts]]
        out["input_labels"] = [[labs]]
    if box is not None:
        b = np.asarray(box, dtype=float).ravel().tolist()
        out["input_boxes"] = [[b]]
    return out


def to_video_prompt(frame_idx, obj_id, box, points, labels):
    """Shape pipeline prompt into SAM3 video-session add-input kwargs."""
    out: dict = {"frame_idx": frame_idx, "obj_ids": obj_id}
    if points is not None and len(points):
        pts = np.asarray(points, dtype=float).tolist()
        labs = np.asarray(labels, dtype=int).tolist()
        out["input_points"] = [[pts]]
        out["input_labels"] = [[labs]]
    if box is not None:
        b = np.asarray(box, dtype=float).ravel().tolist()
        out["input_boxes"] = [[b]]
    return out
