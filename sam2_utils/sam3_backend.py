"""SAM3 (HuggingFace transformers) adapters presenting the SAM2 predictor interface.

torch and transformers are imported LAZILY inside methods so importing this module
stays CPU-only and does not violate the library import-direction rule.
"""
from __future__ import annotations

import numpy as np

DEFAULT_CHECKPOINT_DIR = r"F:\sam3\huggingface"


class Sam3ImagePredictor:
    """Wraps `Sam3TrackerModel` behind the `SAM2ImagePredictor` surface.

    Presents `set_image()` + `predict(...)` so `pipeline.predict.image_predict`
    can drive SAM3 unchanged. torch and transformers are imported lazily here,
    inside `__init__`, so importing this module stays CPU-only.
    """

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR, device=None):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3TrackerModel.from_pretrained(checkpoint_dir).to(self.device)
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint_dir)
        self._image = None
        self._hw = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._image = np.asarray(image_rgb)
        self._hw = (int(self._image.shape[0]), int(self._image.shape[1]))

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        torch = self._torch
        prompts = to_image_prompts(point_coords, point_labels, box)
        inputs = self.processor(images=self._image, return_tensors="pt", **prompts)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=multimask_output)
        H, W = self._hw
        full = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = np.asarray(full).reshape(-1, H, W)
        scores = out.iou_scores.detach().float().cpu().numpy().reshape(-1)
        low = out.pred_masks.detach().float().cpu().numpy()
        logits = low.reshape(-1, low.shape[-2], low.shape[-1])
        return select_image_masks(masks, scores, logits, multimask_output)


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
