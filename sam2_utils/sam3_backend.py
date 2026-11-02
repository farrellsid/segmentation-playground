"""SAM3 (HuggingFace transformers) adapters presenting the SAM2 predictor interface.

torch and transformers are imported LAZILY inside methods so importing this module
stays CPU-only and does not violate the library import-direction rule.
"""
from __future__ import annotations

from pathlib import Path

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

    def reset_predictor(self) -> None:
        """Drop the cached image. Parity with SAM2ImagePredictor.reset_predictor, which
        pipeline.orchestrator.run_chain calls between chains to release the set image. The
        HF processor takes the image fresh on every predict() call, so there is no encoder
        state to clear beyond our own cached frame."""
        self._image = None
        self._hw = None

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
        multimask_output: accepted for call-site parity with the model call; candidate
            selection happens upstream in pipeline._select_anchor_mask, so it is inert here

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


class Sam3VideoPredictor:
    """Wraps `Sam3TrackerVideoModel` behind the SAM2 video-predictor surface.

    Presents `init_state` / `reset_state` / `add_new_points_or_box` / `add_new_mask` /
    `propagate_in_video` so `pipeline.propagate.PropagationSession` (and the headless
    `propagate(...)` driver) run SAM3 unchanged. torch and transformers are imported
    LAZILY here, inside `__init__`, so importing this module stays CPU-only (see
    `tests/test_import_direction.py`).

    Anchor tracking (why it's here, not optional): unlike SAM2, SAM3 does NOT
    auto-infer the propagation start frame; calling `propagate_in_video_iterator`
    with no `start_frame_idx` raises "Cannot determine the starting frame index."
    `PropagationSession.run_bidirectional` calls `propagate(reverse=False)` then
    `propagate(reverse=True)` with no `start_frame_idx` (that's SAM2's own
    auto-start behaviour), so this adapter records the last-seeded frame on the
    inference_state itself (`_anchor_frame_idx`) and defaults `start_frame_idx` to
    it whenever the caller doesn't supply one. See
    docs/explanation/sam3-bakeoff-findings.md, "Reverse propagation verdict".
    """

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR, device=None):
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vmodel = Sam3TrackerVideoModel.from_pretrained(checkpoint_dir).to(
            self.device, dtype=torch.bfloat16)
        self.vproc = Sam3TrackerVideoProcessor.from_pretrained(checkpoint_dir)

    def init_state(self, video_path: str, offload_video_to_cpu: bool = True, **kw):
        """Load the sorted `{idx:05d}.jpg` frames from `video_path` and start a
        SAM3 video inference session. Returns the session; this IS the
        "inference_state" object the pipeline threads through every other method."""
        import cv2

        torch = self._torch
        frame_paths = sorted(Path(video_path).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"no {{idx:05d}}.jpg frames found in {video_path}")
        frames = []
        for p in frame_paths:
            raw = cv2.imread(str(p))
            if raw is None:
                raise FileNotFoundError(f"frame not found or unreadable: {p}")
            frames.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        H, W = frames[0].shape[:2]

        session = self.vproc.init_video_session(
            video=frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
            video_storage_device=("cpu" if offload_video_to_cpu else None),
            processing_device=("cpu" if offload_video_to_cpu else None),
        )
        session._frame_hw = (H, W)          # stashed for post_process_masks' original_sizes
        session._anchor_frame_idx = None    # set on the first add_new_points_or_box/add_new_mask
        return session

    def reset_state(self, inference_state) -> None:
        inference_state.reset_inference_session()

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None,
                               points=None, labels=None, clear_old_points=False) -> None:
        # SAM3 quirk (empirical, hit during the video-adapter smoke, not covered by
        # sam3-bakeoff-findings.md): add_inputs_to_inference_session raises "cannot
        # add box without clearing old points" whenever a box is supplied with
        # clear_old_inputs=False, since a box must be the first prompt on a frame.
        # SAM2 has no such restriction, so PropagationSession.seed (which never
        # passes clear_old_points) works unmodified against SAM2 but trips this on
        # SAM3. The pipeline's only box add is the once-per-chain anchor seed, which
        # has nothing on that frame to preserve, so promoting to clear_old_inputs=True
        # whenever a box is present is safe and keeps point-only corrections (the
        # common add_points path) on the caller's own clear_old_points value.
        clear = True if box is not None else clear_old_points
        self.vproc.add_inputs_to_inference_session(
            inference_session=inference_state,
            clear_old_inputs=clear,
            **to_video_prompt(frame_idx, obj_id, box, points, labels),
        )
        inference_state._anchor_frame_idx = int(frame_idx)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask) -> None:
        self.vproc.add_inputs_to_inference_session(
            inference_session=inference_state,
            frame_idx=int(frame_idx),
            obj_ids=obj_id,
            input_masks=np.asarray(mask),
        )
        inference_state._anchor_frame_idx = int(frame_idx)

    def propagate_in_video(self, inference_state, reverse=False, start_frame_idx=None,
                            max_frame_num_to_track=None):
        """Generator yielding `(frame_idx, obj_ids, mask_logits)` per propagated frame,
        matching SAM2's `propagate_in_video` shape so `PropagationSession._collect` can
        index `mask_logits[i].cpu().numpy()` unchanged. Defaults `start_frame_idx` to the
        anchor frame recorded by add_new_points_or_box/add_new_mask (see class docstring):
        SAM2 auto-infers the start frame, SAM3 does not."""
        torch = self._torch
        sfi = (start_frame_idx if start_frame_idx is not None
               else getattr(inference_state, "_anchor_frame_idx", None))
        maxkw = ({} if max_frame_num_to_track is None
                 else {"max_frame_num_to_track": int(max_frame_num_to_track)})
        H, W = inference_state._frame_hw

        for out in self.vmodel.propagate_in_video_iterator(
                inference_state, start_frame_idx=sfi, reverse=reverse, **maxkw):
            pp = self.vproc.post_process_masks(
                [out.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
            # .float(): the model runs in bfloat16 and post_process_masks preserves that
            # dtype, but numpy has no bfloat16, so PropagationSession._collect's
            # `mask_logits[i].cpu().numpy()` raises "Got unsupported ScalarType BFloat16"
            # (hit empirically in the video-adapter smoke) unless cast to float32 first.
            pp = torch.as_tensor(pp).float()             # frame-res logits, (num_objs, num_masks, H, W)
            obj_ids = [int(o) for o in out.object_ids]
            # reshape(-1, H, W)[0] takes the single tracking mask per object robustly,
            # whether post_process returns (num_objs, H, W) or (num_objs, 1, H, W); a
            # bare reshape(H, W) would raise cryptically if a multimask ever slipped through.
            mask_logits = [pp[i].reshape(-1, H, W)[0] for i in range(len(obj_ids))]
            yield int(out.frame_idx), obj_ids, mask_logits


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
