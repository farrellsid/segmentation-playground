"""Interruptible SAM2 video propagation: FrameResult, the IoU hook, PropagationSession,
propagate, and segment_per_slice (the per-frame node-anchored image-mode alternative)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from sam2_utils import alignment

from .predict import build_prompts, image_predict
from .state import Prompts


@dataclass
class FrameResult:
    """One propagated frame, as the interruptible loop yields it.

    frame_idx  : 0-based video frame index.
    masks      : {obj_id: bool HxW in _sam} for this frame.
    logit_conf : mean-foreground-sigmoid proxy for the session's obj_id
                 (NaN if the mask is empty). The "stop discarding logits" proxy.
    pred_iou   : SAM2 mask-decoder predicted-IoU for the session's obj_id, captured
                 from the decoder's IoU head (see _attach_iou_hook). NaN if the hook
                 couldn't read it (e.g. a SAM2 refactor) -> the QC flag rule treats
                 NaN as inert, so that degrades to the pre-capture behaviour.
    """
    frame_idx: int
    masks: dict[int, np.ndarray]
    logit_conf: float
    pred_iou: float


def _attach_iou_hook(video_predictor, sink: dict[int, float]) -> Callable[[], None]:
    """Wrap the predictor's ``_track_step`` to record the mask-decoder predicted IoU
    per frame into ``sink`` (frame_idx -> float). Returns a ``restore()`` callable.

    Why a hook and not the public API: SAM2 *computes* the IoU head output (``ious``)
    in the mask decoder, but ``track_step`` unpacks the decoder tuple and **discards**
    it, ``(_, _, _, low_res_masks, ...) = sam_outputs``, so it never reaches
    ``current_out``, ``inference_state``, or the ``propagate_in_video`` yield (trace:
    sam2/modeling/sam2_base.py ``_forward_sam_heads`` -> ``track_step``). ``_track_step``
    is the last point the value is in hand: it returns ``(current_out, sam_outputs, ...)``
    and ``sam_outputs[2]`` is ``ious`` ([B, M]; M=1 on propagated frames, M=3 on a
    multimask anchor where SAM2 then argmax-selects). We read it **read-only** and pass
    the call through untouched, so masks are bit-identical to an unhooked run.

    Best-effort by design: if ``_track_step`` is absent or its return shape changes, we
    leave pred_iou unpopulated (NaN) rather than raise, same principle as the timing
    wrapper that must never kill a chain. Scope the hook to one propagation run and
    always ``restore()`` (PropagationSession does this in close()).
    """
    orig = getattr(video_predictor, "_track_step", None)
    if orig is None or not callable(orig):
        return lambda: None

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            frame_idx = args[0] if args else kwargs["frame_idx"]
            ious = out[1][2]                       # sam_outputs[2] == decoder IoU head
            # selected mask's predicted IoU; max() == the argmax SAM2 itself selects
            # when multimask (anchor), and the lone value when single-mask (tracking).
            sink[int(frame_idx)] = float(ious.max())
        except Exception:
            pass                                   # leave NaN; never break tracking
        return out

    video_predictor._track_step = wrapped
    return lambda: setattr(video_predictor, "_track_step", orig)


class PropagationSession:
    """Interruptible, resumable SAM2 video propagation for ONE chain.

    Holds the live ``inference_state`` so propagation can be **stopped** at a degrading
    frame, **corrected** (``add_points`` / ``add_mask``), and **resumed**: the
    interruptible-propagation primitive. The headless ``propagate()``
    function below drives it straight through (and is the only thing run_chain uses); a
    future auto-intervention loop or napari GUI drives the *same* methods. No GUI, napari,
    or torch import lives here, the session only calls the predictor's public API plus the
    one read-only ``_track_step`` IoU hook.

    Continuity lives in ``inference_state``, NOT in any generator object. Consequences the
    caller must respect:
      * ``propagate(...)`` yields frames lazily; the caller may ``break`` at any frame and
        every frame seen so far is already recorded in the accumulators.
      * ``add_points`` / ``add_mask`` mutate ``inference_state`` at a chosen frame.
      * Resuming is a **fresh** ``propagate(start_frame_idx=f)`` over the *same* mutated
        state, never ``reset_state`` to resume (that wipes all prompts; it's a
        fresh-chain call, done once in __init__).
      * A re-propagation after a correction re-tracks from the corrected frame onward and
        **overwrites** the stale frames it revisits in the accumulators (last write wins).

    GUI/auto-intervention sketch (not built here):
        sess = PropagationSession(vp, frames_dir, obj_id=1)
        sess.seed(prompts, anchor_idx)
        for fr in sess.propagate(reverse=False):      # fr: FrameResult (mask, conf, pred_iou)
            if looks_bad(fr): break                   # a QC predicate / human stop
        sess.add_points(fr.frame_idx, pts, labels)    # or sess.add_mask(fr.frame_idx, painted)
        for fr in sess.propagate(reverse=False, start_frame_idx=fr.frame_idx):
            ...                                        # resumes over the corrected state

    Accumulators (``video_segments`` / ``frame_conf`` / ``pred_iou``) are frame_idx-keyed
    and persist across calls, so the final dicts reflect the last mask written per frame.
    """

    def __init__(self, video_predictor, frames_dir: str, *, obj_id: int,
                 offload_video_to_cpu: bool = True):
        self.vp = video_predictor
        self.obj_id = obj_id
        # offload_video_to_cpu keeps VRAM bounded for long (~340-frame) chains.
        self.inference_state = video_predictor.init_state(
            video_path=frames_dir,                 # already a str (see prepare_video_frames)
            offload_video_to_cpu=offload_video_to_cpu,
            # offload_state_to_cpu=True,            # enable if OOM on very long chains
        )
        video_predictor.reset_state(self.inference_state)   # per-object scoping (liver pattern)

        self.video_segments: dict[int, dict[int, np.ndarray]] = {}
        self.frame_conf: dict[int, float] = {}
        self.pred_iou: dict[int, float] = {}
        self._iou_sink: dict[int, float] = {}
        self._restore_hook = _attach_iou_hook(video_predictor, self._iou_sink)
        self._closed = False

    # -- seeding / corrections -----------------------------------------------------
    def seed(self, prompts: Prompts, anchor_frame_idx: int, *,
             seed_box: bool = True, seed_points: bool = True,
             seed_negatives: bool = False, seed_mask: bool = False,
             mask_anchor: Optional[np.ndarray] = None) -> None:
        """Seed the anchor frame. Defaults (box + positive point) mirror the notebook's
        anchor seed exactly so masks reproduce.

        SAM2 treats MASK and POINTS/BOX as mutually-exclusive conditioning per frame
        (add_new_mask pops point_inputs and add_new_points_or_box pops mask_inputs, see
        sam2_video_predictor.py), so seed_mask=True takes the add_new_mask path and ignores
        box/points on the anchor frame. The box/points path composes any subset of
        {box, positive, negative} in a single add_new_points_or_box call.

        Negatives are the same same-z neighbour nodes build_prompts placed in _sam (valid in
        both crop and legacy paths, state.prompts stays in _sam). The mask seed needs the
        anchor mask in the PROPAGATION space (legacy _sam or tier-2 _pcrop); run_chain passes
        the right-space mask and guards the tier-1 crop_anchor case.
        """
        if seed_mask:
            if mask_anchor is None:
                raise ValueError("seed_mask=True requires mask_anchor (in the propagation space)")
            self.add_mask(int(anchor_frame_idx), mask_anchor)
            return

        pts = np.asarray(prompts.points_sam, dtype=np.float32)
        labels = np.asarray(prompts.labels, dtype=np.int32)
        keep = np.ones(len(labels), dtype=bool)
        if not seed_points:
            keep &= labels != 1            # drop positives
        if not seed_negatives:
            keep &= labels != 0            # drop negatives
        pts, labels = pts[keep], labels[keep]
        box = (np.asarray(prompts.box_sam, dtype=np.float32)
               if (seed_box and prompts.box_sam is not None) else None)
        if box is None and len(pts) == 0:
            raise ValueError("empty anchor seed: enable at least one of box / points / mask")
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=int(anchor_frame_idx),
            obj_id=self.obj_id,
            box=box,
            points=pts,
            labels=labels,
        )

    def add_points(self, frame_idx: int, points_sam, labels, *,
                   clear_old_points: bool = False) -> None:
        """Inject a point correction at ``frame_idx`` (refinement click(s); incl. negative
        labels). ``clear_old_points=False`` *adds* to the frame's existing prompts (the
        usual refine-click behaviour); pass True to replace them. Mutates inference_state, resume with ``propagate(start_frame_idx=frame_idx)``."""
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=int(frame_idx),
            obj_id=self.obj_id,
            points=np.asarray(points_sam, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int32),
            clear_old_points=clear_old_points,
        )

    def add_mask(self, frame_idx: int, mask_sam) -> None:
        """Inject a painted-mask correction at ``frame_idx`` (human-painted anchor or a
        mid-propagation fix). Mutates inference_state, resume with
        ``propagate(start_frame_idx=frame_idx)``."""
        self.vp.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=int(frame_idx),
            obj_id=self.obj_id,
            mask=np.asarray(mask_sam, dtype=bool),
        )

    # -- propagation ---------------------------------------------------------------
    def propagate(self, *, reverse: bool = False, start_frame_idx: Optional[int] = None,
                  max_frames: Optional[int] = None) -> Iterator[FrameResult]:
        """Lazy generator over propagated frames. Yields a FrameResult per frame and
        records it into the accumulators *before* yielding, so a caller that ``break``s
        early keeps every frame it saw. ``start_frame_idx`` resumes from a corrected
        frame; ``max_frames`` caps how far to track (both forwarded to SAM2)."""
        kw: dict = {"reverse": reverse}
        if start_frame_idx is not None:
            kw["start_frame_idx"] = int(start_frame_idx)
        if max_frames is not None:
            kw["max_frame_num_to_track"] = int(max_frames)
        for f, obj_ids, mask_logits in self.vp.propagate_in_video(self.inference_state, **kw):
            yield self._collect(f, obj_ids, mask_logits)

    def run_bidirectional(self) -> None:
        """Headless convenience: drain forward from the anchor, then reverse. Same call
        order as the pre-refactor two-loop drain, so masks are unchanged."""
        for _ in self.propagate(reverse=False):
            pass
        for _ in self.propagate(reverse=True):
            pass

    # -- internals -----------------------------------------------------------------
    def _collect(self, frame_idx, obj_ids, mask_logits) -> FrameResult:
        per_obj: dict[int, np.ndarray] = {}
        conf = float("nan")
        for i, oid in enumerate(obj_ids):
            lg = mask_logits[i].cpu().numpy()
            m = lg > 0.0
            per_obj[oid] = m
            if oid == self.obj_id:                 # confidence proxy for THIS chain's obj
                fg = lg[m]
                conf = float((1.0 / (1.0 + np.exp(-fg))).mean()) if fg.size else float("nan")
        fi = int(frame_idx)
        self.video_segments[fi] = per_obj
        self.frame_conf[fi] = conf
        piou = self._iou_sink.get(fi, float("nan"))   # set by the hook during this frame's inference
        self.pred_iou[fi] = piou
        return FrameResult(frame_idx=fi, masks=per_obj, logit_conf=conf, pred_iou=piou)

    def close(self) -> None:
        """Restore the patched ``_track_step``. Idempotent. Call when done (or use the
        session as a context manager)."""
        if not self._closed:
            try:
                self._restore_hook()
            finally:
                self._closed = True

    def __enter__(self) -> "PropagationSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def propagate(video_predictor, frames_dir: str, prompts: Prompts,
              anchor_frame_idx: int, *, obj_id: int, seed_negatives: bool = False,
              seed_box: bool = True, seed_points: bool = True, seed_mask: bool = False,
              mask_anchor: Optional[np.ndarray] = None,
              subtimings: Optional[dict] = None
              ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]:
    """Seed the anchor, propagate bidirectionally, collect masks. Headless straight-through
    driver over PropagationSession (the interruptible primitive); this is what run_chain
    uses. Mid-propagation halt/correct/resume is the session's job, not this function's.

    Returns
    -------
    (video_segments, frame_conf, pred_iou)
        video_segments : {frame_idx: {obj_id: mask_sam bool}}
        frame_conf     : {frame_idx: float}, mean-foreground-sigmoid proxy (the
                         `logit_conf` diagnostic column).
        pred_iou       : {frame_idx: float}, SAM2's mask-decoder predicted IoU, now
                         actually captured. NaN per
                         frame only if the hook couldn't read it. run_chain maps this
                         frame_idx -> catmaid_z and hands it to run_qc, where it populates
                         the `pred_iou` QC column and becomes the 4th flag-rule signal
                         (was inert while NaN). NOTE: enabling this signal changes the
                         flag/queue distribution, clear/re-score the manifest after the
                         switch (the mixed-threshold discipline), and set
                         cfg.qc_pred_iou_min <= 0 to record-but-not-flag.

    Lift from: 'init_state' cell + 'Anchor and propagate bidirectionally' cell.
    """
    _t = perf_counter()
    session = PropagationSession(video_predictor, frames_dir, obj_id=obj_id)
    if subtimings is not None:
        subtimings["jpeg_load"] = perf_counter() - _t      # dominated by SAM2's frame decode
    try:
        session.seed(prompts, anchor_frame_idx, seed_box=seed_box,
                     seed_points=seed_points, seed_negatives=seed_negatives,
                     seed_mask=seed_mask, mask_anchor=mask_anchor)
        _t = perf_counter()
        session.run_bidirectional()
        if subtimings is not None:
            subtimings["propagate_only"] = perf_counter() - _t
        return session.video_segments, session.frame_conf, session.pred_iou
    finally:
        session.close()


def _node_id_at(annotate_df: pd.DataFrame, catmaid_z: int, x_tif: float, y_tif: float):
    """The node_id backing a real (non-interpolated) centreline point, or None.

    `centreline_by_z` returns a (x_tif, y_tif) per catmaid_z: for a z with a real
    chain node it is that node's own coordinates verbatim; for a gap z it is a
    linear interpolation with no backing row. `build_prompts` needs an actual
    node_id (it reads the node's CATMAID x/y to rank same-z neighbours by
    distance), so negatives are only built where a real node exists; an
    interpolated z degrades to a positive-only seed rather than guessing a
    neighbour ranking from a synthetic point.
    """
    if not len(annotate_df):
        return None
    z_col = annotate_df["z"]
    x_col = pd.to_numeric(annotate_df["x_tif"], errors="coerce")
    y_col = pd.to_numeric(annotate_df["y_tif"], errors="coerce")
    # Matches on (z, x_tif, y_tif) alone, not scoped to the chain's own neuron:
    # relies on node coordinates being effectively unique across annotate_df.
    match = annotate_df.loc[(z_col == catmaid_z) & (x_col == x_tif) & (y_col == y_tif), "node_id"]
    return match.iloc[0] if len(match) else None


def apply_blowup_guard(video_segments: dict[int, dict[int, np.ndarray]],
                       frame_conf: dict[int, float], pred_iou: dict[int, float],
                       *, obj_id: int, area_factor: float, min_accepted: int = 3) -> set[int]:
    """Replace per-slice masks that blow up (area > area_factor * median non-empty area)
    with the nearest accepted slice's mask, and flag the guarded frames (frame_conf and
    pred_iou -> 0.0 so QC queues them). Mutates the dicts in place; returns the guarded
    frame indices. No-op (returns empty) when fewer than min_accepted non-empty masks exist
    or the median is 0, so a short or mostly-empty chain sets no spurious baseline."""
    areas = {fi: int(seg[obj_id].sum()) for fi, seg in video_segments.items() if obj_id in seg}
    nonempty = {fi: a for fi, a in areas.items() if a > 0}
    if len(nonempty) < min_accepted:
        return set()
    med = float(np.median(list(nonempty.values())))
    if med <= 0:
        return set()
    cap = area_factor * med
    blown = {fi for fi, a in nonempty.items() if a > cap}
    accepted = sorted(fi for fi in nonempty if fi not in blown)
    if not accepted:
        return set()
    for fi in blown:
        nearest = min(accepted, key=lambda j: abs(j - fi))
        video_segments[fi][obj_id] = video_segments[nearest][obj_id].copy()
        frame_conf[fi] = 0.0
        pred_iou[fi] = 0.0
    return blown


def segment_per_slice(image_predictor, frames_dir: str, frame_to_z: dict[int, int],
                      centreline_tif: dict[int, tuple[float, float]],
                      annotate_df: pd.DataFrame, *, cfg, obj_id: int,
                      cw: Optional["alignment.CropWindow"] = None,
                      ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]:
    """Per-frame node-anchored image-mode segmentation, no video propagation.

    For each prepared frame, seed image-mode SAM2 from THAT frame's own centreline
    point (`centreline_tif`, from `predict.centreline_by_z`) instead of propagating
    one anchor's memory across the whole chain, so a mis-tracked cell can never
    carry into a later slice (roadmap Phase 1 item 1). Returns the SAME
    `(video_segments, frame_conf, pred_iou)` shape `propagate()` returns, so
    save/QC/`chain_masks_in_sam` need no change.

    `cw` is the chain's tier-2 crop window: when set, frames on disk and the
    returned masks are in `_pcrop` (the space `prepare_chain_crop_frames` wrote);
    when None, both are `_sam` (the space `prepare_video_frames` wrote). Frames are
    read by 0-indexed `{frame_idx:05d}.jpg`, exactly the naming `prepare_video_frames`
    / `prepare_chain_crop_frames` use, so no torch/video-predictor frame loader is
    needed here, only `cv2.imread`.

    The seed point is built in `_sam` first (`alignment.tif_to_sam`, same as
    `build_prompts`), then, when `cw` is set, remapped `_sam` -> `_pcrop` via
    `cw.sam_to_crop` (the `_sam -> _tif -> _crop` chain `anchor_crop_predict` uses
    for its own prompt remap): no coordinate math is reinvented here. Neighbour
    negatives, when `cfg.k_max_neg > 0`, are `build_prompts`' own negatives for
    the frame's real backing node (see `_node_id_at`), mapped through the same
    `_sam` (-> `_pcrop`) step as the positive, then, when `cw` is set, filtered
    by the same in-bounds-or-positive rule `anchor_crop_predict` uses: negatives
    landing outside the crop window are dropped, the positive is always kept.

    Returns
    -------
    (video_segments, frame_conf, pred_iou)
        video_segments : {frame_idx: {obj_id: mask bool}}, in `_pcrop` (cw set) or
                         `_sam` (cw None), matching `propagate()`.
        frame_conf     : {frame_idx: float}, mean-foreground-sigmoid proxy over the
                         image-mode logits, the same proxy `PropagationSession._collect`
                         computes for the video path.
        pred_iou       : {frame_idx: float}, SAM2's own predicted-IoU score for the
                         mask `image_predict` returned (its single-mask score, or the
                         selected multimask candidate's score).
    """
    import cv2

    space_ratio = (float(cfg.scale) / float(cw.crop_scale)) if cw is not None else 1.0
    contain_r = int(round(cfg.qc_skeleton_dilation_px * space_ratio))
    area_bounds = (cfg.gate_min_area_frac, cfg.gate_max_area_frac)

    video_segments: dict[int, dict[int, np.ndarray]] = {}
    frame_conf: dict[int, float] = {}
    pred_iou: dict[int, float] = {}

    for frame_idx in sorted(frame_to_z):
        catmaid_z = frame_to_z[frame_idx]
        img_path = Path(frames_dir) / f"{frame_idx:05d}.jpg"
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise FileNotFoundError(f"frame not found: {img_path}")
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        x_tif, y_tif = centreline_tif[catmaid_z]
        pos_sam = alignment.tif_to_sam([x_tif, y_tif], cfg.scale)     # (2,)
        points_sam = [[float(pos_sam[0]), float(pos_sam[1])]]
        labels = [1]

        if cfg.k_max_neg > 0:
            node_id = _node_id_at(annotate_df, catmaid_z, x_tif, y_tif)
            if node_id is not None:
                neg_prompts = build_prompts(
                    node_id, catmaid_z, annotate_df, scale=cfg.scale,
                    k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)
                neg_labels = np.asarray(neg_prompts.labels)
                for pt in np.asarray(neg_prompts.points_sam, dtype=float)[neg_labels == 0]:
                    points_sam.append([float(pt[0]), float(pt[1])])
                    labels.append(0)

        points_sam_arr = np.asarray(points_sam, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        if cw is not None:
            # _sam -> _pcrop, then drop out-of-window negatives (same filter as
            # anchor_crop_predict's own _sam -> _crop prompt remap): the positive
            # anchor is kept unconditionally, an out-of-window negative is inert
            # (and would otherwise raise inside SAM2's predict call).
            pts_crop = cw.sam_to_crop(points_sam_arr)
            H_crop, W_crop = image.shape[:2]
            in_bounds = ((pts_crop[:, 0] >= 0) & (pts_crop[:, 0] < W_crop) &
                         (pts_crop[:, 1] >= 0) & (pts_crop[:, 1] < H_crop))
            keep = in_bounds | (labels_arr == 1)
            points_pred, labels_arr = pts_crop[keep], labels_arr[keep]
        else:
            points_pred = points_sam_arr
        prompts = Prompts(points_sam=points_pred, labels=labels_arr)

        mask, score, logits = image_predict(
            image_predictor, image, prompts, multimask=cfg.multimask_anchor,
            select_contain_radius_px=contain_r, select_area_bounds=area_bounds,
            select_exclude_neg=cfg.multimask_exclude_neg,
            select_generous=cfg.multimask_generous)

        video_segments[frame_idx] = {obj_id: mask}
        pred_iou[frame_idx] = float(score)
        # frame_conf is the same mean-foreground-sigmoid proxy PropagationSession._collect
        # computes for the video path: the foreground comes from THIS logit array's own
        # threshold (lg > 0.0), not the full-res `mask` from image_predict. logits is SAM2's
        # low_res_masks (256x256), a different resolution than mask (full crop res), so
        # indexing the low-res logits with the full-res mask raises IndexError on real SAM2.
        lg = np.asarray(logits[0], dtype=float)
        m_lr = lg > 0.0
        fg = lg[m_lr]
        frame_conf[frame_idx] = float((1.0 / (1.0 + np.exp(-fg))).mean()) if fg.size else float("nan")

    if getattr(cfg, "blowup_guard", False):
        guarded = apply_blowup_guard(video_segments, frame_conf, pred_iou,
                                     obj_id=obj_id, area_factor=cfg.blowup_area_factor)
        if guarded:
            print(f"    [blow-up guard] replaced {len(guarded)} slice(s) over "
                  f"{cfg.blowup_area_factor}x median area: {sorted(guarded)}")
    return video_segments, frame_conf, pred_iou
