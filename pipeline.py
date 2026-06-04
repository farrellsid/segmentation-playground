"""
pipeline.py -> phase functions + per-chain orchestration (milestone 1).

This is the "library of phase functions" the notebook lifts into. Each function
is a near-mechanical extraction of a notebook cell-group. `run_chain` at the
bottom is the thin driver -> reproducing the current AVAL masks is the regression
baseline.

State machine, inline QC, and the GUI are left for later milestones. The hooks
where they plug in are marked `# [Mn]`.

Coordinate-space convention
---------------------------
Tag every coordinate/array with the space it lives in, via a suffix:
    _cm   CATMAID stack-pixel space      (annotate_df x, y)
    _tif  full-resolution tif-pixel      (annotate_df x_tif, y_tif)
    _sam  SAM2 *video* input space = full / SCALE (the propagation frames + the
          canonical on-disk mask space; the video predictor + saved masks live here)
    _crop high-res anchor-crop space = (full - crop_origin) / crop_scale. Only the
          *image/anchor* phase uses it (M3.5 default crop); alignment.CropWindow is
          the one place _crop <-> _tif <-> _sam mapping lives. The crop's box is
          mapped back to _sam before the video seed, so _sam stays the spine.
z is even more error-prone, so name it explicitly too:
    catmaid_z   CATMAID section number   (annotate_df z)
    file_z      tif filename z           (catmaid_z - config.FILE_Z_OFFSET)
    frame_idx   0-based video frame index

Canonical on-disk mask space (resolves PIPELINE_CONTEXT §5.1-5.2)
----------------------------------------------------------------
Masks are *computed* at _sam (SCALE). Store them there too: set
save_downscale == SCALE so there's no resample and no 2x skeleton bug. Filenames
are `mask_<catmaid_z:04d>.png` (no 'z' prefix) so qc._iter_mask_paths finds them.
Only diverge from this if you decide you want interpolated higher-res masks for
Blender meshing -> and if so, make that one decision in PipelineConfig, in one place.

A note on node_id matching
--------------------------
Every node lookup uses `annotate_df["node_id"].astype(str) == str(node)`. The
notebook's "Prompt Construction" and "Video Input Setup" cells already do this;
its "Load Image" cell used a bare `== str(...)`, which only works when node_id
happens to be object-dtype and silently returns an empty match otherwise. The
.astype(str) form is dtype-agnostic and reproduces the intended result, so it's
used everywhere here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from time import perf_counter

from sam2_utils import config, alignment


# =============================================================================
# Module-local helpers (shared across phases -> defined once, used twice)
# =============================================================================

def _parse_file_z(p) -> int:
    """'.../1301____z1300.0.tif' -> 1300. Matches the notebook's parse_file_z."""
    token = Path(p).stem.split("z")[-1]
    return int(float(token))


def _downscale_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Area-downsample by `scale`. scale==1 is a no-op copy (notebook helper)."""
    import cv2
    if scale == 1:
        return img.copy()
    return cv2.resize(img, None, fx=1 / scale, fy=1 / scale,
                      interpolation=cv2.INTER_AREA)


def _ensure_cached_frames(subset_tifs, cache_dir: Path, scale: int) -> None:
    """Decode+downscale any tifs not yet in the shared cache.

    One JPEG per file_z, written once ever at this `scale`, named ``z{file_z}.jpg``
    and reused by every chain whose z-range overlaps it. This is where the prep
    cost actually lives (a ~9k x 9k imread + resize); overlapping chains now pay it
    once across the whole dataset instead of once per chain.
    """
    import cv2
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in subset_tifs
               if not (cache_dir / f"z{_parse_file_z(p)}.jpg").exists()]
    if not missing:
        return
    for tif_path in tqdm(missing, desc="caching JPEG frames", unit="frame"):
        img = cv2.imread(str(tif_path))          # BGR, fine for grayscale EM
        img = _downscale_image(img, scale)       # match image-mode coord space
        cv2.imwrite(str(cache_dir / f"z{_parse_file_z(tif_path)}.jpg"), img)


def _link_frame(src: Path, dst: Path) -> None:
    """Expose cache frame `src` at 0-indexed view path `dst`.

    Tries symlink, then hard-link, then a plain copy. On Windows bare symlinks
    need Developer Mode or admin, so the hard-link branch is the usual one -> it
    requires src and dst on the same volume (both live under frames_root, so OK).
    """
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            import os
            os.link(src, dst)                    # hard-link fallback (no privilege)
        except OSError:
            import shutil
            shutil.copy2(src, dst)               # last resort


# =============================================================================
# Run settings (the notebook's top-level knobs) and small structs
# =============================================================================

@dataclass
class PipelineConfig:
    """Tunable run settings -> the notebook's top-level knobs, in one place.

    These are the fields a future settings GUI binds to. Kept deliberately as a
    plain dataclass: no file format, no loader, no validation yet -> that pairs
    with the GUI later (M3/M4). Static project facts (WORM_PATH, checkpoint
    registry, affine, CATMAID) stay in sam2_utils.config; this is per-run tuning
    only.

    A copy lives on each ChainState (see below) so a resumed/re-opened chain
    reproduces with the settings it actually ran under, even if these defaults
    drift later.
    """
    # model / resolution
    model_size: str = "large"          # tiny / small / base_plus / large; read at predictor-build
    scale: int = 8                     # SAM2 input downscale (1 = full-res)
    save_downscale: int = 8            # on-disk mask downscale; == scale is canonical
                                       # (PIPELINE_CONTEXT §5.2). Diverge only if you
                                       # want interpolated higher-res Blender masks.

    # prompt construction
    k_max_neg: int = 7                 # max negative points per object
    neg_radius: int = 150              # neg-point exclusion radius, _sam pixels
    box_margin: int = 10               # anchor-box padding, _sam pixels

    # QC thresholds (M2). Forwarded to qc.compute_metrics; defaults match its
    # original hardcoded rule. PIPELINE_CONTEXT §7 flags these as needing tuning
    # on AVAL -> this is the one place to turn the knobs.
    qc_area_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    qc_temporal_iou_min: float = 0.3
    qc_pred_iou_min: float = 0.5       # inert in M2: pred_iou stays NaN (see propagate)
    qc_skeleton_dilation_px: int = 3
    # chain-level verdict: mark the whole chain "flagged" when this many frames
    # hit `intervene` (>=2 signals). 1 = flag the chain if any frame needs a human.
    qc_intervene_to_flag_chain: int = 1

    # anchor-quality gate (M3.5 item 1). The raw image-mode anchor mask is scored
    # in _sam space *before* propagation (see score_anchor). Deliberately loose
    # first-pass values: per the PIPELINE_CONTEXT §6 ruler, M3.5 levers are judged
    # by *relative* queue deltas at fixed thresholds, not absolute correctness, so
    # these start permissive and get tuned, not trusted. The containment probe
    # reuses qc_skeleton_dilation_px (no separate knob) so anchor- and per-frame
    # containment mean the same thing -> and the §6 item-0 dilation sweep informs both.
    gate_min_area_frac: float = 1e-5       # area floor (frac of frame): catch empty/near-empty
    gate_max_area_frac: float = 0.4       # area ceiling: catch a runaway background grab
    gate_min_largest_cc_frac: float = 0.8  # single-CC: >= this share of fg in the largest blob

    # anchor crop (M3.5 item 2 -> DEFAULT). The image/anchor phase runs image mode
    # on a high-res crop around the node (space _crop, via alignment.CropWindow)
    # instead of the scale-8 full frame, then maps the resulting box back to _sam
    # for the video seed. `scale` is UNCHANGED by this: it still governs video
    # propagation + the canonical mask space; the crop only changes the *anchor*
    # resolution. Set crop_anchor=False to fall back to the legacy scale-8 image
    # phase (A/B testing + the M1 pixel-for-pixel regression baseline).
    # NB the gate's contain radius and area_frac are space-relative, so under the
    # crop the radius is rescaled (x scale/crop_scale) and the area_frac thresholds
    # are measured against the crop, not the full frame -> re-tune on the next run.
    crop_anchor: bool = True           # False -> legacy scale-8 full-frame image phase
    crop_size_tif: int = 1200          # crop window edge in full-res tif px
    crop_scale: int = 2                # crop read downscale (1 = full-res); input edge = size/scale px

    # paths (project-static paths like WORM_PATH stay in sam2_utils.config)
    output_root: Optional[Path] = None     # e.g. .../output_masks; per-chain subdir is derived
    frames_root: Optional[Path] = None     # parent dir for SAM2 JPEG frame folders


    # video seed (M3.5 item 3). Pass build_prompts' neighbour-node negatives to the
    # video seed, not just the positive anchor. Off = M1 seed; on = A/B (§7).
    seed_negatives: bool = False

    # mask post-processing (M3.5 item 5) -> deterministic, no model. Runs before
    # save+QC so QC scores the delivered mask. Off = M1 baseline. Kernels are in
    # scale-8 _sam px; keep <= the neurite half-width.
    postprocess_masks: bool = False
    postproc_open_px: int = 1
    postproc_close_px: int = 1
    postproc_keep_largest_cc: bool = True
    postproc_fill_holes: bool = True

    # which per-frame severity enters the human triage queue (item 4). A frame is
    # queued when flag_count >= this. 2 = intervene-level (>=2 corroborating signals),
    # the default since item 0 (June 2026): single-signal flags are dominated by
    # dilation-sensitive `noskel` noise (flag_rate moved 0.33->0.19 over a 0..10px
    # dilation sweep) while the intervene set is dilation-robust (rate moved <0.005).
    # Set to 1 to restore the legacy "queue every flag" behaviour.
    qc_triage_min_signals: int = 2

@dataclass
class Prompts:
    """SAM2-space prompts for one chain's anchor frame."""
    points_sam: np.ndarray            # (N, 2) float, _sam space
    labels: np.ndarray                # (N,) int, 1 = positive / 0 = negative
    box_sam: Optional[np.ndarray] = None   # (4,) xyxy float, _sam space; None until box_from_mask


@dataclass
class ChainState:
    """
    Everything needed to run, pause, resume, or re-open one chain.

    Persist this to <neuron>/chain_<idx>/state.json. It holds *references and
    metadata*, never the mask arrays themselves -> those live on disk under
    masks/. video_segments stays in RAM during a run and is reconstructed from
    PNGs if you re-open the chain.
    """
    neuron: str                       # = the notebook's TARGET_CELL_NAME (identity, not a knob)
    chain_idx: int
    status: str = "pending"           # pending / running / done / flagged / failed

    # anchor (filled by select_anchor)
    anchor_node_id: Optional[int] = None
    anchor_catmaid_z: Optional[int] = None
    anchor_frame_idx: Optional[int] = None     # filled once video frames are prepped

    # prompts (filled by build_prompts, updated by box_from_mask / GUI edits)
    prompts: Optional[Prompts] = None

    # image-phase result summary (mask itself goes to disk)
    image_score: Optional[float] = None

    # anchor-quality gate verdict (M3.5 item 1), as a plain JSON-ready dict ->
    # parallel to qc_summary. Filled by score_anchor in run_chain. M4 logs this as
    # the per-chain "anchor verdict" feature for the learned P(error) detector.
    anchor_score: Optional[dict] = None

    # video input metadata (filled by prepare_video_frames)
    frames_dir: Optional[str] = None
    frame_to_z: Optional[dict[int, int]] = None
    n_frames: Optional[int] = None

    # qc summary + triage (filled in milestone 2)
    qc_summary: Optional[dict] = None          # flag counts, worst frames, etc.
    triage_frames: list[int] = field(default_factory=list)

    obj_id: int = 1                            # per-chain; increments for multi-obj merge (M5)

    # snapshot of the run settings this chain was processed under (reproducibility):
    # a resumed/re-opened chain replays with the knobs it actually ran under, even
    # if the global defaults have since drifted.
    config: PipelineConfig = field(default_factory=PipelineConfig)


# =============================================================================
# Phase functions  (each ~= one notebook cell-group)
# =============================================================================

def select_anchor(chain: dict, annotate_df: pd.DataFrame) -> tuple[int, int]:
    """Pick the anchor node for a chain and resolve its CATMAID z.

    Currently the mid-node heuristic. Returns (anchor_node_id, anchor_catmaid_z).

    Lift from: 'Load Image' cell (midnode / TARGET_Z).
    # [M4] failed-anchor auto re-pick policy lives here later, not in the driver.
    """
    nodes = chain["nodes"]
    midnode = nodes[len(nodes) // 2]
    z_series = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(midnode), "z"
    ]
    anchor_catmaid_z = int(z_series.item())   # .item() asserts exactly one match
    return midnode, anchor_catmaid_z


def load_frame_sam(catmaid_z: int, *, scale: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Find the tif for `catmaid_z`, read it, downscale by `scale`.

    Returns (image_sam RGB uint8, full_hw) -> full_hw is the pre-downscale (H, W),
    kept only so later steps can map back to full-res if ever needed.

    Lift from: parse_file_z + tif glob + cv2.imread + downscale_image.
    """
    import cv2

    target_file_z = catmaid_z - config.FILE_Z_OFFSET
    tif_files = sorted(config.WORM_PATH.glob("*.tif"))
    matches = [f for f in tif_files if _parse_file_z(f) == target_file_z]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected 1 tif for file_z={target_file_z} "
            f"(CATMAID_z={catmaid_z}), got {len(matches)}: {matches}"
        )
    tif_path = matches[0]

    image_full = cv2.cvtColor(cv2.imread(str(tif_path)), cv2.COLOR_BGR2RGB)
    H_full, W_full = image_full.shape[:2]
    image_sam = _downscale_image(image_full, scale)
    return image_sam, (H_full, W_full)


def build_prompts(anchor_node_id: int, catmaid_z: int, annotate_df: pd.DataFrame,
                  *, scale: int, k_max_neg: int, neg_radius: int) -> Prompts:
    """Anchor skeleton node (positive) + K nearest same-z nodes (negative), in _sam.

    Returns a Prompts with box_sam still None.

    Lift from: 'Prompt Construction' cell. Note the x_tif/y_tif -> _sam division
    by `scale` -> that division is exactly the kind of thing the space-suffix
    convention is meant to make un-loseable.

    `neg_radius` is accepted for signature stability but is intentionally NOT
    applied: the notebook's prompt-construction cell never filtered negatives by
    radius (it only capped count via k_max_neg). Applying it now would change the
    masks and break the M1 regression match. Wire the radius gate in M2 when QC
    thresholds are being tuned, not here.
    """
    # --- positive: the anchor (mid) node, _tif -> _sam ---
    cell_node = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(anchor_node_id)
    ]
    pos_sam = cell_node[["x_tif", "y_tif"]].to_numpy(dtype=float) / scale  # (1, 2)

    points: list[list[float]] = [pos_sam[0].tolist()]
    labels: list[int] = [1]

    # --- negatives: nearest same-z nodes by CATMAID (_cm) distance ---
    cell_x_cm = float(cell_node["x"].iloc[0])
    cell_y_cm = float(cell_node["y"].iloc[0])

    z_points = annotate_df[annotate_df["z"] == catmaid_z].copy()
    z_points["x"] = pd.to_numeric(z_points["x"], errors="coerce")
    z_points["y"] = pd.to_numeric(z_points["y"], errors="coerce")
    z_points["distance"] = np.sqrt(
        (z_points["x"] - cell_x_cm) ** 2 + (z_points["y"] - cell_y_cm) ** 2
    )
    z_points = z_points.sort_values(by="distance").reset_index(drop=True)
    if len(z_points) and z_points.iloc[0]["distance"] == 0:
        z_points = z_points.drop(0).reset_index(drop=True)   # drop the anchor itself

    negnodes_sam = (z_points[["x_tif", "y_tif"]] / scale).reset_index(drop=True)
    n_neg = min(len(z_points), k_max_neg)
    for i in range(n_neg):
        points.append([float(negnodes_sam.iloc[i]["x_tif"]),
                       float(negnodes_sam.iloc[i]["y_tif"])])
        labels.append(0)

    return Prompts(points_sam=np.array(points, dtype=float),
                   labels=np.array(labels, dtype=int))


def image_predict(image_predictor, image_sam: np.ndarray,
                  prompts: Prompts) -> tuple[np.ndarray, float, np.ndarray]:
    """Run image-mode SAM2 on the anchor frame.

    Returns (mask_sam bool HxW, score, logits). Single-mask (multimask_output=False).

    Lift from: 'Image Prediction' cell.
    # [M3] the GUI refinement loop wraps this call (re-predict on each point edit).
    """
    import torch

    with torch.inference_mode():
        image_predictor.set_image(image_sam)
        masks, scores, logits = image_predictor.predict(
            point_coords=np.asarray(prompts.points_sam, dtype=float),
            point_labels=np.asarray(prompts.labels, dtype=int),
            multimask_output=False,
        )
    return masks[0].astype(bool), float(scores[0]), logits


def box_from_mask(mask_sam: np.ndarray, *, margin: int,
                  image_hw_sam: tuple[int, int]) -> Optional[np.ndarray]:
    """Largest connected component -> xyxy box (+margin), clipped to image, _sam space.

    Returns the box, or None if the mask is empty -> None is the signal to flag the
    chain for human review rather than feed garbage into propagation.

    Lift from: 'Bounding Box Generation' cell.
    """
    from skimage.measure import label as cc_label, regionprops

    m = np.asarray(mask_sam).astype(bool)
    if not m.any():
        return None

    # largest blob only (suppresses stray membrane fragments)
    lbl = cc_label(m, connectivity=2)
    m = lbl == (1 + int(np.argmax([r.area for r in regionprops(lbl)])))

    H_sam, W_sam = image_hw_sam
    ys, xs = np.where(m)
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, W_sam - 1)
    y1 = min(int(ys.max()) + margin, H_sam - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


@dataclass
class AnchorScore:
    """Threshold-light quality verdict for one chain's anchor (image-phase) mask.

    The geometry here is judged entirely in _sam space -> the space image_predict
    works in, and the space prompts.points_sam already lives in -> so there is *no*
    coordinate transform in this function (deliberately: the anchor mask and the
    positive prompt point share one frame). That keeps it off the §4/§5 bug-prone
    transform path.

    Three sub-checks, mirroring the M3.5 gate (PIPELINE_CONTEXT §6 item 1):
      contained        -> does the mask cover the positive (skeleton) prompt point,
                          within a small radius? Tri-state, same meaning as
                          qc.skeleton_contained but encoded JSON-clean:
                          True / False / None(no positive point -> abstain).
      n_components,
      largest_cc_frac   -> single-CC health: fraction of foreground in the largest
                          connected component (a clean anchor is ~one blob).
      area_frac         -> foreground as a fraction of the frame: floored to catch an
                          empty/near-empty mask, ceiled to catch a runaway grab of
                          background.

    `passed` is the AND of the enabled checks; an abstaining (None) containment does
    not fail. `reasons` lists the checks that fired, reusing the qc vocabulary
    ('noskel' / 'area' / 'frag') so the gate and the per-frame QC speak the same
    language downstream.
    """
    contained: Optional[bool]
    n_components: int
    largest_cc_frac: float
    area_frac: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _anchor_score_to_dict(s: AnchorScore) -> dict:
    """JSON-ready plain dict (no numpy types) for ChainState.anchor_score."""
    return {
        "contained": None if s.contained is None else bool(s.contained),
        "n_components": int(s.n_components),
        "largest_cc_frac": float(s.largest_cc_frac),
        "area_frac": float(s.area_frac),
        "passed": bool(s.passed),
        "reasons": list(s.reasons),
    }


def score_anchor(mask_sam: np.ndarray, prompts: Prompts, *,
                 image_hw_sam: tuple[int, int],
                 contain_radius_px: int,
                 min_area_frac: float,
                 max_area_frac: float,
                 min_largest_cc_frac: float) -> AnchorScore:
    """Score the raw image-mode anchor mask for propagation-readiness, in _sam space.

    Called *before* box_from_mask (it judges the raw multi-blob mask, not the
    largest-CC box) and before propagation, so a bad anchor costs one frame's
    compute instead of a wasted ~300-frame propagate. This is M3.5 item 1's
    *scoring* half only: it is pure (reads the mask + prompts, writes nothing) and
    decides nothing -> the gate that escalates prompts / re-picks the node / blocks
    propagation consumes this verdict in the next increment.

    Lift/parallel: the containment probe is the same neighbourhood test as
    qc.compute_metrics (so anchor- and per-frame containment agree); the single-CC
    measure generalises the largest-component pick already in box_from_mask.
    """
    from skimage.measure import label as cc_label

    H_sam, W_sam = image_hw_sam
    frame_px = int(H_sam) * int(W_sam)
    m = np.asarray(mask_sam).astype(bool)
    area = int(m.sum())
    area_frac = (area / frame_px) if frame_px else 0.0

    # --- containment: does the mask cover the positive (anchor) prompt point? ---
    # Tri-state, matching qc.skeleton_contained: an empty mask with a node present
    # is an explicit miss (False); no positive point at all is an abstain (None).
    pos = None
    if prompts is not None and prompts.points_sam is not None:
        pts = np.asarray(prompts.points_sam, dtype=float)
        lbl = np.asarray(prompts.labels)
        pos_pts = pts[lbl == 1]
        if len(pos_pts):
            pos = pos_pts[0]                       # the anchor (skeleton) node, _sam
    if pos is None:
        contained: Optional[bool] = None
    elif area == 0:
        contained = False
    else:
        px_sam, py_sam = float(pos[0]), float(pos[1])
        sx_i, sy_i = int(round(px_sam)), int(round(py_sam))
        if 0 <= sy_i < m.shape[0] and 0 <= sx_i < m.shape[1]:
            r = contain_radius_px
            y0, y1 = max(0, sy_i - r), min(m.shape[0], sy_i + r + 1)
            x0, x1 = max(0, sx_i - r), min(m.shape[1], sx_i + r + 1)
            contained = bool(m[y0:y1, x0:x1].any())
        else:
            contained = False                      # node maps outside the frame

    # --- single-CC health ---
    if area == 0:
        n_cc, largest_cc_frac = 0, 0.0
    else:
        lbl_img = cc_label(m, connectivity=2)
        sizes = np.bincount(lbl_img.ravel())[1:]   # drop background (label 0)
        n_cc = int(sizes.size)
        largest_cc_frac = float(sizes.max() / area) if sizes.size else 0.0

    # --- compose the verdict ---
    reasons: list[str] = []
    if not (min_area_frac <= area_frac <= max_area_frac):
        reasons.append("area")
    if largest_cc_frac < min_largest_cc_frac:
        reasons.append("frag")
    if contained is False:                         # None abstains, must not fail
        reasons.append("noskel")

    return AnchorScore(
        contained=contained,
        n_components=n_cc,
        largest_cc_frac=largest_cc_frac,
        area_frac=area_frac,
        passed=(len(reasons) == 0),
        reasons=reasons,
    )


def anchor_crop_predict(image_predictor, image_full: np.ndarray, full_hw: tuple[int, int],
                        anchor_node_id: int, prompts_sam: "Prompts", annotate_df: pd.DataFrame,
                        *, scale: int, crop_size_tif: int, crop_scale: int
                        ) -> tuple[np.ndarray, float, "alignment.CropWindow", "Prompts"]:
    """Image-mode anchor prediction on a high-res crop (M3.5 item 2 -> default path).

    Crops a `crop_size_tif` window around the anchor node (alignment.CropWindow ->
    the single home of _crop<->_tif<->_sam mapping), runs image mode in _crop at
    `crop_scale`, and returns the mask + the CropWindow so the caller can map the
    box back to _sam for the video seed.

    The prompt POINTS are the already-built _sam prompts remapped into _crop
    (_sam -> _tif via *scale, then CropWindow.tif_to_crop); negatives that fall
    outside the window are dropped (the positive anchor is inside by construction).
    The returned Prompts is in _crop, so the gate (score_anchor) can score in the
    same space the mask lives in. The original _sam prompts are untouched -> they
    still seed the video positive point; only the box comes from the crop.

    Returns (mask_crop bool HxW, score, cw, prompts_crop).
    """
    from sam2_utils import alignment

    # anchor node in _tif (the space CropWindow centers in)
    node = annotate_df.loc[annotate_df["node_id"].astype(str) == str(anchor_node_id)]
    node_xy_tif = node[["x_tif", "y_tif"]].to_numpy(dtype=float)[0]

    cw = alignment.CropWindow.around_node(
        node_xy_tif, size_tif=crop_size_tif, image_hw_tif=full_hw,
        crop_scale=crop_scale, sam_scale=scale)

    crop_full = image_full[cw.slice_tif()]                  # _tif window
    crop_img = _downscale_image(crop_full, crop_scale)      # _crop input image
    H_crop, W_crop = crop_img.shape[:2]

    # _sam prompt points -> _tif -> _crop. Keep all positives; drop out-of-window negatives.
    pts_sam = np.asarray(prompts_sam.points_sam, dtype=float)
    labels = np.asarray(prompts_sam.labels, dtype=int)
    pts_crop = cw.tif_to_crop(pts_sam * scale)              # _sam->_tif is *scale
    in_bounds = ((pts_crop[:, 0] >= 0) & (pts_crop[:, 0] < W_crop) &
                 (pts_crop[:, 1] >= 0) & (pts_crop[:, 1] < H_crop))
    keep = in_bounds | (labels == 1)
    prompts_crop = Prompts(points_sam=pts_crop[keep], labels=labels[keep])  # NB: _crop coords

    mask_crop, score, _logits = image_predict(image_predictor, crop_img, prompts_crop)
    return mask_crop, score, cw, prompts_crop


def prepare_video_frames(chain: dict, annotate_df: pd.DataFrame, *, scale: int,
                         frames_root: Optional[Path],
                         anchor_catmaid_z: int,
                         neuron: str, chain_idx: int
                         ) -> tuple[str, dict[int, int], int, int]:
    """Give SAM2 the 0-indexed downscaled JPEG sequence it needs -> with reuse.

    Two-tier layout under frames_root:
      * a shared cache  ``frames_cache_s{scale}/z{file_z}.jpg`` -> each frame
        decoded+downscaled ONCE ever (see _ensure_cached_frames);
      * a per-chain view ``chain_views/{neuron}_chain{idx:02d}_s{scale}/{i:05d}.jpg``
        of links into that cache, contiguous and 0-indexed as init_state requires.

    Overlapping chains share the cache, so the expensive decode happens once per z
    across the whole dataset instead of once per chain -> this is the fix for the
    frame-prep bottleneck. The cached JPEG bytes are identical to the old per-range
    writer (same imread -> downscale -> imwrite), so masks still reproduce
    pixel-for-pixel.

    The view is namespaced by `neuron`+`chain_idx` so a batch over many neurons
    can't collide (AVAL chain0 vs AVAR chain0), and is rebuilt from scratch each
    call -> links are free, so this sidesteps stale-link risk if a chain's z-range
    changed between runs.

    Returns (view_dir str, frame_to_z, anchor_frame_idx, n_frames).

    Lift from: 'Video Input Setup' frame-prep cell.
    """
    import shutil

    if frames_root is None:
        raise ValueError("PipelineConfig.frames_root must be set for video frame prep")

    # z-extent over ALL chain nodes (non-monotonic in z -> can't use nodes[0]/[-1])
    chain_z = [
        int(annotate_df.loc[
            annotate_df["node_id"].astype(str) == str(n), "z"
        ].item())
        for n in chain["nodes"]
    ]
    start_z, end_z = min(chain_z), max(chain_z)

    start_file_z = start_z - config.FILE_Z_OFFSET
    end_file_z = end_z - config.FILE_Z_OFFSET
    target_file_z = anchor_catmaid_z - config.FILE_Z_OFFSET

    all_tifs = sorted(config.WORM_PATH.glob("*.tif"), key=_parse_file_z)
    subset_tifs = [f for f in all_tifs
                   if start_file_z <= _parse_file_z(f) <= end_file_z]

    frames_root = Path(frames_root)

    # 1. shared decode cache (write-once, keyed by file_z + scale)
    cache_dir = frames_root / f"frames_cache_s{scale}"
    _ensure_cached_frames(subset_tifs, cache_dir, scale)

    # 2. per-chain 0-indexed link view (namespaced; rebuilt fresh each call)
    view_dir = frames_root / "chain_views" / f"{neuron}_chain{chain_idx:02d}_s{scale}"
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)

    anchor_frame_idx: Optional[int] = None
    for i, tif_path in enumerate(subset_tifs):
        if _parse_file_z(tif_path) == target_file_z:
            anchor_frame_idx = i                 # anchor, in 0-based video index
        _link_frame(cache_dir / f"z{_parse_file_z(tif_path)}.jpg",
                    view_dir / f"{i:05d}.jpg")

    if anchor_frame_idx is None:
        raise AssertionError(
            f"anchor file_z={target_file_z} not in [{start_file_z}, {end_file_z}]"
        )

    frame_to_z = {i: _parse_file_z(tif) + config.FILE_Z_OFFSET
                  for i, tif in enumerate(subset_tifs)}

    # init_state needs a STRING path, not a Path, or it raises
    # "Only MP4 video and JPEG folder are supported".
    return str(view_dir), frame_to_z, anchor_frame_idx, len(subset_tifs)


def propagate(video_predictor, frames_dir: str, prompts: Prompts,
              anchor_frame_idx: int, *, obj_id: int, seed_negatives: bool = False,
              subtimings: Optional[dict] = None
              ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float]]:
    """Seed box+point on the anchor frame, propagate bidirectionally, collect masks.

    Returns
    -------
    (video_segments, frame_conf)
        video_segments : {frame_idx: {obj_id: mask_sam bool}}
        frame_conf      : {frame_idx: float} -> a per-frame mask confidence proxy
                          (mean foreground sigmoid of the mask logits). This is the
                          M2 resolution of PIPELINE_CONTEXT §5.3 "scores discarded":
                          we stop throwing the logits away. NOTE it is a *proxy*,
                          not SAM2's calibrated predicted-IoU (which propagate_in_video
                          does not surface), so it is recorded for inspection as the
                          `logit_conf` column but is NOT wired into the flag rule yet
                          (pred_iou stays NaN). Calibrating/promoting it to a flag
                          signal is deferred -> the geometric + temporal + skeleton
                          signals drive M2 flagging.

    Lift from: 'init_state' cell + 'Anchor and propagate bidirectionally' cell.
    # [M3/M4] propagate_in_video is a generator: this is the loop you later
    #      restructure to break at a degrading frame, inject a correction with
    #      add_new_points_or_box, and resume.
    """
    _t = perf_counter()
    # offload_video_to_cpu keeps VRAM bounded for long (~340-frame) chains.
    inference_state = video_predictor.init_state(
        video_path=frames_dir,                # already a str (see prepare_video_frames)
        offload_video_to_cpu=True,
        # offload_state_to_cpu=True,          # enable if OOM on very long chains
    )
    if subtimings is not None:
        subtimings["jpeg_load"] = perf_counter() - _t      # SAM2's frame decode
    video_predictor.reset_state(inference_state)   # per-object scoping (liver pattern)
    # box + the positive (skeleton) point(s); optionally the neighbour-node
    # negatives too (M3.5 item 3). Negatives are the same same-z neighbour nodes
    # build_prompts already placed in _sam -> passing them here is the video-seed
    # analogue of the image-mode negatives, valid in both crop and legacy paths
    # (state.prompts stays in _sam regardless of crop). Default off = positives
    # only = the M1 seed; flip on to A/B, esp. concave E/U chains (measure whether
    # the negatives land in the concavity -> PIPELINE_CONTEXT §7 *Negative points*).
    pts = np.asarray(prompts.points_sam, dtype=np.float32)
    labels = np.asarray(prompts.labels, dtype=np.int32)
    if not seed_negatives:
        keep = labels == 1
        pts, labels = pts[keep], labels[keep]
    video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=anchor_frame_idx,
        obj_id=obj_id,
        box=np.asarray(prompts.box_sam, dtype=np.float32),
        points=pts,
        labels=labels,
    )

    video_segments: dict[int, dict[int, np.ndarray]] = {}
    frame_conf: dict[int, float] = {}

    def _collect(frame_idx, obj_ids, mask_logits):
        per_obj = {}
        for i, oid in enumerate(obj_ids):
            lg = mask_logits[i].cpu().numpy()
            m = lg > 0.0
            per_obj[oid] = m
            # confidence proxy for THIS chain's object only (single-obj in M1/M2)
            if oid == obj_id:
                fg = lg[m]
                # mean foreground probability; NaN when the mask is empty
                frame_conf[frame_idx] = (
                    float((1.0 / (1.0 + np.exp(-fg))).mean()) if fg.size else float("nan")
                )
        video_segments[frame_idx] = per_obj

    _t = perf_counter()
    for f, ids, logits in video_predictor.propagate_in_video(inference_state):
        _collect(f, ids, logits)
    for f, ids, logits in video_predictor.propagate_in_video(inference_state, reverse=True):
        _collect(f, ids, logits)
    if subtimings is not None:
        subtimings["propagate_only"] = perf_counter() - _t

    return video_segments, frame_conf


def save_masks(video_segments: dict[int, dict[int, np.ndarray]],
               frame_to_z: dict[int, int], out_dir: Path, *,
               obj_id: int, mask_space_downscale: int) -> int:
    """Write one 0/255 uint8 PNG per frame at the canonical mask space.

    Returns count written. Files are `mask_<catmaid_z:04d>.png`, single-channel,
    255 = inside the neurite, 0 = background -> the notebook's exact format, so the
    masks are directly viewable AND pixel-comparable to the notebook output (the
    M1 done-check). qc._load_binary reads `arr > 0`, so this stays fully
    compatible with compute_metrics.

    Why NOT qc.save_masks here: that writer stores uint16 *instance labels*
    (foreground pixel value == obj_id). For a single object obj_id is 1, and value
    1 in a 16-bit image is visually indistinguishable from black -> it looks empty
    and is destroyed by any 16->8-bit conversion, which is exactly the "empty
    masks" confusion. Instance-label encoding is a multi-object concern; adopt it
    in M5 when aggregating several objects per neuron, not now.

    No resample happens: masks already live at _sam, and under the canonical rule
    (scale == save_downscale) that IS the on-disk space. `mask_space_downscale` is
    accepted for signature stability / future divergence but isn't applied here;
    if you ever set save_downscale != scale you'll need the source scale too, and
    the resample should live in this one function.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    for frame_idx in sorted(video_segments):
        seg = video_segments[frame_idx]
        if obj_id not in seg:
            skipped += 1
            continue
        raw = np.asarray(seg[obj_id])
        mask_hw = raw[0] if raw.ndim == 3 else raw     # (H_sam, W_sam) bool
        png = (mask_hw.astype("uint8") * 255)          # 0 / 255, viewable
        catmaid_z = frame_to_z[frame_idx]
        cv2.imwrite(str(out_dir / f"mask_{catmaid_z:04d}.png"), png)
        written += 1

    print(f"[save_masks] wrote {written} masks ({skipped} frames skipped, "
          f"obj_id {obj_id} absent) -> {out_dir}  (0/255 uint8, named by catmaid_z)")
    return written

def postprocess_mask(mask_sam: np.ndarray, *, open_px: int = 1, close_px: int = 1,
                     keep_largest_cc: bool = True, fill_holes: bool = True) -> np.ndarray:
    """Deterministic, model-free cleanup of one propagated mask, in _sam space.

    Masks are downscaled then NN-upscaled, so they come out blocky / speckled /
    holey while true neurite borders are smooth -> cleanup priors are safe (§7).
    Order: open (despeckle) -> close (bridge grid gaps) -> largest-CC (drop
    detached fragments, generalising box_from_mask's pick) -> fill holes. Keep
    kernels small or thin neurites erode away. Empty in -> empty out.

    NB keep_largest_cc drops genuinely-split components too -> fine for a box, but
    it can silently erase a real second process in the *saved* mask, and it can
    paper over a propagation failure QC is meant to flag. Run QC *after* this so
    QC scores the delivered mask, and watch the flag distribution when enabling.
    """
    from scipy import ndimage
    from skimage.measure import label as cc_label

    m = np.asarray(mask_sam).astype(bool)
    if not m.any():
        return m
    if open_px > 0:
        m = ndimage.binary_opening(m, iterations=open_px)
    if close_px > 0:
        m = ndimage.binary_closing(m, iterations=close_px)
    if keep_largest_cc and m.any():
        lbl = cc_label(m, connectivity=2)
        sizes = np.bincount(lbl.ravel())[1:]          # drop background
        if sizes.size:
            m = lbl == (1 + int(np.argmax(sizes)))
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    return m

def run_qc(masks_dir: Path, skeleton: pd.DataFrame, *,
           frame_to_z: dict[int, int],
           frame_conf: Optional[dict[int, float]], cfg: PipelineConfig,
           qc_csv_path: Optional[Path] = None) -> tuple[dict, list[int], str]:
    """Compute QC over the saved masks, write qc.csv, return (summary, triage_z, status).

    Resolves PIPELINE_CONTEXT §5.4 to the extent M2 needs: QC runs over the
    just-saved chain (joining the inline-captured confidence), produces flags, and
    drives the chain's verdict -> all headless, no human. It still reads the PNGs
    back off disk rather than scoring inside the propagate loop; that fully-inline,
    interleaved form is only required for *halt-and-re-prompt*, which is M3/M4. So
    this is "QC moved into the run," not yet "QC moved into the propagation loop."

    Signals and the composite flag/intervene rule come straight from
    ``qc.compute_metrics`` (single source of truth); thresholds come from ``cfg``.

    Parameters
    ----------
    skeleton : DataFrame
        The skeleton of *this chain only* (columns z, x_tif, y_tif), NOT the whole
        neuron. This matters: a neuron like AVAL is many chains, so its nodes cross
        a given z at several xy positions and their centroid lands off any single
        process -> using it makes containment fail on every frame (the AVAL 100%-flag
        bug). Filtering to the chain's own nodes gives a meaningful per-z probe.

    Returns
    -------
    qc_summary : dict (json-safe)   -> counts + worst frames, for ChainState
    triage_z   : list[int]          -> CATMAID-z of every flagged frame (the queue;
                                      z-keyed to match qc, mask filenames, and
                                      review.load_chain's triage_is_z default)
    status     : "done" | "flagged"
    """
    from sam2_utils import qc   # lazy: keeps pipeline import free of qc's heavy deps

    df = qc.compute_metrics(
        masks_dir,
        skeleton=skeleton,
        scale=cfg.scale,
        save_downscale=cfg.save_downscale,
        skeleton_dilation_px=cfg.qc_skeleton_dilation_px,
        area_ratio_bounds=cfg.qc_area_ratio_bounds,
        temporal_iou_min=cfg.qc_temporal_iou_min,
        pred_iou_min=cfg.qc_pred_iou_min,
    )

    # Attach the inline confidence proxy as a *diagnostic* column (z-keyed).
    # Deliberately NOT named pred_iou and NOT in the flag rule -> see propagate().
    if frame_conf:
        z_conf = {frame_to_z[fi]: c for fi, c in frame_conf.items()
                  if fi in frame_to_z}
        df["logit_conf"] = df.index.map(lambda z: z_conf.get(int(z), float("nan")))

    # The human triage queue is the frames at/above the configured severity
    # (flag_count >= qc_triage_min_signals; default 2 = intervene-level). `flag`
    # (>=1 signal) stays in the row as a diagnostic — single-signal flags are kept on
    # disk for M4 labels, just not surfaced to a human. Persisted to qc.csv so the
    # cross-chain rollup (batch.build_triage_queue) can filter on the artifact alone.
    df["queue"] = df["flag_count"] >= cfg.qc_triage_min_signals

    if qc_csv_path is not None:
        qc_csv_path = Path(qc_csv_path)
        qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(qc_csv_path)   # index = z

    n = int(len(df))
    n_flag = int(df["flag"].sum())
    n_int = int(df["intervene"].sum())
    n_queue = int(df["queue"].sum())
    n_noskel = int((df["skeleton_contained"] == False).sum())   # noqa: E712
    n_skel_na = int(df["skeleton_contained"].isna().sum())
    triage_z = sorted(int(z) for z in df.index[df["queue"]])

    # worst queue frames first, for a quick human glance from state.json alone
    worst = (df[df["queue"]].sort_values("flag_count", ascending=False).head(10))
    worst_frames = [
        {
            "z": int(z),
            "flag_count": int(r["flag_count"]),
            "area_ratio": (None if pd.isna(r["area_ratio"]) else round(float(r["area_ratio"]), 3)),
            "temporal_iou": (None if pd.isna(r["temporal_iou"]) else round(float(r["temporal_iou"]), 3)),
            "skeleton_contained": bool(r["skeleton_contained"]),
        }
        for z, r in worst.iterrows()
    ]

    qc_summary = {
        "n_frames": n,
        "n_flagged": n_flag,          # all >=1-signal flags (diagnostic; kept for M4 labels)
        "n_queue": n_queue,           # frames surfaced to a human (item 4 gate)
        "n_intervene": n_int,
        "n_missing_skel": n_noskel,
        "n_skel_not_assessable": n_skel_na,
        "flag_rate": (round(n_flag / n, 4) if n else 0.0),
        "queue_rate": (round(n_queue / n, 4) if n else 0.0),
        "thresholds": {
            "area_ratio_bounds": list(cfg.qc_area_ratio_bounds),
            "temporal_iou_min": cfg.qc_temporal_iou_min,
            "pred_iou_min": cfg.qc_pred_iou_min,
            "skeleton_dilation_px": cfg.qc_skeleton_dilation_px,
            "triage_min_signals": cfg.qc_triage_min_signals,
        },
        "worst_frames": worst_frames,
    }

    # chain verdict keyed on the SAME queue definition as the frame queue, so the two
    # never disagree. Behaviour-preserving at defaults: qc_triage_min_signals=2 makes
    # n_queue == n_intervene, so this is identical to the prior
    # `n_int >= qc_intervene_to_flag_chain` rule.
    status = "flagged" if n_queue >= cfg.qc_intervene_to_flag_chain else "done"
    return qc_summary, triage_z, status


# =============================================================================
# Serialization  (ChainState <-> state.json)
# =============================================================================
# Three things plain json won't handle on its own:
#   - numpy arrays in Prompts (points / labels / box) -> lists on dump, arrays on load
#   - frame_to_z keys come back as strings           -> cast to int on load
#   - Path fields in PipelineConfig                   -> str on dump, Path on load

def _prompts_to_dict(p: Optional[Prompts]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "points_sam": np.asarray(p.points_sam).tolist(),
        "labels": np.asarray(p.labels).tolist(),
        "box_sam": None if p.box_sam is None else np.asarray(p.box_sam).tolist(),
    }


def _prompts_from_dict(d: Optional[dict]) -> Optional[Prompts]:
    if d is None:
        return None
    box = d.get("box_sam")
    return Prompts(
        points_sam=np.array(d["points_sam"], dtype=float),
        labels=np.array(d["labels"], dtype=int),
        box_sam=None if box is None else np.array(box, dtype=np.float32),
    )


def _config_to_dict(c: PipelineConfig) -> dict:
    d = asdict(c)
    d["output_root"] = None if c.output_root is None else str(c.output_root)
    d["frames_root"] = None if c.frames_root is None else str(c.frames_root)
    return d


def _config_from_dict(d: Optional[dict]) -> PipelineConfig:
    d = dict(d or {})
    if d.get("output_root") is not None:
        d["output_root"] = Path(d["output_root"])
    if d.get("frames_root") is not None:
        d["frames_root"] = Path(d["frames_root"])
    return PipelineConfig(**d)


def state_to_dict(state: ChainState) -> dict:
    """Plain-json-safe dict view of a ChainState."""
    ftz = state.frame_to_z
    return {
        "neuron": state.neuron,
        "chain_idx": state.chain_idx,
        "status": state.status,
        "anchor_node_id": state.anchor_node_id,
        "anchor_catmaid_z": state.anchor_catmaid_z,
        "anchor_frame_idx": state.anchor_frame_idx,
        "prompts": _prompts_to_dict(state.prompts),
        "image_score": None if state.image_score is None else float(state.image_score),
        "anchor_score": state.anchor_score,
        "frames_dir": state.frames_dir,
        "frame_to_z": None if ftz is None else {str(k): int(v) for k, v in ftz.items()},
        "n_frames": state.n_frames,
        "qc_summary": state.qc_summary,
        "triage_frames": list(state.triage_frames),
        "obj_id": state.obj_id,
        "config": _config_to_dict(state.config),
    }


def state_from_dict(d: dict) -> ChainState:
    ftz = d.get("frame_to_z")
    return ChainState(
        neuron=d["neuron"],
        chain_idx=d["chain_idx"],
        status=d.get("status", "pending"),
        anchor_node_id=d.get("anchor_node_id"),
        anchor_catmaid_z=d.get("anchor_catmaid_z"),
        anchor_frame_idx=d.get("anchor_frame_idx"),
        prompts=_prompts_from_dict(d.get("prompts")),
        image_score=d.get("image_score"),
        anchor_score=d.get("anchor_score"),
        frames_dir=d.get("frames_dir"),
        frame_to_z=None if ftz is None else {int(k): int(v) for k, v in ftz.items()},
        n_frames=d.get("n_frames"),
        qc_summary=d.get("qc_summary"),
        triage_frames=list(d.get("triage_frames", [])),
        obj_id=d.get("obj_id", 1),
        config=_config_from_dict(d.get("config")),
    )


def save_state(state: ChainState, path: str | Path) -> Path:
    """Serialize a ChainState to state.json (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_to_dict(state), indent=2))
    return path


def load_state(path: str | Path) -> ChainState:
    """Reload a ChainState from state.json."""
    return state_from_dict(json.loads(Path(path).read_text()))


# =============================================================================
# Thin driver  (the ~20-line regression target)
# =============================================================================

def run_chain(state: ChainState, *, image_predictor, video_predictor,
              annotate_df: pd.DataFrame, chain: dict,
              on_video_phase: Optional[Callable[[], None]] = None) -> ChainState:
    """Run one chain end-to-end by composing the phases above.

    No new behavior vs. the notebook -> this just makes the call order explicit and
    threads ChainState through, so you can see exactly which field each phase fills.
    All tunables come from state.config; the state carries its own settings so it
    stays self-contained for serialize / resume. Getting this to reproduce the
    AVAL masks is the milestone-1 done condition.

    `on_video_phase` is an optional callback fired once, after the image phase and
    before video propagation. The driver passes diagnostics.cleanup_vram here so
    VRAM is reclaimed between phases (the notebook does this; skipping it leaks
    VRAM). Keeping it a callback lets pipeline.py stay free of a torch/diagnostics
    import -> the library doesn't decide *how* to clean up, the driver does.
    """
    cfg = state.config
    state.status = "running"

    tag = f"{state.neuron} chain {state.chain_idx:02d}"

    timings: dict[str, float] = {}
    subtimings: dict[str, float] = {}
    _clk = {"t": perf_counter(), "label": None}

    def _step(n: int, label: str) -> None:
        now = perf_counter()
        if _clk["label"] is not None:
            timings[_clk["label"]] = now - _clk["t"]
        _clk["t"], _clk["label"] = now, label
        print(f"\n[{tag}] step {n}/9 -> {label}", flush=True)
        
    def _finish() -> None:
        """Close out the in-flight phase and stamp timings onto the state.
        Call before every return so phase_seconds is set on all paths."""
        if _clk["label"] is not None:
            timings[_clk["label"]] = perf_counter() - _clk["t"]
            _clk["label"] = None
        state.phase_seconds = timings
        state.phase_subseconds = subtimings

    # 1. anchor
    _step(1, "select anchor")
    state.anchor_node_id, state.anchor_catmaid_z = select_anchor(chain, annotate_df)
    print(f"    anchor node {state.anchor_node_id}  (CATMAID z={state.anchor_catmaid_z})")

    # 2. anchor frame. The crop path needs the FULL-res frame to crop from; the
    #    legacy path needs the scale-8 frame the image predictor runs on directly.
    _step(2, "load anchor frame")
    if cfg.crop_anchor:
        image_full, full_hw = load_frame_sam(state.anchor_catmaid_z, scale=1)   # _tif
        image_sam = None
        print(f"    full-res frame {full_hw[1]}x{full_hw[0]} -> {cfg.crop_size_tif}px "
              f"_tif crop @ crop_scale {cfg.crop_scale}")
    else:
        image_sam, full_hw = load_frame_sam(state.anchor_catmaid_z, scale=cfg.scale)
        image_full = None
        print(f"    _sam frame {image_sam.shape[1]}x{image_sam.shape[0]} "
              f"(full {full_hw[1]}x{full_hw[0]}, scale {cfg.scale})")

    # 3. prompts (always built in _sam -> they seed the video positive point; the
    #    crop path remaps a copy into _crop for the anchor prediction).
    _step(3, "build prompts")
    state.prompts = build_prompts(state.anchor_node_id, state.anchor_catmaid_z,
                                  annotate_df, scale=cfg.scale,
                                  k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)
    n_pos = int((state.prompts.labels == 1).sum())
    n_neg = int((state.prompts.labels == 0).sum())
    print(f"    {n_pos} positive + {n_neg} negative point(s)")

    # 4. image mode -> on a high-res crop by default (M3.5 item 2), else scale-8.
    _step(4, "image-mode prediction")
    if cfg.crop_anchor:
        mask_anchor, state.image_score, cw, prompts_anchor = anchor_crop_predict(
            image_predictor, image_full, full_hw, state.anchor_node_id,
            state.prompts, annotate_df, scale=cfg.scale,
            crop_size_tif=cfg.crop_size_tif, crop_scale=cfg.crop_scale)
        # gate tolerances are space-relative: 1 _sam px = scale/crop_scale _crop px,
        # so rescale the contain radius + box margin. (area_frac is now measured
        # against the crop, not the full frame -> its thresholds want re-tuning.)
        space_ratio = cfg.scale / cfg.crop_scale
        contain_r = int(round(cfg.qc_skeleton_dilation_px * space_ratio))
        margin_local = int(round(cfg.box_margin * space_ratio))
    else:
        mask_anchor, state.image_score, _logits = image_predict(
            image_predictor, image_sam, state.prompts)
        cw, prompts_anchor = None, state.prompts
        contain_r, margin_local = cfg.qc_skeleton_dilation_px, cfg.box_margin
    image_hw_anchor = mask_anchor.shape[:2]
    print(f"    mask {int(mask_anchor.sum())} px  |  score {state.image_score:.4f}"
          + (f"  | _crop {image_hw_anchor[1]}x{image_hw_anchor[0]}" if cfg.crop_anchor else ""))

    # 5. anchor gate + box (empty mask -> flag, stop). The gate scores the anchor
    #    mask in WHATEVER space it lives in (_crop under the crop path, _sam in
    #    legacy); score_anchor is space-agnostic, we just feed it matching coords +
    #    a space-correct contain radius. Still OBSERVATIONAL -> record, don't branch.
    _step(5, "box from mask")
    anchor = score_anchor(
        mask_anchor, prompts_anchor, image_hw_sam=image_hw_anchor,
        contain_radius_px=contain_r,
        min_area_frac=cfg.gate_min_area_frac,
        max_area_frac=cfg.gate_max_area_frac,
        min_largest_cc_frac=cfg.gate_min_largest_cc_frac,
    )
    state.anchor_score = _anchor_score_to_dict(anchor)
    _contained = "n/a" if anchor.contained is None else anchor.contained
    print(f"    anchor gate: {'PASS' if anchor.passed else 'FAIL'}  "
          f"(cc={anchor.n_components} lcc={anchor.largest_cc_frac:.2f} "
          f"area_frac={anchor.area_frac:.5f} contained={_contained})"
          + (f"  reasons: {', '.join(anchor.reasons)}" if anchor.reasons else ""))

    box_local = box_from_mask(mask_anchor, margin=margin_local, image_hw_sam=image_hw_anchor)
    if box_local is None:
        print("    empty anchor mask -> flagging chain for human review")
        state.status = "flagged"
        _finish()
        return state                          # [M4] later: re-pick anchor before flagging
    # map the crop box back to _sam for the video seed (crop path), or use as-is.
    box = box_local if cw is None else cw.box_crop_to_sam(box_local)
    state.prompts.box_sam = np.asarray(box, dtype=np.float32)
    print(f"    box (xyxy, _sam): {state.prompts.box_sam.astype(int).tolist()}")

    # free the image embedding before video propagation (notebook does this).
    image_predictor.reset_predictor()
    if on_video_phase is not None:
        on_video_phase()

    # 6. video frames
    _step(6, "prepare video frames")
    (state.frames_dir, state.frame_to_z,
     state.anchor_frame_idx, state.n_frames) = prepare_video_frames(
        chain, annotate_df, scale=cfg.scale, frames_root=cfg.frames_root,
        anchor_catmaid_z=state.anchor_catmaid_z,
        neuron=state.neuron, chain_idx=state.chain_idx)
    print(f"    {state.n_frames} frames  (anchor frame_idx={state.anchor_frame_idx})")

    # 7. propagate
    _step(7, "propagate (bidirectional)")
    video_segments, frame_conf = propagate(
        video_predictor, state.frames_dir, state.prompts,
        state.anchor_frame_idx, obj_id=state.obj_id,
        seed_negatives=cfg.seed_negatives, subtimings=subtimings)

    # 8. (post-process, then) save at canonical space. Cleanup is a §3a phase
    #    folded in here so it lands BEFORE QC (step 9) reads the masks back.
    _step(8, "save masks")
    if cfg.postprocess_masks:
        n_changed = 0
        for seg in video_segments.values():
            if state.obj_id in seg:
                cleaned = postprocess_mask(
                    seg[state.obj_id],
                    open_px=cfg.postproc_open_px, close_px=cfg.postproc_close_px,
                    keep_largest_cc=cfg.postproc_keep_largest_cc,
                    fill_holes=cfg.postproc_fill_holes)
                n_changed += int(not np.array_equal(cleaned, seg[state.obj_id]))
                seg[state.obj_id] = cleaned
        print(f"    post-processed {n_changed}/{len(video_segments)} masks")
    chain_dir = Path(cfg.output_root) / state.neuron / f"chain_{state.chain_idx:02d}"
    out_dir = chain_dir / "masks"
    save_masks(video_segments, state.frame_to_z, out_dir,
               obj_id=state.obj_id, mask_space_downscale=cfg.save_downscale)

    # 9. QC + flagging (M2): score the run, write qc.csv, set the chain verdict.
    _step(9, "qc + flag")
    # this chain's own skeleton (NOT the whole neuron -> see run_qc docstring)
    chain_node_ids = {str(n) for n in chain["nodes"]}
    skel_chain = annotate_df[
        annotate_df["node_id"].astype(str).isin(chain_node_ids)
    ][["z", "x_tif", "y_tif"]]
    state.qc_summary, state.triage_frames, state.status = run_qc(
        out_dir, skel_chain,
        frame_to_z=state.frame_to_z,
        frame_conf=frame_conf, cfg=cfg,
        qc_csv_path=chain_dir / "qc.csv",
    )
    s = state.qc_summary
    print(f"    {s['n_flagged']}/{s['n_frames']} flagged "
          f"({s['n_queue']} queued)"
          f"({s['flag_rate']:.0%}), {s['n_intervene']} intervene "
          f"-> status '{state.status}'")
    _finish()
    return state