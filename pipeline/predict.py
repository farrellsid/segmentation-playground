"""Anchor + image-mode phase functions: anchor pick, prompts, image predict, box, gate."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sam2_utils import alignment

from .frames import _downscale_image
from .state import AnchorScore, Prompts


def select_anchor(chain: dict, annotate_df: pd.DataFrame) -> tuple[int, int]:
    """Pick the anchor node for a chain and resolve its CATMAID z.

    Currently the mid-node heuristic. Returns (anchor_node_id, anchor_catmaid_z).

    Lift from: 'Load Image' cell (midnode / TARGET_Z).
    not implemented: failed-anchor auto re-pick policy lives here later, not in the driver.
    """
    nodes = chain["nodes"]
    midnode = nodes[len(nodes) // 2]
    z_series = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(midnode), "z"
    ]
    anchor_catmaid_z = int(z_series.item())   # .item() asserts exactly one match
    return midnode, anchor_catmaid_z


def centreline_by_z(chain: dict, annotate_df: pd.DataFrame) -> dict:
    """{catmaid_z: (x_tif, y_tif)} centreline point for each z in the chain's node
    z-range. Uses the chain's own nodes (real and virtual); a z with no node is
    linearly interpolated between the nearest present z's (which is what a virtual
    node already is). Foreign neurons' nodes are never used."""
    ids = {str(n) for n in chain["nodes"]}
    sub = annotate_df[annotate_df["node_id"].astype(str).isin(ids)]
    by_z: dict[int, tuple[float, float]] = {}
    for z, x, y in zip(sub["z"].astype(int), sub["x_tif"].astype(float), sub["y_tif"].astype(float)):
        by_z.setdefault(int(z), (float(x), float(y)))   # first node wins on a shared z
    if not by_z:
        return {}
    present = sorted(by_z)
    out: dict[int, tuple[float, float]] = {}
    for z in range(present[0], present[-1] + 1):
        if z in by_z:
            out[z] = by_z[z]
            continue
        lo = max(p for p in present if p < z)
        hi = min(p for p in present if p > z)
        t = (z - lo) / (hi - lo)
        (x0, y0), (x1, y1) = by_z[lo], by_z[hi]
        out[z] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out


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
    masks and break the regression match. Wire the radius gate in later when QC
    thresholds are being tuned, not here.
    """
    # --- positive: the anchor (mid) node, _tif -> _sam ---
    cell_node = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(anchor_node_id)
    ]
    pos_sam = alignment.tif_to_sam(
        cell_node[["x_tif", "y_tif"]].to_numpy(dtype=float), scale)   # (1, 2)

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

    negnodes_sam = alignment.tif_to_sam(
        z_points[["x_tif", "y_tif"]].to_numpy(dtype=float), scale)   # (M, 2)
    n_neg = min(len(z_points), k_max_neg)
    for i in range(n_neg):
        points.append([float(negnodes_sam[i, 0]), float(negnodes_sam[i, 1])])
        labels.append(0)

    return Prompts(points_sam=np.array(points, dtype=float),
                   labels=np.array(labels, dtype=int))


def _point_in_mask(mask: np.ndarray, x: float, y: float, radius: int) -> bool:
    """True if any foreground pixel lies within `radius` of (x, y).

    Point and mask must share a space (no transform here). Out-of-frame -> False.
    The single neighbourhood-containment test reused by score_anchor (anchor gate)
    and _select_anchor_mask (multimask pick) and matching qc's per-frame probe, so
    "node is inside the mask" means one thing everywhere.
    """
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= yi < h and 0 <= xi < w):
        return False
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    return bool(mask[y0:y1, x0:x1].any())


def _largest_cc_frac(mask: np.ndarray) -> tuple[int, float]:
    """(n_components, largest-CC fraction of foreground) for a bool mask.

    (0, 0.0) when empty. The single-CC health measure generalises the
    largest-component pick already in box_from_mask; reused by score_anchor and
    _select_anchor_mask so the gate and the multimask pick agree.
    """
    from skimage.measure import label as cc_label

    m = np.asarray(mask).astype(bool)
    area = int(m.sum())
    if area == 0:
        return 0, 0.0
    lbl = cc_label(m, connectivity=2)
    sizes = np.bincount(lbl.ravel())[1:]            # drop background (label 0)
    return int(sizes.size), (float(sizes.max() / area) if sizes.size else 0.0)


def _positive_point(prompts: Optional[Prompts]) -> Optional[np.ndarray]:
    """The first positive (anchor/skeleton) prompt point, or None. Mask space."""
    if prompts is None or prompts.points_sam is None:
        return None
    pts = np.asarray(prompts.points_sam, dtype=float)
    pos = pts[np.asarray(prompts.labels) == 1]
    return pos[0] if len(pos) else None


def _negative_points(prompts: Optional[Prompts]) -> np.ndarray:
    """The negative (neighbour) prompt points, (M, 2), or empty. Mask space."""
    if prompts is None or prompts.points_sam is None:
        return np.empty((0, 2), dtype=float)
    pts = np.asarray(prompts.points_sam, dtype=float)
    return pts[np.asarray(prompts.labels) == 0]


def _select_anchor_mask(masks: np.ndarray, scores: np.ndarray, prompts: Optional[Prompts],
                        image_hw: tuple[int, int], *, contain_radius_px: int,
                        area_bounds: tuple[float, float],
                        exclude_neg: bool = False,
                        generous: bool = False) -> tuple[int, np.ndarray, float]:
    """Pick the best of SAM2's multimask candidates for an anchor seed.

    Ranking is lexicographic and *graceful*, it always returns one candidate, so a
    chain with no clean mask still produces a box (and is then caught by the gate /
    empty-mask flag downstream) rather than crashing. Priority order is
    node-containment / plausible-area / single-CC:
      1. contains the positive node      : domain anchor, the mask must sit on the neurite
      2. excludes negatives (exclude_neg) : OPTIONAL, when on, prefer a candidate that
                                            contains none of the negative neighbour nodes,
                                            the anti-bleed pick (a mask that swallows a
                                            neighbour's node is bleeding into it)
      3. plausible area (in area_bounds)  : reject runaway background grabs / empty masks
                                            *before* single-CC, since a runaway grab is
                                            usually one huge clean blob (lcc ~ 1.0) that
                                            would otherwise win on step 4
      4. single-CC health (largest_cc_frac) : one clean blob over fragmented membrane
      5. tiebreak (generous)              : SAM predicted IoU (scores) by default; with
                                            `generous=True`, the LARGER area_frac instead,
                                            among candidates already tied on steps 1-4. The
                                            area gate at step 3 outranks the tiebreak, so a
                                            whole-frame over-cap blob loses to any candidate
                                            that passes the gate, EVEN under `generous`. But
                                            if every candidate exceeds the cap, area_ok ties
                                            at 0 for all of them and generous then picks the
                                            largest (worst) of the failing candidates, a
                                            degenerate slice the merge-metric will surface,
                                            not a case this tiebreak protects against.

    Everything is judged in the space the masks live in (the caller passes matching
    `prompts`, `image_hw`, and `contain_radius_px`), so this is transform-free like
    score_anchor. The chosen mask still only sources the video-seed *box*; the
    positive seed point is unchanged, so a multimask pick never moves the seed point.
    With `exclude_neg=False` and `generous=False` the key is byte-identical to the
    original ranking.
    Returns (best_idx, mask_bool, score).
    """
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float).ravel()
    H, W = int(image_hw[0]), int(image_hw[1])
    frame_px = H * W
    min_af, max_af = area_bounds
    pos = _positive_point(prompts)
    negs = _negative_points(prompts) if exclude_neg else np.empty((0, 2), dtype=float)

    best_idx, best_key = 0, None
    for i in range(masks.shape[0]):
        m = masks[i]
        area_frac = (int(m.sum()) / frame_px) if frame_px else 0.0
        contained = pos is not None and _point_in_mask(m, float(pos[0]), float(pos[1]), contain_radius_px)
        _, lcc = _largest_cc_frac(m)
        score = float(scores[i]) if i < scores.size else 0.0
        area_ok = int(min_af <= area_frac <= max_af)
        tiebreak = area_frac if generous else score
        if exclude_neg:
            no_neg = int(not any(
                _point_in_mask(m, float(nx), float(ny), contain_radius_px) for nx, ny in negs))
            key = (int(contained), no_neg, area_ok, lcc, tiebreak)
        else:
            key = (int(contained), area_ok, lcc, tiebreak)
        if best_key is None or key > best_key:
            best_idx, best_key = i, key
    return best_idx, masks[best_idx], (float(scores[best_idx]) if best_idx < scores.size else 0.0)


def image_predict(image_predictor, image_sam: np.ndarray, prompts: Prompts, *,
                  multimask: bool = False, select_contain_radius_px: int = 0,
                  select_area_bounds: tuple[float, float] = (0.0, 1.0),
                  select_exclude_neg: bool = False,
                  select_generous: bool = False,
                  ) -> tuple[np.ndarray, float, np.ndarray]:
    """Run image-mode SAM2 on the anchor frame.

    Returns (mask bool HxW, score, logits) in whatever space `image_sam` is.

    `multimask=False` (default) is the single-mask path, `multimask_output=False`,
    the regression baseline, exactly reproduces the notebook. `multimask=True`
    asks SAM2 for its 3 candidate masks and auto-selects one via `_select_anchor_mask`.
    This is near-free: SAM2's mask decoder
    *always* computes all 3 candidates regardless of the flag (it only slices the
    output, see sam2/modeling/sam/mask_decoder.py), and the heavy image-encoder
    `set_image` runs once either way; the only added work is scoring 3 masks on CPU.
    The selection params are only consulted when `multimask=True`. `select_generous`
    forwards to `_select_anchor_mask`'s `generous` tiebreak (prefer the larger
    gate-passing candidate over the higher-scoring one); default False is unchanged.

    A `prompts.box_sam` (xyxy in `image_sam` space) is forwarded as SAM2's `box` seed,
    so a human-drawn box (GUI) shapes the mask alongside any points. Points are passed
    as None when there are none, so a box-only seed is valid. This is a no-op for the
    batch, whose box_sam is None here (box_from_mask runs AFTER this call).

    Lift from: 'Image Prediction' cell.
    The GUI refinement loop wraps this call (re-predict on each point/box edit).
    """
    import torch

    pts = np.asarray(prompts.points_sam, dtype=float)
    labs = np.asarray(prompts.labels, dtype=int)
    has_pts = len(pts) > 0
    box = None if prompts.box_sam is None else np.asarray(prompts.box_sam, dtype=float)
    with torch.inference_mode():
        image_predictor.set_image(image_sam)
        masks, scores, logits = image_predictor.predict(
            point_coords=pts if has_pts else None,
            point_labels=labs if has_pts else None,
            box=box,
            multimask_output=multimask,
        )
    if not multimask:
        return masks[0].astype(bool), float(scores[0]), logits
    best, mask_b, score = _select_anchor_mask(
        masks, scores, prompts, masks.shape[1:],
        contain_radius_px=select_contain_radius_px, area_bounds=select_area_bounds,
        exclude_neg=select_exclude_neg, generous=select_generous)
    return mask_b.astype(bool), score, logits[best:best + 1]


def box_from_mask(mask_sam: np.ndarray, *, margin: int, margin_frac: float = 0.0,
                  image_hw_sam: tuple[int, int]) -> Optional[np.ndarray]:
    """Largest connected component -> xyxy box (+margin), clipped to image, _sam space.

    Returns the box, or None if the mask is empty -> None is the signal to flag the
    chain for human review rather than feed garbage into propagation.

    ``margin`` is a fixed pad in mask-space px (the historical behaviour). ``margin_frac``
    (>0) adds a pad scaled to the box's own size (``round(margin_frac * max(w, h))`` of
    the largest-CC bbox, applied per side) and the effective pad is the LARGER of the two.
    Rationale (seed ablation): when the anchor mask *under*-fills the cell, a
    fixed 10px box doesn't enclose the whole process, so propagation can't recover the
    missing extent; a size-relative pad widens the box in proportion to the object, which
    is the cheap way to keep the box seed competitive with the (curated) mask seed.

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
    w = int(xs.max()) - int(xs.min()) + 1
    h = int(ys.max()) - int(ys.min()) + 1
    pad = max(int(margin), int(round(margin_frac * max(w, h))))   # frac scales with object size
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W_sam - 1)
    y1 = min(int(ys.max()) + pad, H_sam - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def score_anchor(mask_sam: np.ndarray, prompts: Prompts, *,
                 image_hw_sam: tuple[int, int],
                 contain_radius_px: int,
                 min_area_frac: float,
                 max_area_frac: float,
                 min_largest_cc_frac: float) -> AnchorScore:
    """Score the raw image-mode anchor mask for propagation-readiness, in _sam space.

    Called *before* box_from_mask (it judges the raw multi-blob mask, not the
    largest-CC box) and before propagation, so a bad anchor costs one frame's
    compute instead of a wasted ~300-frame propagate. This is the
    *scoring* half only: it is pure (reads the mask + prompts, writes nothing) and
    decides nothing -> the gate that escalates prompts / re-picks the node / blocks
    propagation consumes this verdict in the next increment.

    Lift/parallel: the containment probe is the same neighbourhood test as
    qc.compute_metrics (so anchor- and per-frame containment agree); the single-CC
    measure generalises the largest-component pick already in box_from_mask.
    """
    H_sam, W_sam = image_hw_sam
    frame_px = int(H_sam) * int(W_sam)
    m = np.asarray(mask_sam).astype(bool)
    area = int(m.sum())
    area_frac = (area / frame_px) if frame_px else 0.0

    # --- containment: does the mask cover the positive (anchor) prompt point? ---
    # Tri-state, matching qc.skeleton_contained: an empty mask with a node present
    # is an explicit miss (False); no positive point at all is an abstain (None).
    # (_point_in_mask returns False for a node that maps outside the frame.)
    pos = _positive_point(prompts)                 # the anchor (skeleton) node, mask space
    if pos is None:
        contained: Optional[bool] = None
    elif area == 0:
        contained = False
    else:
        contained = _point_in_mask(m, float(pos[0]), float(pos[1]), contain_radius_px)

    # --- single-CC health ---
    n_cc, largest_cc_frac = _largest_cc_frac(m)

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
                        *, scale: int, crop_size_tif: int, crop_scale: int,
                        cw: Optional["alignment.CropWindow"] = None,
                        multimask: bool = False, select_contain_radius_px: int = 0,
                        select_area_bounds: tuple[float, float] = (0.0, 1.0),
                        select_exclude_neg: bool = False,
                        select_generous: bool = False,
                        ) -> tuple[np.ndarray, float, "alignment.CropWindow", "Prompts"]:
    """Image-mode anchor prediction on a high-res crop (default path).

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

    `multimask` + the `select_*` params forward straight to image_predict's
    multimask auto-select. They must already be in _crop space: the
    caller rescales `select_contain_radius_px` by scale/crop_scale, and
    `select_area_bounds` are frame-fraction bounds the crop config already tunes,
    so selection scores in the same _crop space as the mask.

    A prebuilt ``cw`` (tier-2 per-chain window) is used as-is: the image phase then
    runs in the SAME crop the whole chain propagates in, so the seed needs no
    _crop->_sam remap. When ``cw`` is None (tier-1 default) a fresh ``crop_size_tif``
    window is centred on the anchor node.

    Returns (mask_crop bool HxW, score, cw, prompts_crop).
    """
    from sam2_utils import alignment

    if cw is None:
        # tier-1: a fresh window centred on the anchor node (_tif).
        node = annotate_df.loc[annotate_df["node_id"].astype(str) == str(anchor_node_id)]
        node_xy_tif = node[["x_tif", "y_tif"]].to_numpy(dtype=float)[0]
        cw = alignment.CropWindow.around_node(
            node_xy_tif, size_tif=crop_size_tif, image_hw_tif=full_hw,
            crop_scale=crop_scale, sam_scale=scale)

    crop_full = image_full[cw.slice_tif()]                  # _tif window
    crop_img = _downscale_image(crop_full, cw.crop_scale)   # _crop input image (window governs scale)
    H_crop, W_crop = crop_img.shape[:2]

    # _sam prompt points -> _tif -> _crop. Keep all positives; drop out-of-window negatives.
    pts_sam = np.asarray(prompts_sam.points_sam, dtype=float)
    labels = np.asarray(prompts_sam.labels, dtype=int)
    pts_crop = cw.tif_to_crop(alignment.sam_to_tif(pts_sam, scale))   # _sam -> _tif -> _crop
    in_bounds = ((pts_crop[:, 0] >= 0) & (pts_crop[:, 0] < W_crop) &
                 (pts_crop[:, 1] >= 0) & (pts_crop[:, 1] < H_crop))
    keep = in_bounds | (labels == 1)
    prompts_crop = Prompts(points_sam=pts_crop[keep], labels=labels[keep])  # NB: _crop coords

    mask_crop, score, _logits = image_predict(
        image_predictor, crop_img, prompts_crop, multimask=multimask,
        select_contain_radius_px=select_contain_radius_px,
        select_area_bounds=select_area_bounds, select_exclude_neg=select_exclude_neg,
        select_generous=select_generous)
    return mask_crop, score, cw, prompts_crop
