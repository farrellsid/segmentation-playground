"""Mask output: save propagated masks at the canonical space, and model-free cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_masks(video_segments: dict[int, dict[int, np.ndarray]],
               frame_to_z: dict[int, int], out_dir: Path, *,
               obj_id: int, mask_space_downscale: int) -> int:
    """Write one 0/255 uint8 PNG per frame at the canonical mask space.

    Returns count written. Files are `mask_<catmaid_z:04d>.png`, single-channel,
    255 = inside the neurite, 0 = background -> the notebook's exact format, so the
    masks are directly viewable AND pixel-comparable to the notebook output (the
    done-check). qc._load_binary reads `arr > 0`, so this stays fully
    compatible with compute_metrics.

    Why NOT qc.save_masks here: that writer stores uint16 *instance labels*
    (foreground pixel value == obj_id). For a single object obj_id is 1, and value
    1 in a 16-bit image is visually indistinguishable from black -> it looks empty
    and is destroyed by any 16->8-bit conversion, which is exactly the "empty
    masks" confusion. Instance-label encoding is a multi-object concern; adopt it
    later when aggregating several objects per neuron, not now.

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
    holey while true neurite borders are smooth -> cleanup priors are safe.
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

    m = np.asarray(mask_sam)
    if m.ndim == 3:                       # SAM2 yields (1, H, W); squeeze the channel axis,
        m = m[0]                          # else binary_opening on the singleton axis empties it
    m = m.astype(bool)
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


def _as_bool_2d(mask: np.ndarray) -> np.ndarray:
    """Coerce a mask to a 2D bool array, squeezing SAM2's (1, H, W) channel axis."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    return m.astype(bool)


def remove_small_islands(mask: np.ndarray, *, min_size: int = 64,
                         connectivity: int = 2) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` px, keeping ALL larger ones.

    Unlike ``postprocess_mask``'s ``keep_largest_cc`` (which keeps a single component), this
    keeps every component at or above the size floor, so a legitimate second cross-section of
    the cell survives while detached specks are removed. ``connectivity`` 2 is 8-neighbour.
    Empty in (or ``min_size`` <= 1) -> empty/unchanged out."""
    from skimage.morphology import remove_small_objects
    m = _as_bool_2d(mask)
    if not m.any() or int(min_size) <= 1:
        return m
    # skimage 0.26's max_size removes components <= its value, so max_size = min_size - 1
    # reproduces "remove components smaller than min_size" (keep >= min_size).
    return remove_small_objects(m, max_size=int(min_size) - 1, connectivity=int(connectivity))


def fill_small_holes(mask: np.ndarray, *, area_threshold: int = 64,
                     connectivity: int = 1) -> np.ndarray:
    """Fill background holes smaller than ``area_threshold`` px inside the mask, leaving
    larger holes intact.

    Unlike ``postprocess_mask``'s ``fill_holes`` (which fills every hole), a genuine large
    cavity is preserved. Empty in (or ``area_threshold`` <= 1) -> unchanged out."""
    from skimage.morphology import remove_small_holes
    m = _as_bool_2d(mask)
    if not m.any() or int(area_threshold) <= 1:
        return m
    # max_size fills holes <= its value, so max_size = area_threshold - 1 fills holes
    # strictly smaller than area_threshold.
    return remove_small_holes(m, max_size=int(area_threshold) - 1, connectivity=int(connectivity))


def smooth_edges(mask: np.ndarray, *, radius: int = 2) -> np.ndarray:
    """Smooth a frayed / netty mask boundary by a morphological closing then opening with a
    disk of ``radius`` px: the closing bridges thin gaps in the mesh, the opening shaves thin
    protrusions. Topology-light, but keep ``radius`` small or thin neurites erode away.
    ``radius`` <= 0 or empty in -> returned unchanged."""
    from skimage.morphology import closing, disk, opening
    m = _as_bool_2d(mask)
    if radius <= 0 or not m.any():
        return m
    footprint = disk(int(radius))
    return opening(closing(m, footprint), footprint).astype(bool)
