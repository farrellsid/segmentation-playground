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
