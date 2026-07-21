"""Per-frame segmentation primitives (torch-free): node index, overlap resolution,
metric-guided candidate selection, and AMG-to-node matching. The SAM2-touching runner
lives in run_perframe.py. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from sam2_utils import membrane as _mb


def nodes_in_frame(annotate_df, catmaid_z: int, scale: int
                   ) -> list[tuple[float, float, str, str]]:
    """Every node at catmaid_z across all neurons, as (x_sam, y_sam, cell_name, node_id).
    Coords are x_tif/scale (the _sam grid merge_metric uses)."""
    z = annotate_df["z"].astype(int)
    sub = annotate_df[z == int(catmaid_z)]
    out = []
    for _, r in sub.iterrows():
        out.append((float(r["x_tif"]) / scale, float(r["y_tif"]) / scale,
                    str(r["cell_name"]), str(r["node_id"])))
    return out


def resolve_overlaps_argmax(masks, node_xy, membrane_map=None) -> np.ndarray:
    """Assign one label per pixel. Uncontested pixels keep their only claimant. A pixel
    claimed by several masks goes to the claimant whose seed node is nearest (Euclidean).
    membrane_map is accepted for signature parity with the watershed resolver and is not
    used by the nearest-node rule. Returns an int label map (0 background, i+1 = masks[i])."""
    if not masks:
        raise ValueError("no masks")
    h, w = masks[0].shape
    stack = np.stack([m.astype(bool) for m in masks], axis=0)   # (K, H, W)
    count = stack.sum(axis=0)
    lab = np.zeros((h, w), dtype=np.int32)
    # uncontested: exactly one claimant -> that label (argmax of the one True plane)
    single = count == 1
    lab[single] = stack[:, single].argmax(axis=0) + 1
    # contested: nearest seed node among claimants
    ys, xs = np.where(count > 1)
    if ys.size:
        seeds = np.asarray(node_xy, dtype=float)                # (K, 2) as (x, y)
        for y, x in zip(ys, xs):
            claim = np.where(stack[:, y, x])[0]
            d = (seeds[claim, 0] - x) ** 2 + (seeds[claim, 1] - y) ** 2
            lab[y, x] = int(claim[int(np.argmin(d))]) + 1
    return lab


def resolve_overlaps_watershed(masks, node_xy, membrane_map) -> np.ndarray:
    """Seeded watershed on the membrane map: each seed node is a marker, the membrane map
    is the elevation (walls at ridges), flooding restricted to the union of the masks.
    Returns an int label map (0 background, i+1 = masks[i])."""
    from skimage.segmentation import watershed
    if not masks:
        raise ValueError("no masks")
    h, w = masks[0].shape
    union = np.zeros((h, w), bool)
    for m in masks:
        union |= m.astype(bool)
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (x, y) in enumerate(node_xy):
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < h and 0 <= xi < w:
            markers[yi, xi] = i + 1
    elevation = (membrane_map if membrane_map is not None
                 else np.zeros((h, w), np.float32)).astype(np.float32)
    lab = watershed(elevation, markers=markers, mask=union)
    return lab.astype(np.int32)


def _rim(mask: np.ndarray) -> np.ndarray:
    """1-px inner boundary of a bool mask (shared by tests + boundary scoring)."""
    return mask & ~ndi.binary_erosion(mask)


def _contains(mask: np.ndarray, xy, radius: int) -> bool:
    import pipeline
    return pipeline._point_in_mask(mask, float(xy[0]), float(xy[1]), radius)


def select_by_metric(candidates, node_xy, foreign_xy, membrane_map, *,
                     radius: int = 3, tau: float = 0.5) -> int:
    """Index of the candidate that best satisfies the F2 composite: must contain node_xy
    and no foreign node; among those, maximise boundary_on_membrane, then minimise the
    spanning-membrane bled_fraction. Returns -1 if none contain the node without a foreign
    hit."""
    best, best_key = -1, None
    for i, m in enumerate(candidates):
        m = m.astype(bool)
        if not _contains(m, node_xy, radius):
            continue
        if any(_contains(m, f, radius) for f in foreign_xy):
            continue
        bo = _mb.boundary_on_membrane(m, membrane_map, tau=tau)
        _, bled = _mb.spanning_membrane(m, membrane_map, tau=tau)
        key = (bo, -bled)                       # higher boundary, lower bled
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def match_amg_to_nodes(amg_masks, node_index, membrane_map, *,
                       radius: int = 3, tau: float = 0.5):
    """Assign each node its AMG mask (the containing mask best on the F2 composite, via
    select_by_metric with the OTHER nodes as foreign), label it by cell_name, and return
    the leftover (unmatched) masks as unlabelled competitors."""
    labels: dict[str, np.ndarray] = {}
    used = set()
    for (x, y, cell, _nid) in node_index:
        foreign = [(fx, fy) for (fx, fy, fc, _f) in node_index if fc != cell]
        idx = select_by_metric(amg_masks, (x, y), foreign, membrane_map,
                               radius=radius, tau=tau)
        if idx >= 0:
            labels[cell] = amg_masks[idx].astype(bool)
            used.add(idx)
    leftover = [m.astype(bool) for i, m in enumerate(amg_masks) if i not in used]
    return labels, leftover
