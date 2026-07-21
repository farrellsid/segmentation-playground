"""Per-frame scoring metric (F2): grade a frame's labelled instance masks with the same
GT-free primitives merge_metric + membrane use, plus a pre-resolution overlap scalar.
Torch-free; doubles as the tuner objective and the metric-guided selector's basis."""
from __future__ import annotations

import numpy as np

import pipeline
from sam2_utils import membrane as mb


def _contains(mask, xy, radius):
    return pipeline._point_in_mask(mask, float(xy[0]), float(xy[1]), radius)


def pairwise_overlap_fraction(masks) -> float:
    """Sum of pairwise pixel intersections over total area, for an arbitrary list of bool
    masks: the "fight for pixels" diagnostic. Near 0 when the masks are already disjoint
    (e.g. a resolved label map's per-cell slices), so it is informative only when fed
    pre-resolution masks. run_perframe.py's two segment_frame_* functions both compute
    this on their pre-resolution masks and use it to override score_frame's own (post-
    resolution, so near-0-by-construction) overlap_fraction; see their docstrings."""
    ms = [m.astype(bool) for m in masks]
    total_area = float(sum(int(m.sum()) for m in ms)) or 1.0
    overlap = 0
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            overlap += int((ms[i] & ms[j]).sum())
    return float(overlap / total_area)


def score_frame(cell_masks, node_index, membrane_map=None, *, radius=3, tau=0.5) -> dict:
    """cell_masks: {cell_name: bool mask}. node_index: [(x, y, cell, node_id), ...] for this
    frame (F1). membrane_map: float [0,1] frame map or None (membrane columns then None)."""
    cells = list(cell_masks)
    per = []
    for cell in cells:
        m = cell_masks[cell].astype(bool)
        own = [(x, y) for (x, y, c, _n) in node_index if c == cell]
        foreign = [(x, y) for (x, y, c, _n) in node_index if c != cell]
        own_ok = any(_contains(m, xy, radius) for xy in own) if own else False
        n_foreign = sum(_contains(m, xy, radius) for xy in foreign)
        row = {"cell": cell, "own_contained": bool(own_ok), "n_foreign": int(n_foreign),
               "area": int(m.sum())}
        if membrane_map is not None and m.any():
            sp, bled = mb.spanning_membrane(m, membrane_map, tau=tau)
            row["spanning"] = bool(sp)
            row["boundary_on_membrane"] = float(mb.boundary_on_membrane(m, membrane_map, tau=tau))
            row["underfill"] = float(mb.underfill_fraction(m, membrane_map, tau=tau))
        per.append(row)

    n = len(per)
    ms = [cell_masks[c].astype(bool) for c in cells]
    have_mem = any("boundary_on_membrane" in r for r in per)
    summary = {
        "n_cells": n,
        "own_coverage": float(np.mean([r["own_contained"] for r in per])) if n else 0.0,
        "foreign_frame_rate": float(np.mean([r["n_foreign"] > 0 for r in per])) if n else 0.0,
        "total_foreign": int(sum(r["n_foreign"] for r in per)),
        "overlap_fraction": pairwise_overlap_fraction(ms),
        "mean_boundary_on_membrane": (float(np.mean([r["boundary_on_membrane"] for r in per
                                                     if "boundary_on_membrane" in r]))
                                      if have_mem else None),
        "spanning_rate": (float(np.mean([r["spanning"] for r in per if "spanning" in r]))
                          if have_mem else None),
        "mean_underfill": (float(np.mean([r["underfill"] for r in per if "underfill" in r]))
                           if have_mem else None),
        "per_cell": per,
    }
    return summary


def objective(score: dict) -> float:
    """Scalar the AMG tuner maximises, from a score_frame dict. Rewards own-node coverage
    and boundary-on-membrane; penalises foreign bleed, spanning, and pre-resolution overlap.
    Membrane terms are dropped (treated as 0 contribution) when None so a no-membrane run
    still ranks by coverage/bleed. Weights are a starting point, tune-able."""
    bo = score.get("mean_boundary_on_membrane") or 0.0
    sp = score.get("spanning_rate") or 0.0
    return (1.0 * score["own_coverage"]
            + 0.5 * bo
            - 1.0 * score["foreign_frame_rate"]
            - 0.5 * sp
            - 0.5 * score["overlap_fraction"])
