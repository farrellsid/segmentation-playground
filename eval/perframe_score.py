"""Per-frame scoring metric (F2): grade a frame's labelled instance masks with the same
GT-free primitives merge_metric + membrane use, plus a pre-resolution overlap scalar.
Torch-free; doubles as the tuner objective and the metric-guided selector's basis."""
from __future__ import annotations

import numpy as np

import pipeline
from sam2_utils import membrane as mb


def _contains(mask, xy, radius):
    return pipeline._point_in_mask(mask, float(xy[0]), float(xy[1]), radius)


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
    total_area = float(sum(r["area"] for r in per)) or 1.0
    # pairwise overlap fraction (pre-resolution fight for pixels)
    overlap = 0
    ms = [cell_masks[c].astype(bool) for c in cells]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            overlap += int((ms[i] & ms[j]).sum())
    have_mem = any("boundary_on_membrane" in r for r in per)
    summary = {
        "n_cells": n,
        "own_coverage": float(np.mean([r["own_contained"] for r in per])) if n else 0.0,
        "foreign_frame_rate": float(np.mean([r["n_foreign"] > 0 for r in per])) if n else 0.0,
        "total_foreign": int(sum(r["n_foreign"] for r in per)),
        "overlap_fraction": float(overlap / total_area),
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
