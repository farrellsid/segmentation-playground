"""Per-frame segmentation primitives (torch-free): node index, overlap resolution,
metric-guided candidate selection, and AMG-to-node matching. The SAM2-touching runner
lives in run_perframe.py. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md
"""
from __future__ import annotations


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
