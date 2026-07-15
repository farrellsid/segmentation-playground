"""Target-worm skeleton merge-metric (roadmap Phase 0).

A ground-truth-free severe-bleed / dropout scorer for a run's RAW per-chain masks,
scored against the target worm's own CATMAID skeletons. See docs/explanation/roadmap.md
section 5 Phase 0 for the scope: this is a severe-merge floor, not an ERL benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import alignment, config


def load_node_table() -> pd.DataFrame:
    """The CATMAID node table with x_tif / y_tif attached (mirrors batch.py setup)."""
    df = pd.read_csv(config.CSV_PATH)
    xy_tif = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
    df["x_tif"] = xy_tif[:, 0]
    df["y_tif"] = xy_tif[:, 1]
    return df


def nodes_by_z(annotate_df: pd.DataFrame, scale: int
               ) -> dict[int, list[tuple[float, float, str, str]]]:
    """Group nodes by z with coordinates on the run's _sam grid (x_tif / scale).

    Matches chain_masks_in_sam's grid so a node maps into a returned mask directly.
    """
    out: dict[int, list[tuple[float, float, str, str]]] = {}
    for z, x_tif, y_tif, cell, nid in zip(
        annotate_df["z"].astype(int), annotate_df["x_tif"].astype(float),
        annotate_df["y_tif"].astype(float), annotate_df["cell_name"].astype(str),
        annotate_df["node_id"].astype(str),
    ):
        out.setdefault(int(z), []).append((x_tif / scale, y_tif / scale, cell, nid))
    return out


def own_contained(mask: np.ndarray, x0: int, y0: int,
                  node_xy: tuple[float, float], radius: int) -> bool:
    """True if the mask covers its own node (grid coords), accounting for the
    mask's (x0, y0) grid offset."""
    x, y = node_xy
    return pipeline._point_in_mask(mask, x - x0, y - y0, radius)


def foreign_hits(mask: np.ndarray, x0: int, y0: int,
                 nodes: list[tuple[float, float, str, str]],
                 own_neuron: str, radius: int) -> list[str]:
    """node_ids of nodes belonging to a DIFFERENT neuron that fall inside the mask."""
    hits: list[str] = []
    for x, y, cell, nid in nodes:
        if cell == own_neuron:
            continue
        if pipeline._point_in_mask(mask, x - x0, y - y0, radius):
            hits.append(nid)
    return hits


def score_chain(chain_dir: Path, neuron: str,
                nodes_by_z: dict[int, list[tuple[float, float, str, str]]],
                radius: int) -> list[dict]:
    """Per-z merge/dropout records for one chain, from its RAW saved masks."""
    masks = pipeline.chain_masks_in_sam(Path(chain_dir))
    recs: list[dict] = []
    for z, (mask, x0, y0) in sorted(masks.items()):
        nodes = nodes_by_z.get(int(z), [])
        own = [(x, y) for (x, y, cell, _nid) in nodes if cell == neuron]
        own_ok = any(own_contained(mask, x0, y0, xy, radius) for xy in own) if own else False
        fids = foreign_hits(mask, x0, y0, nodes, neuron, radius)
        recs.append({
            "z": int(z),
            "own_contained": bool(own_ok),
            "n_foreign": len(fids),
            "foreign_ids": fids,
            "empty": bool(not mask.any()),
        })
    return recs
