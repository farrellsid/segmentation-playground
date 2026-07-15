"""Target-worm skeleton merge-metric (roadmap Phase 0).

A ground-truth-free severe-bleed / dropout scorer for a run's RAW per-chain masks,
scored against the target worm's own CATMAID skeletons. See docs/explanation/roadmap.md
section 5 Phase 0 for the scope: this is a severe-merge floor, not an ERL benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import alignment, config

DEFAULT_RADIUS = 3


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


def run_scale(root: Path) -> int:
    """Read resolution.scale from <root>/_run_meta.json, check it matches save_downscale."""
    meta = json.loads((Path(root) / "_run_meta.json").read_text())
    res = meta.get("resolution", {})
    scale = int(res["scale"])
    sd = int(res.get("save_downscale", scale))
    if sd != scale:
        raise ValueError(
            f"{root}: scale ({scale}) != save_downscale ({sd}); the node grid "
            "assumption does not hold, extend the scorer before trusting it.")
    return scale


def score_run(root, annotate_df: pd.DataFrame | None = None,
              radius: int = DEFAULT_RADIUS) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-chain records, write CSV, return per-frame DataFrame and summary dict.

    The n_chains count includes only chains that produced at least one scored frame;
    a chain with no frames (empty masks/ directory) is not counted. No CSV is written
    if the run has zero scored frames."""
    root = Path(root)
    scale = run_scale(root)
    if annotate_df is None:
        annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, scale)

    rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)

    per = pd.DataFrame(rows)
    n_frames = len(per)
    summary = {
        "n_chains": int(per[["neuron", "chain_idx"]].drop_duplicates().shape[0]) if n_frames else 0,
        "n_frames": int(n_frames),
        "foreign_frame_rate": float((per["n_foreign"] > 0).mean()) if n_frames else 0.0,
        "dropout_rate": float((per["empty"] | ~per["own_contained"]).mean()) if n_frames else 0.0,
        "total_foreign_nodes": int(per["n_foreign"].sum()) if n_frames else 0,
    }
    if n_frames:
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        per_out.to_csv(root / "_merge_metric.csv", index=False)
    return per, summary


def format_summary(name: str, s: dict) -> str:
    return (f"{name:<28} chains={s['n_chains']:>4} frames={s['n_frames']:>6} "
            f"foreign_frame_rate={s['foreign_frame_rate']:.3f} "
            f"dropout_rate={s['dropout_rate']:.3f} "
            f"total_foreign={s['total_foreign_nodes']:>5}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Target-worm skeleton merge-metric (roadmap Phase 0).")
    ap.add_argument("--root", action="append", required=True, dest="roots",
                    help="a merged run tree; repeat to compare runs")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    args = ap.parse_args(argv)

    annotate_df = load_node_table()
    for root in args.roots:
        _per, summ = score_run(root, annotate_df=annotate_df, radius=args.radius)
        print(format_summary(Path(root).name, summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
