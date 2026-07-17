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
from sam2_utils import alignment, config, membrane

DEFAULT_RADIUS = 3


class MembraneSource:
    """Supplies membrane maps for a run's masks, cropped to each mask's _sam
    window. Reads the raw EM per z via pipeline.load_frame_sam (the FrameStore
    seam), caches the grayscale _sam frame per z, and runs membrane_map on the
    window. Returns None when the EM for z is unavailable or the window is out
    of bounds, so the scorer degrades to the Phase-0 (node-only) metric."""

    def __init__(self, scale: int, *, sigmas=membrane.DEFAULT_SIGMAS, frame_store=None):
        self.scale = int(scale)
        self.sigmas = sigmas
        self.frame_store = frame_store
        self._gray: dict[int, np.ndarray | None] = {}

    def _frame_gray(self, z: int):
        if z in self._gray:
            return self._gray[z]
        try:
            img, _ = pipeline.load_frame_sam(
                int(z), scale=self.scale, frame_store=self.frame_store)
            gray = (img.mean(axis=2) if img.ndim == 3 else img).astype(np.float32)
        except Exception:
            gray = None
        self._gray[z] = gray
        return gray

    def map_for(self, z: int, x0: int, y0: int, h: int, w: int):
        gray = self._frame_gray(int(z))
        if gray is None:
            return None
        H, W = gray.shape[:2]
        if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            return None
        crop = gray[y0:y0 + h, x0:x0 + w]
        if crop.size == 0 or crop.shape != (h, w):
            return None
        return membrane.membrane_map(crop, sigmas=self.sigmas)


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
                radius: int, membrane_source=None,
                tau: float = membrane.DEFAULT_TAU,
                tol: int = membrane.DEFAULT_TOL) -> list[dict]:
    """Per-z merge/dropout records for one chain, from its RAW saved masks.

    When membrane_source is given, each record also carries the membrane-aware
    detector scalars (spanning_merge, bled_fraction, boundary_on_membrane,
    underfill_fraction); they are None when the source has no map for that
    frame."""
    masks = pipeline.chain_masks_in_sam(Path(chain_dir))
    recs: list[dict] = []
    for z, (mask, x0, y0) in sorted(masks.items()):
        nodes = nodes_by_z.get(int(z), [])
        own = [(x, y) for (x, y, cell, _nid) in nodes if cell == neuron]
        own_ok = any(own_contained(mask, x0, y0, xy, radius) for xy in own) if own else False
        fids = foreign_hits(mask, x0, y0, nodes, neuron, radius)
        rec = {
            "z": int(z),
            "own_contained": bool(own_ok),
            "n_foreign": len(fids),
            "foreign_ids": fids,
            "empty": bool(not mask.any()),
            "spanning_merge": None,
            "bled_fraction": None,
            "boundary_on_membrane": None,
            "underfill_fraction": None,
        }
        if membrane_source is not None:
            h, w = mask.shape[:2]
            mem = membrane_source.map_for(int(z), int(x0), int(y0), h, w)
            if mem is not None:
                spanning, frac = membrane.spanning_membrane(mask, mem, tau=tau)
                rec["spanning_merge"] = bool(spanning)
                rec["bled_fraction"] = float(frac)
                rec["boundary_on_membrane"] = float(
                    membrane.boundary_on_membrane(mask, mem, tau=tau, tol=tol))
                rec["underfill_fraction"] = float(
                    membrane.underfill_fraction(mask, mem, tau=tau))
        recs.append(rec)
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
              radius: int = DEFAULT_RADIUS, membrane_source="auto",
              tau: float = membrane.DEFAULT_TAU, tol: int = membrane.DEFAULT_TOL
              ) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-chain records, write CSV, return per-frame DataFrame and summary.

    The n_chains count includes only chains that produced at least one scored frame;
    a chain with no frames (empty masks/ directory) is not counted. No CSV is written
    if the run has zero scored frames.

    membrane_source: "auto" builds a MembraneSource for the run scale; None
    disables the membrane pass (Phase-0-only); or pass an object with map_for()
    for tests. When membrane scalars are absent, the membrane summary keys are
    None and the Phase-0 keys are unchanged."""
    root = Path(root)
    scale = run_scale(root)
    if annotate_df is None:
        annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, scale)
    if membrane_source == "auto":
        membrane_source = MembraneSource(scale)

    rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius,
                                   membrane_source, tau=tau, tol=tol):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)

    per = pd.DataFrame(rows)
    n_frames = len(per)
    have_mem = bool(n_frames) and per["spanning_merge"].notna().any()
    summary = {
        "n_chains": int(per[["neuron", "chain_idx"]].drop_duplicates().shape[0]) if n_frames else 0,
        "n_frames": int(n_frames),
        "foreign_frame_rate": float((per["n_foreign"] > 0).mean()) if n_frames else 0.0,
        "dropout_rate": float((per["empty"] | ~per["own_contained"]).mean()) if n_frames else 0.0,
        "total_foreign_nodes": int(per["n_foreign"].sum()) if n_frames else 0,
        "mild_bleed_rate": None,
        "spanning_merge_rate": None,
        "mean_boundary_on_membrane": None,
        "mean_underfill_fraction": None,
    }
    if have_mem:
        scored = per[per["spanning_merge"].notna()]
        span = scored["spanning_merge"].astype(bool)
        summary["spanning_merge_rate"] = float(span.mean())
        summary["mild_bleed_rate"] = float((span & (scored["n_foreign"] == 0)).mean())
        summary["mean_boundary_on_membrane"] = float(scored["boundary_on_membrane"].mean())
        summary["mean_underfill_fraction"] = float(scored["underfill_fraction"].mean())
    if n_frames:
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        per_out.to_csv(root / "_merge_metric.csv", index=False)
    return per, summary


def format_summary(name: str, s: dict) -> str:
    line = (f"{name:<28} chains={s['n_chains']:>4} frames={s['n_frames']:>6} "
            f"foreign_frame_rate={s['foreign_frame_rate']:.3f} "
            f"dropout_rate={s['dropout_rate']:.3f} "
            f"total_foreign={s['total_foreign_nodes']:>5}")
    if s.get("mild_bleed_rate") is not None:
        line += (f" | mild_bleed_rate={s['mild_bleed_rate']:.3f} "
                 f"spanning_merge_rate={s['spanning_merge_rate']:.3f} "
                 f"boundary_on_membrane={s['mean_boundary_on_membrane']:.3f} "
                 f"underfill={s['mean_underfill_fraction']:.3f}")
    return line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Target-worm skeleton merge-metric (roadmap Phase 0 + Phase 2).")
    ap.add_argument("--root", action="append", required=True, dest="roots",
                    help="a merged run tree; repeat to compare runs")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    ap.add_argument("--no-membrane", action="store_true",
                    help="skip the Phase-2 membrane detectors (Phase-0-only, no EM reads)")
    ap.add_argument("--tau", type=float, default=membrane.DEFAULT_TAU,
                    help="membrane threshold on the normalised [0,1] map")
    ap.add_argument("--tol", type=int, default=membrane.DEFAULT_TOL,
                    help="px tolerance for boundary-on-membrane")
    args = ap.parse_args(argv)

    annotate_df = load_node_table()
    for root in args.roots:
        src = None if args.no_membrane else MembraneSource(run_scale(root))
        _per, summ = score_run(root, annotate_df=annotate_df, radius=args.radius,
                               membrane_source=src, tau=args.tau, tol=args.tol)
        print(format_summary(Path(root).name, summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
