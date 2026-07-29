"""Target-worm skeleton merge-metric (roadmap Phase 0 + Phase 2).

A ground-truth-free bleed / dropout scorer for a run's RAW per-chain masks, scored
against the target worm's own CATMAID skeletons (Phase 0: foreign-node containment,
a severe-merge floor) plus an optional membrane-aware pass (Phase 2: mild bleed and
underfill via sam2_utils.membrane). See docs/explanation/roadmap.md section 5 for the
scope of each phase.
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


def z_transitions(masks: dict[int, tuple[np.ndarray, int, int]]) -> list[dict]:
    """Per-transition z-to-z consistency records for one chain's raw masks.

    For each pair of masks adjacent in sorted z order (masks is {z: (mask, x0, y0)}
    from pipeline.chain_masks_in_sam), computes IoU and centroid drift in a shared
    coordinate frame built from their own _sam-grid offsets: two masks can have
    different local shapes (a tier-2 crop window can move or resize between frames),
    so this pastes both onto a canvas sized to their combined bounding box rather
    than comparing local arrays directly.

    A transition touching an empty mask (matching the empty = not mask.any()
    convention score_chain already uses) gets iou=None, centroid_drift_px=None:
    dropout is the existing empty/dropout_rate signal's job, not a consistency
    reading, so it is not conflated with a low-IoU score here."""
    zs = sorted(masks.keys())
    out: list[dict] = []
    for z_from, z_to in zip(zs, zs[1:]):
        mask_a, x0_a, y0_a = masks[z_from]
        mask_b, x0_b, y0_b = masks[z_to]
        rec = {"z_from": int(z_from), "z_to": int(z_to), "gap": int(z_to - z_from),
               "iou": None, "centroid_drift_px": None}
        if not mask_a.any() or not mask_b.any():
            out.append(rec)
            continue
        h_a, w_a = mask_a.shape[:2]
        h_b, w_b = mask_b.shape[:2]
        x_min = min(x0_a, x0_b)
        y_min = min(y0_a, y0_b)
        x_max = max(x0_a + w_a, x0_b + w_b)
        y_max = max(y0_a + h_a, y0_b + h_b)
        canvas_a = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_b = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_a[y0_a - y_min:y0_a - y_min + h_a, x0_a - x_min:x0_a - x_min + w_a] = mask_a
        canvas_b[y0_b - y_min:y0_b - y_min + h_b, x0_b - x_min:x0_b - x_min + w_b] = mask_b
        intersection = int((canvas_a & canvas_b).sum())
        union = int((canvas_a | canvas_b).sum())
        rec["iou"] = intersection / union if union > 0 else 0.0

        ys_a, xs_a = np.where(mask_a)
        ys_b, xs_b = np.where(mask_b)
        cx_a, cy_a = float(xs_a.mean()) + x0_a, float(ys_a.mean()) + y0_a
        cx_b, cy_b = float(xs_b.mean()) + x0_b, float(ys_b.mean()) + y0_b
        rec["centroid_drift_px"] = float(np.hypot(cx_b - cx_a, cy_b - cy_a))
        out.append(rec)
    return out


def summarize_z_consistency(transitions: list[dict], *, low_iou_threshold: float = 0.5) -> dict:
    """Aggregate z_transitions records (one chain's, or a whole run's concatenated).

    n_dropout_transitions counts transitions with iou=None separately from the
    IoU/drift means, so a chain that scores well only because most transitions were
    skipped by dropout does not look falsely consistent. frac_low_iou is restricted
    to gap==1 transitions (a gap>1 transition spans real missed frames, not a
    consistency failure, and would understate the run's true low-IoU rate if mixed
    in)."""
    n = len(transitions)
    dropout = [t for t in transitions if t["iou"] is None]
    scored = [t for t in transitions if t["iou"] is not None]
    gap1_scored = [t for t in scored if t["gap"] == 1]
    return {
        "n_transitions": n,
        "n_dropout_transitions": len(dropout),
        "mean_z2z_iou": float(np.mean([t["iou"] for t in scored])) if scored else None,
        "mean_centroid_drift_px": (
            float(np.mean([t["centroid_drift_px"] for t in scored])) if scored else None),
        "frac_gap1_transitions": (
            sum(1 for t in transitions if t["gap"] == 1) / n) if n else None,
        "frac_low_iou": (
            sum(1 for t in gap1_scored if t["iou"] < low_iou_threshold) / len(gap1_scored)
        ) if gap1_scored else None,
    }


def summarize(per: pd.DataFrame) -> dict:
    """Aggregate a per-frame merge-metric DataFrame into the summary dict.

    Split out of score_run so the sharded-eval concat step (eval.concat_merge_shards)
    can recompute the same summary from stitched shard CSVs without re-reading masks.
    The membrane keys stay None unless the frames carry membrane scalars (a non-NaN
    spanning_merge column)."""
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
    return summary


def z_csv_path_for(dest: Path, root: Path) -> Path:
    """Derive the z-consistency CSV path from the per-frame CSV path ``dest``.

    Mirrors the shard naming convention a sharded eval task uses for the per-frame
    CSV: ``.../_merge_metric.shard_3.csv`` -> ``.../_z_consistency.shard_3.csv``, so
    concurrent array tasks never write the same z-consistency path either. Falls back
    to ``root / "_z_consistency.csv"`` when ``dest``'s filename does not follow the
    ``"_merge_metric"`` naming convention (a custom out_csv, or the common case where
    dest is already the canonical ``root / "_merge_metric.csv"``, which also contains
    the substring and substitutes correctly)."""
    dest = Path(dest)
    if "_merge_metric" in dest.name:
        return dest.with_name(dest.name.replace("_merge_metric", "_z_consistency"))
    return Path(root) / "_z_consistency.csv"


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
              tau: float = membrane.DEFAULT_TAU, tol: int = membrane.DEFAULT_TOL,
              scale: int | None = None, neurons=None, out_csv=None,
              low_iou_threshold: float = 0.5
              ) -> tuple[pd.DataFrame, dict]:
    """Aggregate per-chain records, write CSV, return per-frame DataFrame and summary.

    The n_chains count includes only chains that produced at least one scored frame;
    a chain with no frames (empty masks/ directory) is not counted. No CSV is written
    if the run has zero scored frames.

    scale: the _sam grid scale for this tree. Defaults to reading it from the
    tree's _run_meta.json (run_scale); pass an explicit value to score a tree
    that has no _run_meta.json (e.g. a merged shard tree).

    neurons: when given, score ONLY these neuron dirs (a subset of the tree). This
    is how the sharded eval array (cluster/run_eval_array.sh) splits one tree across
    CPUs: each task scores its own neurons into its own shard CSV. None scores every
    neuron dir, the whole-tree default.

    out_csv: where to write the per-frame CSV. None writes the canonical
    <root>/_merge_metric.csv; a shard task passes an explicit path
    (<root>/_merge_metric.shard_<i>.csv) so parallel tasks never clobber each other
    or the final file, which concat_merge_shards stitches together afterwards. The
    z-consistency CSV mirrors this: its path is derived from out_csv (see
    z_csv_path_for), so a shard task's z-consistency write is shard-scoped too and
    never races another task's.

    membrane_source: "auto" builds a MembraneSource for the run scale; None
    disables the membrane pass (Phase-0-only); or pass an object with map_for()
    for tests. When membrane scalars are absent, the membrane summary keys are
    None and the Phase-0 keys are unchanged."""
    root = Path(root)
    scale = run_scale(root) if scale is None else int(scale)
    if annotate_df is None:
        annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, scale)
    if membrane_source == "auto":
        membrane_source = MembraneSource(scale)
    want = set(neurons) if neurons is not None else None

    rows: list[dict] = []
    z_rows: list[dict] = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        if want is not None and neuron not in want:
            continue
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            cidx = int(chain_dir.name.split("_")[-1])
            for rec in score_chain(chain_dir, neuron, nbz, radius,
                                   membrane_source, tau=tau, tol=tol):
                rec.update(neuron=neuron, chain_idx=cidx)
                rows.append(rec)
            chain_masks = pipeline.chain_masks_in_sam(chain_dir)
            for trec in z_transitions(chain_masks):
                trec.update(neuron=neuron, chain_idx=cidx)
                z_rows.append(trec)

    per = pd.DataFrame(rows)
    summary = summarize(per)
    z_summary = summarize_z_consistency(z_rows, low_iou_threshold=low_iou_threshold)
    summary.update(z_summary)
    dest = Path(out_csv) if out_csv is not None else root / "_merge_metric.csv"
    if len(per):
        per_out = per.copy()
        per_out["foreign_ids"] = per_out["foreign_ids"].apply(lambda ids: ";".join(ids))
        per_out.to_csv(dest, index=False)
    if z_rows:
        z_dest = z_csv_path_for(dest, root)
        pd.DataFrame(z_rows).to_csv(z_dest, index=False)
    return per, summary


def _fmt_or_na(v, spec: str) -> str:
    """Format v with spec, or "n/a" when v is None.

    mean_z2z_iou being non-None does not guarantee every other z-consistency field
    is: a heavily z-sparse tree (every scored transition has gap != 1) can have a
    real mean_z2z_iou while frac_low_iou's gap-1-only denominator is empty, giving
    frac_low_iou=None. Each z field is formatted independently through this rather
    than assuming the mean_z2z_iou gate covers the whole segment."""
    return format(v, spec) if v is not None else "n/a"


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
    if s.get("mean_z2z_iou") is not None:
        line += (f" | mean_z2z_iou={s['mean_z2z_iou']:.3f} "
                 f"mean_centroid_drift_px={s['mean_centroid_drift_px']:.2f} "
                 f"frac_low_iou={_fmt_or_na(s.get('frac_low_iou'), '.3f')} "
                 f"frac_gap1={_fmt_or_na(s.get('frac_gap1_transitions'), '.3f')}")
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
    ap.add_argument("--low-iou-threshold", type=float, default=0.5,
                    help="z-to-z IoU below this counts toward frac_low_iou")
    ap.add_argument("--scale", type=int, default=None,
                    help="override the _sam grid scale (for merged trees with no _run_meta.json)")
    ap.add_argument("--neurons", default=None,
                    help="score only these neuron dirs (comma- or space-separated); "
                         "used by the sharded eval array to split one tree across CPUs")
    ap.add_argument("--out-csv", default=None,
                    help="write the per-frame CSV here instead of <root>/_merge_metric.csv "
                         "(a shard task passes <root>/_merge_metric.shard_<i>.csv). Only "
                         "valid with a single --root.")
    args = ap.parse_args(argv)

    neurons = None
    if args.neurons is not None:
        neurons = [n for n in args.neurons.replace(",", " ").split() if n]
    if args.out_csv is not None and len(args.roots) != 1:
        ap.error("--out-csv is only valid with exactly one --root")

    annotate_df = load_node_table()
    for root in args.roots:
        scale = args.scale if args.scale is not None else run_scale(root)
        src = None if args.no_membrane else MembraneSource(scale)
        _per, summ = score_run(root, annotate_df=annotate_df, radius=args.radius,
                               membrane_source=src, tau=args.tau, tol=args.tol,
                               scale=scale, neurons=neurons, out_csv=args.out_csv,
                               low_iou_threshold=args.low_iou_threshold)
        print(format_summary(Path(root).name, summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
