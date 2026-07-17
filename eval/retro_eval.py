"""Retro-eval: one comparison table across every run we have.

For each merged run tree, aggregates three families of numbers so methods can be
compared objectively on one row each:

  * merge-metric (eval.merge_metric): Phase-0 foreign-node + dropout (always,
    GT-free, node-anchored) and, with --membrane, the Phase-2 membrane detectors
    (mild bleed, spanning, boundary-on-membrane, underfill);
  * compute, from the run's _timing.csv (total GPU-seconds, mean per chain, peak VRAM);
  * legacy QC flags, from the run's _manifest.csv (flagged chains, frame flag rate).

Each tree is scored at ITS OWN _sam scale (from _run_meta.json), because the runs
differ: fullres is scale 1, wholeimg_s4 is scale 4, the rest scale 8. Membrane
numbers are only comparable within one scale, so restrict --membrane to same-scale
trees when comparing.

Usage:
  py -3 -m eval.retro_eval --root <tree> [--root ...] [--membrane] [--out <dir>]
  py -3 -m eval.retro_eval --glob "F:/.../resolution_experiments/*_merged" [--membrane]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import merge_metric as mm


def _timing_row(root: Path) -> dict:
    p = root / "_timing.csv"
    if not p.exists():
        return {"n_timed": 0, "gpu_min_total": None, "s_per_chain": None, "peak_vram_gb": None}
    t = pd.read_csv(p)
    tot = pd.to_numeric(t.get("t_total"), errors="coerce")
    vram = pd.to_numeric(t.get("peak_vram_gb"), errors="coerce")
    return {
        "n_timed": int(tot.notna().sum()),
        "gpu_min_total": float(np.nansum(tot) / 60.0),
        "s_per_chain": float(np.nanmean(tot)),
        "peak_vram_gb": float(np.nanmax(vram)) if vram.notna().any() else None,
    }


def _flag_row(root: Path) -> dict:
    p = root / "_manifest.csv"
    if not p.exists():
        return {"chains_flagged": None, "chains_clean": None, "frame_flag_rate": None}
    m = pd.read_csv(p)
    n_flagged = pd.to_numeric(m.get("n_flagged"), errors="coerce").fillna(0)
    n_frames = pd.to_numeric(m.get("n_frames"), errors="coerce").fillna(0)
    status = m.get("status")
    total_frames = float(n_frames.sum())
    return {
        "chains_flagged": int((status == "flagged").sum()) if status is not None else None,
        "chains_clean": int((n_flagged == 0).sum()),
        "frame_flag_rate": float(n_flagged.sum() / total_frames) if total_frames else None,
    }


def _meta_row(root: Path) -> dict:
    p = root / "_run_meta.json"
    if not p.exists():
        return {"preset": root.name, "scale": None, "image_size": None}
    m = json.loads(p.read_text())
    res = m.get("resolution", {})
    return {
        "preset": m.get("preset", root.name),
        "scale": res.get("scale"),
        "image_size": res.get("image_size_actual"),
    }


MERGE_KEYS = ("n_chains", "n_frames", "foreign_frame_rate", "dropout_rate",
              "total_foreign_nodes", "mild_bleed_rate", "spanning_merge_rate",
              "mean_boundary_on_membrane", "mean_underfill_fraction")


def eval_tree(root: Path, annotate_df: pd.DataFrame, *, membrane: bool,
              min_scale: int = 4) -> dict:
    """One row per tree. Compute + flags + metadata are always captured (they read
    only the CSVs, no masks). The merge-metric is attempted separately and may be
    skipped or fail without losing the rest of the row (full-res masks are 9k x 9k,
    so loading a whole chain at once needs a lot of RAM). Trees below min_scale are
    skipped by default; lower min_scale on a big-memory machine (e.g. CCDB) to
    include them."""
    root = Path(root)
    meta = _meta_row(root)
    try:
        scale = mm.run_scale(root)
    except (FileNotFoundError, KeyError, ValueError):
        scale = meta.get("scale")
    row = {"method": root.name, **meta, "merge_note": ""}
    row.update(_timing_row(root))
    row.update(_flag_row(root))
    row.update({k: None for k in MERGE_KEYS})

    if scale is None:
        row["merge_note"] = "no scale"
        return row
    if int(scale) < min_scale:
        row["merge_note"] = f"merge-metric skipped (scale {scale} < min_scale {min_scale}; needs more RAM)"
        return row
    try:
        src = mm.MembraneSource(int(scale)) if membrane else None
        _per, summ = mm.score_run(root, annotate_df=annotate_df,
                                  membrane_source=src, scale=int(scale))
        if summ["n_frames"] == 0:
            row["merge_note"] = "no scored frames"
            return row
        row.update({k: summ[k] for k in MERGE_KEYS})
    except Exception as e:                                  # noqa: BLE001 (report, don't crash the sweep)
        row["merge_note"] = f"merge-metric error: {type(e).__name__}"
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retro-eval: one comparison table across runs.")
    ap.add_argument("--root", action="append", default=[], dest="roots")
    ap.add_argument("--glob", default=None, help="glob of tree roots (quote it)")
    ap.add_argument("--membrane", action="store_true", help="include the Phase-2 membrane pass (slow)")
    ap.add_argument("--min-scale", type=int, default=4,
                    help="skip the merge-metric for trees below this _sam scale (full-res masks "
                         "need lots of RAM); use 1 on a big-memory machine, 999 for timing+flags only")
    ap.add_argument("--out", default=None, help="dir to write retro_eval.csv (default: cwd)")
    args = ap.parse_args(argv)

    roots = list(args.roots)
    if args.glob:
        roots += sorted(globmod.glob(args.glob))
    if not roots:
        ap.error("pass --root and/or --glob")

    annotate_df = mm.load_node_table()
    rows = []
    for r in roots:
        print(f"[retro] scoring {Path(r).name} (membrane={args.membrane}) ...", flush=True)
        try:
            row = eval_tree(Path(r), annotate_df, membrane=args.membrane, min_scale=args.min_scale)
        except Exception as e:                             # noqa: BLE001 (flaky drive: log, keep going)
            print(f"[retro]   ERROR on {Path(r).name}: {type(e).__name__}: {e}")
            rows.append({"method": Path(r).name, "merge_note": f"row error: {type(e).__name__}"})
            continue
        rows.append(row)
        ffr = row["foreign_frame_rate"]
        summ = (f"foreign_rate={ffr:.3f} dropout={row['dropout_rate']:.3f}"
                if ffr is not None else f"[{row['merge_note']}]")
        gm = row["gpu_min_total"]
        print(f"[retro]   done {Path(r).name}: {summ} "
              f"gpu_min={gm:.1f}" if gm is not None else f"[retro]   done {Path(r).name}: {summ}")

    df = pd.DataFrame(rows)
    out_dir = Path(args.out) if args.out else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "retro_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[retro] wrote {csv_path}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
