"""
sweep_dilation.py, threshold-sensitivity sweep over qc_skeleton_dilation_px.

Read-only post-hoc analysis over a FINISHED batch run (batch.py output_root).
It does NOT re-segment and it does NOT touch _manifest.csv (avoiding the
mixed-threshold trap: the manifest is per-chain append-mode; mixing dilations
into it would silently blend configs).

Why this is cheap *and* exact
-----------------------------
In ``qc.compute_metrics`` only ``skeleton_contained`` depends on the dilation
radius. The composite flag rule is additive:

    flag_count(d) = fc_other + [ skeleton_contained(d) == False ]

where ``fc_other`` (the pred_iou / area_ratio / temporal_iou terms) is invariant
in d. So we read ``fc_other`` straight from each chain's qc.csv, recompute ONLY
containment at every candidate radius from the on-disk masks + this chain's
skeleton, and recombine. One mask read per frame total; all radii are evaluated
in the inner loop.

The per-chain-skeleton trap
---------------------------
Containment must use THIS CHAIN's skeleton nodes, not the whole neuron's (the
AVAL 100%-flag bug). We rebuild the per-chain skeleton exactly as
``pipeline.run_chain`` does (chains.json ``nodes`` -> ``annotate_df`` filter on
``node_id``) so the recomputed tri-state matches the run.

Success marker (what to check)
------------------------------
At d == the run-time ``qc_skeleton_dilation_px`` (read from each chain's
state.json), the recomputed flag_count must equal that chain's qc.csv
flag_count for every frame (``baseline_match``). A mismatch means the
skeleton/coordinate reconstruction is wrong and nothing downstream is
trustworthy, the same role "reproduce AVAL pixel-for-pixel" plays.

This measures *sensitivity*, not *correctness*: it shows how many flags dilation
removes and whether they are structural (won't heal) or tolerance (heal at some
radius), but it cannot say which removed flags were genuine errors. That waits
for collected labels.

Outputs (written under output_root, prefixed to sort beside _manifest.csv)
--------------------------------------------------------------------------
  _dilation_sweep.csv           one row per dilation, the sensitivity curve
  _dilation_sweep_by_chain.csv  one row per (chain, dilation), which chains flip

Usage
-----
  python sweep_dilation.py
  # or: from sweep_dilation import run_sweep
  #     run_sweep(OUTPUT_ROOT, CSV_PATH, CHAINS_PATH)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sam2_utils import qc, alignment, config   # qc helpers reused; no model / torch needed

# Data + output locations live once in sam2_utils.config (shared with batch.py).
CSV_PATH    = config.CSV_PATH
CHAINS_PATH = config.CHAINS_PATH
OUTPUT_ROOT = config.OUTPUT_ROOT

# Candidate radii (mask-pixel units, _sam space). The run-time dilation of each
# chain is unioned in automatically so the baseline row always exists.
DILATIONS: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 8, 10)

COMPLETE_STATUSES = {"done", "flagged"}   # only these chains have masks/ + qc.csv


# ---------------------------------------------------------------------------
# Inputs that must be reloaded (the per-chain skeleton can't come from qc.csv)
# ---------------------------------------------------------------------------

def _load_annotate(csv_path: Path) -> pd.DataFrame:
    """CATMAID CSV -> annotate_df with x_tif/y_tif (same as batch._build_session)."""
    df = pd.read_csv(csv_path)
    xy_tif = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
    df["x_tif"] = xy_tif[:, 0]
    df["y_tif"] = xy_tif[:, 1]
    return df


def _chain_nodes_index(chains: Sequence[dict]) -> dict[tuple[str, int], list]:
    """{(neuron, chain_idx): nodes}. Reproduces batch.enumerate_chains' indexing
    (positional within each neuron's chain list), so keys match the on-disk
    chain_<idx:02d> folders."""
    by_neuron: "OrderedDict[str, list]" = OrderedDict()
    for ch in chains:
        by_neuron.setdefault(ch["cell_name"], []).append(ch)
    idx: dict[tuple[str, int], list] = {}
    for neuron, chs in by_neuron.items():
        for i, ch in enumerate(chs):
            idx[(neuron, i)] = ch["nodes"]
    return idx


# ---------------------------------------------------------------------------
# Per-chain sweep
# ---------------------------------------------------------------------------

def _sweep_one_chain(chain_dir: Path, nodes: Sequence,
                     annotate_df: pd.DataFrame, dilations: Sequence[int]) -> Optional[dict]:
    """Re-score one finished chain across dilation radii. Returns a counts dict,
    or None if the chain lacks the artifacts to score (anchor-flagged early
    return has neither masks/ nor qc.csv)."""
    state_path = chain_dir / "state.json"
    qc_csv     = chain_dir / "qc.csv"
    masks_dir  = chain_dir / "masks"
    if not (state_path.exists() and qc_csv.exists() and masks_dir.exists()):
        return None

    cfg = json.loads(state_path.read_text())["config"]
    save_ds = int(cfg["save_downscale"])
    run_d   = int(cfg["qc_skeleton_dilation_px"])
    ar_lo, ar_hi = float(cfg["qc_area_ratio_bounds"][0]), float(cfg["qc_area_ratio_bounds"][1])
    ti_min  = float(cfg["qc_temporal_iou_min"])
    pi_min  = float(cfg["qc_pred_iou_min"])
    int_to_flag = int(cfg.get("qc_intervene_to_flag_chain", 1))

    grid = sorted(set(dilations) | {run_d})

    # Invariant partial flag count (the non-containment terms), straight from the
    # run's own signal columns + thresholds. NaN handling mirrors qc.compute_metrics:
    # pred_iou/temporal_iou fillna(1.0); a NaN area_ratio compares False -> 0.
    qdf = pd.read_csv(qc_csv).set_index("z")
    pi = qdf["pred_iou"].fillna(1.0)
    ti = qdf["temporal_iou"].fillna(1.0)
    ar = qdf["area_ratio"]
    fc_other = (
        (pi < pi_min).astype(int)
        + ((ar < ar_lo) | (ar > ar_hi)).astype(int)
        + (ti < ti_min).astype(int)
    ).to_dict()
    fc_run = qdf["flag_count"].astype(int).to_dict()   # for the baseline check

    # This chain's skeleton only (the per-chain-skeleton filter), in full-res tif coords.
    node_ids = {str(n) for n in nodes}
    skel = annotate_df[
        annotate_df["node_id"].astype(str).isin(node_ids)
    ][["z", "x_tif", "y_tif"]]

    counts = {d: dict(n_noskel=0, n_struct=0, n_flag=0, n_int=0) for d in grid}
    n_frames = n_assess = 0
    baseline_ok = True

    for z, p in qc._iter_mask_paths(masks_dir):
        if z not in fc_other:        # masks vs qc.csv drift, shouldn't happen
            continue
        n_frames += 1
        m = qc._load_binary(p)
        area = int(m.sum())
        xy = qc._skeleton_xy_for_z(skel, int(z))

        # Reproduce compute_metrics' tri-state structure exactly. Only the
        # area>0/in-frame branch depends on the radius.
        if xy is None:
            contained = {d: None for d in grid}      # no chain node at z -> abstain (NaN)
            structural = False
        else:
            n_assess += 1
            sx_i = int(round(xy[0] / save_ds))
            sy_i = int(round(xy[1] / save_ds))
            if area == 0:
                contained = {d: False for d in grid}  # node exists, mask empty
                structural = True
            elif not (0 <= sy_i < m.shape[0] and 0 <= sx_i < m.shape[1]):
                contained = {d: False for d in grid}  # node maps outside the frame
                structural = True
            else:
                contained = {d: qc._node_contained(m, sx_i, sy_i, d) for d in grid}
                structural = False                    # tolerance-sensitive

        fco = int(fc_other[z])
        for d in grid:
            noskel = (contained[d] is False)          # NaN/True must NOT count
            fc = fco + int(noskel)
            if noskel:
                counts[d]["n_noskel"] += 1
                if structural:
                    counts[d]["n_struct"] += 1
            if fc >= 1:
                counts[d]["n_flag"] += 1
            if fc >= 2:
                counts[d]["n_int"] += 1

        # baseline self-consistency: recomputed flag_count at the run dilation
        # must reproduce what the run actually wrote.
        if fco + int(contained[run_d] is False) != fc_run.get(z, -999):
            baseline_ok = False

    return dict(n_frames=n_frames, n_assess=n_assess, counts=counts,
                int_to_flag=int_to_flag, baseline_ok=baseline_ok)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_sweep(output_root: Path | str, csv_path: Path | str, chains_path: Path | str,
              dilations: Sequence[int] = DILATIONS) -> pd.DataFrame:
    """Walk a finished output_root, re-score every completed chain across
    `dilations`, and write the two sweep CSVs. Returns the aggregate DataFrame."""
    output_root = Path(output_root)
    manifest_path = output_root / "_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no _manifest.csv under {output_root}, run batch.py first")

    annotate_df = _load_annotate(Path(csv_path))
    with open(chains_path) as f:
        nodes_index = _chain_nodes_index(json.load(f))

    manifest = pd.read_csv(manifest_path)
    todo = manifest[manifest["status"].isin(COMPLETE_STATUSES)]
    grid = sorted(set(dilations) | {3})   # 3 is the usual default; per-chain run_d is added inside

    agg = {d: dict(n_noskel=0, n_struct=0, n_flag=0, n_int=0, n_chains_flagged=0) for d in grid}
    by_chain_rows: list[dict] = []
    tot_frames = tot_assess = n_chains = 0
    no_artifacts = 0
    baseline_failures: list[tuple[str, int]] = []

    for _, row in todo.iterrows():
        neuron, idx = str(row["neuron"]), int(row["chain_idx"])
        nodes = nodes_index.get((neuron, idx))
        if nodes is None:
            print(f"[sweep] WARN no chains.json entry for {neuron}/chain_{idx:02d}, skipped")
            continue
        chain_dir = output_root / neuron / f"chain_{idx:02d}"
        res = _sweep_one_chain(chain_dir, nodes, annotate_df, grid)
        if res is None:
            no_artifacts += 1   # e.g. anchor-flagged early return: no masks/ or qc.csv
            continue

        n_chains += 1
        tot_frames += res["n_frames"]
        tot_assess += res["n_assess"]
        if not res["baseline_ok"]:
            baseline_failures.append((neuron, idx))

        for d in grid:
            c = res["counts"].get(d)
            if c is None:        # d not in this chain's grid (shouldn't happen; grid superset)
                continue
            flagged = c["n_int"] >= res["int_to_flag"]
            agg[d]["n_noskel"] += c["n_noskel"]
            agg[d]["n_struct"] += c["n_struct"]
            agg[d]["n_flag"]   += c["n_flag"]
            agg[d]["n_int"]    += c["n_int"]
            agg[d]["n_chains_flagged"] += int(flagged)
            by_chain_rows.append(dict(
                neuron=neuron, chain_idx=idx, dilation_px=d,
                n_frames=res["n_frames"], n_assessable=res["n_assess"],
                n_noskel=c["n_noskel"], n_noskel_structural=c["n_struct"],
                n_flag=c["n_flag"], n_intervene=c["n_int"],
                flag_rate=round(c["n_flag"] / res["n_frames"], 4) if res["n_frames"] else 0.0,
                status="flagged" if flagged else "done",
                baseline_match=res["baseline_ok"],
            ))

    agg_rows = [dict(
        dilation_px=d, n_chains=n_chains, n_frames=tot_frames, n_assessable=tot_assess,
        n_noskel=agg[d]["n_noskel"], n_noskel_structural=agg[d]["n_struct"],
        n_noskel_tolerance=agg[d]["n_noskel"] - agg[d]["n_struct"],
        n_flag=agg[d]["n_flag"],
        flag_rate=round(agg[d]["n_flag"] / tot_frames, 4) if tot_frames else 0.0,
        n_intervene=agg[d]["n_int"],
        intervene_rate=round(agg[d]["n_int"] / tot_frames, 4) if tot_frames else 0.0,
        n_chains_flagged=agg[d]["n_chains_flagged"],
    ) for d in grid]
    agg_df = pd.DataFrame(agg_rows)
    by_chain_df = pd.DataFrame(by_chain_rows)

    agg_df.to_csv(output_root / "_dilation_sweep.csv", index=False)
    by_chain_df.to_csv(output_root / "_dilation_sweep_by_chain.csv", index=False)

    _print_summary(agg_df, n_chains, tot_frames, no_artifacts, baseline_failures, output_root)
    return agg_df


def _print_summary(agg_df: pd.DataFrame, n_chains: int, tot_frames: int,
                   no_artifacts: int, baseline_failures: list, output_root: Path) -> None:
    print(f"\n[sweep] scored {n_chains} chains / {tot_frames} frames "
          f"({no_artifacts} completed chains had no masks/qc.csv, skipped)")

    # 1. baseline self-consistency (the load-bearing marker)
    if baseline_failures:
        print(f"[sweep] BASELINE MISMATCH on {len(baseline_failures)} chain(s): "
              f"{baseline_failures[:10]}{' ...' if len(baseline_failures) > 10 else ''}")
        print("[sweep]   -> skeleton/coordinate reconstruction is wrong; do NOT trust the curve.")
    else:
        print("[sweep] baseline OK: recomputed flag_count matches qc.csv at run-time dilation "
              "for every chain.")

    # 2. monotonicity + structural plateau
    s = agg_df.sort_values("dilation_px")
    noskel = s["n_noskel"].tolist()
    monotone = all(b <= a for a, b in zip(noskel, noskel[1:]))
    floor = int(s["n_noskel_structural"].iloc[0])
    print(f"[sweep] noskel monotonic non-increasing: {'OK' if monotone else 'WARN (bug?)'}; "
          f"structural floor (dilation cannot heal) = {floor}")

    # 3. the curve
    print("[sweep] sensitivity curve (n_noskel = structural + tolerance):")
    for _, r in s.iterrows():
        print(f"    d={int(r['dilation_px']):>2}  "
              f"noskel={int(r['n_noskel']):>5} "
              f"(struct={int(r['n_noskel_structural']):>4} tol={int(r['n_noskel_tolerance']):>5})  "
              f"flag_rate={r['flag_rate']:.3f}  "
              f"intervene_rate={r['intervene_rate']:.3f}  "
              f"chains_flagged={int(r['n_chains_flagged'])}")
    print(f"[sweep] wrote {output_root / '_dilation_sweep.csv'} "
          f"and {output_root / '_dilation_sweep_by_chain.csv'}")


def main() -> None:
    run_sweep(OUTPUT_ROOT, CSV_PATH, CHAINS_PATH)


if __name__ == "__main__":
    main()
