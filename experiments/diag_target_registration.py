"""diag_target_registration.py — is the sensory-ablated-dauer (proj 336) single global
affine z-stable, or does it drift across the stack?

Context: SEM-Dauer 1 (the cross-worm GT) needed a *per-section* affine
because its VAST masks/EM were realigned section-by-section vs the pre-realignment
skeleton trace. The target worm uses ONE global affine (`config.M_AFFINE/T_AFFINE`,
fit at CATMAID z=1293 == the FIRST stack slice). This script asks the cautious
question the GT result raised: does that single-z affine hold across all ~338 z, or
does it drift toward the far end (z far from 1293)?

Method (no GT masks needed; uses the pipeline's own outputs):
For every saved frame of every output chain, compare
  * node_sam   = the chain's skeleton node at that z, placed EXACTLY as the pipeline
                 placed it: catmaid_to_tif(x,y) / scale  (so this measures the same
                 registration the pipeline used), and
  * centroid_sam = the saved mask's centroid (qc.csv centroid_x/centroid_y),
and take the offset vector node - centroid.

NB: ~⅔ of the chains here are tier-2 (`chain_crop`) so their qc centroid is in the
per-chain crop space `_pcrop`, NOT `_sam`. We map it back via the chain's persisted
`crop_window` (tif = origin_tif + p·crop_scale → sam = tif/scale) before differencing.
Forgetting this compares two different frames and fabricates a huge fake "drift".

A single chain's offset mixes registration error with propagation drift. To ISOLATE
registration we use cross-chain *coherence* per z:
propagation drift points in random per-chain directions and CANCELS in the mean;
a registration error is shared by every chain on that slice and SURVIVES the mean.
So the signature of registration drift is a per-z MEAN offset vector whose magnitude
grows coherently with |z - 1293|. A flat, ~zero mean (with nonzero per-chain spread)
says the global affine is fine and the spread is just propagation/thinness.

Run:  py -3 -u -m experiments.diag_target_registration
Torch-free (pandas+numpy). Reads config.OUTPUT_ROOT (E:) + in-repo skeleton CSV/chains.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_utils import config, alignment

SCALE = 8            # pipeline default (save_downscale == scale == 8)
Z_FIT = 1293         # CATMAID z the global affine was fit at (== first stack slice)


def _node_lookup() -> pd.DataFrame:
    """node_id(str) -> registered _sam x/y + catmaid z, for every skeleton node."""
    df = pd.read_csv(config.CSV_PATH)
    df["node_id"] = df["node_id"].astype(str)
    xy = alignment.catmaid_to_tif(df["x"].to_numpy(), df["y"].to_numpy())
    df["xs"] = xy[:, 0] / SCALE
    df["ys"] = xy[:, 1] / SCALE
    df["zc"] = df["z"].round().astype(int)
    return df.set_index("node_id")[["xs", "ys", "zc"]]


def _centroid_to_sam(cx, cy, crop_window):
    """qc centroid -> _sam. Identity for _sam chains; for tier-2 chains the centroid
    is in `_pcrop`, so map tif = origin_tif + p*crop_scale, then /SCALE."""
    if not crop_window:
        return cx, cy
    ox, oy = crop_window["origin_tif"]
    cs = crop_window["crop_scale"]
    return (cx * cs + ox) / SCALE, (cy * cs + oy) / SCALE


def collect(root: Path) -> pd.DataFrame:
    nodes = _node_lookup()
    chains_all = json.load(open(config.CHAINS_PATH))

    rows = []
    for neuron_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        neuron = neuron_dir.name
        cell_chains = [c for c in chains_all if c["cell_name"] == neuron]
        for chain_dir in sorted(neuron_dir.glob("chain_*")):
            idx = int(chain_dir.name.split("_")[1])
            if idx >= len(cell_chains):
                continue
            qc_path = chain_dir / "qc.csv"
            if not qc_path.exists():
                continue
            qc = pd.read_csv(qc_path)
            crop_window = None
            st_path = chain_dir / "state.json"
            if st_path.exists():
                crop_window = json.load(open(st_path)).get("crop_window")
            # registered nodes of THIS chain, grouped by catmaid z -> mean _sam pos
            ids = [str(n) for n in cell_chains[idx]["nodes"]]
            sub = nodes.reindex(ids).dropna()
            if sub.empty:
                continue
            node_by_z = sub.groupby("zc")[["xs", "ys"]].mean()
            for _, r in qc.iterrows():
                z = int(r["z"])
                if z not in node_by_z.index or pd.isna(r.get("centroid_x")):
                    continue
                cx, cy = _centroid_to_sam(float(r["centroid_x"]),
                                          float(r["centroid_y"]), crop_window)
                nx, ny = node_by_z.loc[z, "xs"], node_by_z.loc[z, "ys"]
                rows.append({
                    "neuron": neuron, "chain": idx, "z": z,
                    "offx": nx - cx, "offy": ny - cy,
                    "dist": np.hypot(nx - cx, ny - cy),
                    "zoff": z - Z_FIT,
                    "tier2": crop_window is not None,
                    "contained": r.get("skeleton_contained"),
                })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("no rows collected"); return
    print(f"\n=== target-worm registration drift check ===")
    print(f"frames: {len(df)}  chains: {df.groupby(['neuron','chain']).ngroups}  "
          f"neurons: {df['neuron'].nunique()}  z: {df['z'].min()}..{df['z'].max()} "
          f"(fit at {Z_FIT})")
    print(f"per-frame |node-centroid| offset (_sam px): "
          f"median {df['dist'].median():.1f}  mean {df['dist'].mean():.1f}  "
          f"p90 {df['dist'].quantile(.9):.1f}")
    cont = pd.to_numeric(df["contained"], errors="coerce")
    print(f"skeleton_contained: {np.nanmean(cont==1)*100:.1f}% True "
          f"(of {(~cont.isna()).sum()} non-NaN)")
    # sanity: tier-2 (crop-mapped) and _sam chains should AGREE if the crop_window
    # mapping is right; contained=False frames are the real per-chain drift.
    t2, sm = df[df["tier2"]], df[~df["tier2"]]
    print(f"  by frame type: tier2 median {t2['dist'].median():.1f} (n={len(t2)}) | "
          f"_sam median {sm['dist'].median():.1f} (n={len(sm)})")
    print(f"  contained=True median {df.loc[cont==1,'dist'].median():.1f} | "
          f"contained=False median {df.loc[cont==0,'dist'].median():.1f}")

    # bin by z-distance from the fit slice; per-bin COHERENT (cross-chain mean) offset
    df = df.assign(zbin=(df["zoff"] // 40 * 40).astype(int))
    print(f"\n  zoff-bin   n   |mean off|  mean|off|  coherence   contain%   "
          f"mean(offx,offy)")
    for zb, g in df.groupby("zbin"):
        mvec = np.array([g["offx"].mean(), g["offy"].mean()])
        mean_mag = float(np.hypot(g["offx"], g["offy"]).mean())
        coh = np.linalg.norm(mvec) / max(mean_mag, 1e-6)
        c = pd.to_numeric(g["contained"], errors="coerce")
        cpct = np.nanmean(c == 1) * 100
        print(f"  {zb:>6}  {len(g):>4}   {np.linalg.norm(mvec):>7.1f}   "
              f"{mean_mag:>7.1f}   {coh:>6.2f}    {cpct:>5.1f}    "
              f"({mvec[0]:+.1f},{mvec[1]:+.1f})")

    # headline correlations vs distance from the fit slice
    r_mag = np.corrcoef(df["zoff"], df["dist"])[0, 1]
    # coherent drift: regress the per-z MEAN offset magnitude on zoff
    perz = df.groupby("z").agg(mx=("offx", "mean"), my=("offy", "mean"),
                               n=("offx", "size"))
    perz = perz[perz["n"] >= 3]
    perz["cmag"] = np.hypot(perz["mx"], perz["my"])
    r_coh = (np.corrcoef(perz.index - Z_FIT, perz["cmag"])[0, 1]
             if len(perz) > 3 else float("nan"))
    print(f"\n  corr(|offset|, zoff)         = {r_mag:+.2f}   "
          "(per-frame; mixes registration + propagation)")
    print(f"  corr(coherent |meanoff|, zoff) = {r_coh:+.2f}   "
          f"(per-z>=3-chain mean; ISOLATES registration drift)   "
          f"[{len(perz)} z-slices]")
    # far end = the stringent test: max distance from the fit slice, where a bad
    # single-z affine drifts worst.
    fe = df[df["z"] >= df["z"].max() - 30]
    fe_coh = np.hypot(fe["offx"].mean(), fe["offy"].mean())
    print(f"  far end (z>={df['z'].max()-30}): coherent |mean off| = {fe_coh:.1f} px "
          f"(n={len(fe)}) — the worst-case slice for single-z drift")

    drift = (not np.isnan(r_coh)) and r_coh > 0.4 and fe_coh > 15
    print("\n  VERDICT: " + (
        "registration DRIFTS toward the far end — a single global affine is insufficient."
        if drift else
        "no coherent drift — the single global affine is z-STABLE across the imaged "
        "range. The few-px spread is node-vs-centroid scatter (propagation/thinness), "
        "not registration."))
    print("  read: POSITIVE coherent corr + growing per-bin |mean off| + large far-end "
          "coherent offset => drift.\n  ~0 coherent corr + small far-end coherent offset "
          "=> global affine is fine.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(config.OUTPUT_ROOT),
                    help="output root (per-chain qc.csv + state.json); default config.OUTPUT_ROOT")
    args = ap.parse_args()
    report(collect(args.root))
