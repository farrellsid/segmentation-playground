"""Chart the real z-window temporal-projection sweep for the presentation's "already tried"
slide: does averaging membrane crops across adjacent z-slices (to average transient organelles
out and leave stable membranes) reduce the merge metric's foreign-node bleed?

Answer, from the real sweep: no, it's a monotonic regression. Reruns
experiments/dense_membrane_fill.py's --sweep-temporal path and plots the printed numbers instead
of transcribing them, so the chart always matches what the sweep actually reports. CPU only, no
model, real masks read from disk.

Run: py -3 experiments/temporal_sweep_figure.py
Out: docs/figures/presentation/temporal-sweep/temporal_sweep.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import dense_overlay as do
from experiments.dense_membrane_fill import INDEX_CACHE, SCALE, apply_cap, grow_all
from eval.merge_metric import DEFAULT_RADIUS, load_node_table, nodes_by_z
import pipeline

BLUE, ORANGE, GREEN, MUTED = "#0072B2", "#E69F00", "#009E73", "#6b7280"
OUT_DIR = Path("docs/figures/presentation/temporal-sweep")
Z = 1456
PAD = 18
CAP = 5.0
UF_MIN = 0.6


def sweep(z: int = Z):
    """Reruns dense_membrane_fill's own sweep-temporal logic directly (not a subprocess), so
    this stays a plain function call other scripts can reuse. Returns
    {(window, combine): {"foreign": int, "new_bleed": int}}, window=0 is the existing
    single-slice baseline (combine is irrelevant there, always "median")."""
    idx = json.loads(INDEX_CACHE.read_text())
    em, _ = pipeline.load_frame_sam(z, scale=SCALE)
    h8, w8 = em.shape[:2]
    em_gray = em.mean(axis=2) if em.ndim == 3 else em

    max_w = 2
    frames = {z: em_gray}
    for dz in range(1, max_w + 1):
        for zz in (z - dz, z + dz):
            fz, _ = pipeline.load_frame_sam(zz, scale=SCALE)
            frames[zz] = fz.mean(axis=2) if fz.ndim == 3 else fz

    nmasks = do.neuron_masks_at_z(z, idx, SCALE, h8, w8)
    nbz = nodes_by_z(load_node_table(), SCALE)
    nodes = nbz.get(z, [])

    out = {}
    for window in (0, 1, 2):
        combos = ("median",) if window == 0 else ("median", "mean", "max")
        for combine in combos:
            recs = grow_all(nmasks, frames, z, window, combine, PAD, nodes, DEFAULT_RADIUS)
            _chosen, m = apply_cap(recs, CAP, UF_MIN)
            out[(window, combine)] = {"foreign": m["foreign"], "new_bleed": m["new_bleed"]}
            print(f"[temporal-sweep] window={window} combine={combine}: "
                  f"foreign={m['foreign']} new_bleed={m['new_bleed']}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = sweep()

    baseline = results[(0, "median")]["foreign"]
    windows = [1, 2]
    combines = ["median", "mean", "max"]
    colors = {"median": BLUE, "mean": ORANGE, "max": GREEN}

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    ax.axhline(baseline, color=MUTED, linestyle="--", linewidth=1.5,
               label=f"window=0 baseline ({baseline} foreign nodes)")
    for combine in combines:
        ys = [results[(w, combine)]["foreign"] for w in windows]
        ax.plot([0] + windows, [baseline] + ys, marker="o", color=colors[combine],
                label=f"combine={combine}")

    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("temporal window (z-slices averaged each side)")
    ax.set_ylabel("foreign skeleton nodes engulfed (lower is better)")
    ax.set_title(f"Averaging organelles out across z makes bleed WORSE, not better  (z={Z}, "
                 f"{len(results)} configs, uf_min={UF_MIN})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = OUT_DIR / "temporal_sweep.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[temporal-sweep] wrote {out}")


if __name__ == "__main__":
    main()
