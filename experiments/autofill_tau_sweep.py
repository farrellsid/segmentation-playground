"""Tau sweep for membrane autofill (grow-to-membrane).

For a sample of real per-slice masks, grow each out to the ridge walls at a range
of thresholds tau (walls = membrane_map > tau), and measure the tradeoff:

  underfill AFTER grow  (want LOW  -> falls as tau rises, more of the cell fills)
  bleed rate            (want LOW  -> rises as tau rises, growth leaks across weak
                         membranes and swallows a neighbour's skeleton node)

The useful tau minimises underfill while keeping bleed near zero. CPU only.

Run: py -3 experiments/autofill_tau_sweep.py [--n 40]
Out: F:\ZhenLab\Data\repo_offload\presentation_figures\autofill-demo\tau_sweep.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label

import pipeline
from sam2_utils import membrane
from pipeline.predict import _point_in_mask
from eval.merge_metric import load_node_table, nodes_by_z, DEFAULT_RADIUS

RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
TREE = RE / "target_perslice_only_guard_sam3_merged"
SCALE = 8
BLUE, ORANGE, GREEN, MUTED = "#0072B2", "#E69F00", "#009E73", "#6b7280"


def grow(mask_win, mem, tau):
    """Grow the mask into the free-space (ridge <= tau) component(s) it sits in."""
    free = mem <= tau
    lab = label(free | mask_win, connectivity=1)
    ids = [i for i in np.unique(lab[mask_win]) if i != 0]
    return np.isin(lab, ids)


def foreign_hit(mask_win, x1, y1, nodes, neuron, radius):
    """True if any OTHER neuron's node falls inside the (window) mask."""
    for x, y, cell, _ in nodes:
        if cell != neuron and _point_in_mask(mask_win, x - x1, y - y1, radius):
            return True
    return False


def sample_masks(n, min_area=200, max_area=6000):
    """One mid-z mask per chain, spread across neurons, within an area band."""
    out = []
    neuron_dirs = sorted(d for d in TREE.iterdir() if d.is_dir())
    for nd in neuron_dirs:
        for cd in sorted(nd.glob("chain_*")):
            masks = pipeline.chain_masks_in_sam(cd)
            if not masks:
                continue
            zs = sorted(masks)
            z = zs[len(zs) // 2]                       # mid slice of the chain
            m, x0, y0 = masks[z]
            a = int(m.sum())
            if min_area <= a <= max_area:
                out.append((nd.name, z, m, x0, y0))
                break                                  # one mask per neuron, spread wide
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    args = ap.parse_args()

    taus = np.round(np.linspace(0.15, 0.85, 15), 3)
    nbz = nodes_by_z(load_node_table(), SCALE)
    samples = sample_masks(args.n)
    print(f"sampled {len(samples)} masks; sweeping {len(taus)} taus")

    uf_before = []
    uf_after = np.zeros((len(samples), len(taus)))
    bled = np.zeros((len(samples), len(taus)), bool)
    arearatio = np.zeros((len(samples), len(taus)))

    for i, (neuron, z, m, x0, y0) in enumerate(samples):
        em, _ = pipeline.load_frame_sam(int(z), scale=SCALE)
        em = em.mean(axis=2) if em.ndim == 3 else em
        H, W = em.shape[:2]
        full = np.zeros((H, W), bool); full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        ys, xs = np.where(full); p = args.pad
        y1, y2 = max(0, ys.min() - p), min(H, ys.max() + p)
        x1, x2 = max(0, xs.min() - p), min(W, xs.max() + p)
        win = full[y1:y2, x1:x2]
        mem = membrane.membrane_map(em[y1:y2, x1:x2].astype(np.float32))
        nodes = nbz.get(int(z), [])
        a0 = max(int(win.sum()), 1)
        uf_before.append(membrane.underfill_fraction(win, mem))
        for j, t in enumerate(taus):
            g = grow(win, mem, t)
            uf_after[i, j] = membrane.underfill_fraction(g, mem)
            arearatio[i, j] = g.sum() / a0
            bled[i, j] = foreign_hit(g, x1, y1, nodes, neuron, args.radius)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)}")

    mean_uf = uf_after.mean(axis=0)
    bleed_rate = bled.mean(axis=0)
    base_uf = float(np.mean(uf_before))
    # best tradeoff: minimise underfill + bleed together (both want to be low)
    star = int(np.argmin(mean_uf + bleed_rate))

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e6e6", lw=1.0); ax.xaxis.grid(False)
    ax.axhline(base_uf, color=MUTED, ls="--", lw=1.4)
    ax.text(taus[0], base_uf + 0.015, f"original underfill  {base_uf:.2f}", color=MUTED, fontsize=10)
    ax.plot(taus, mean_uf, "-o", color=BLUE, lw=2.5, ms=7, mec="white", mew=1.1, zorder=3)
    ax.plot(taus, bleed_rate, "-o", color=ORANGE, lw=2.5, ms=7, mec="white", mew=1.1, zorder=3)
    ax.text(taus[-1] + 0.01, mean_uf[-1], "underfill\nafter grow", color=BLUE, va="center",
            ha="left", fontsize=11, fontweight="bold")
    ax.text(taus[-1] + 0.01, bleed_rate[-1], "bleed rate\n(foreign node)", color=ORANGE,
            va="center", ha="left", fontsize=11, fontweight="bold")
    ax.axvline(taus[star], color=GREEN, lw=1.6, ls=":")
    ax.plot([taus[star]], [mean_uf[star]], "o", color=GREEN, ms=11, zorder=5)
    ax.annotate(f"best tradeoff  tau {taus[star]}\nunderfill {mean_uf[star]:.2f}, "
                f"bleed {bleed_rate[star]:.2f}",
                (taus[star], mean_uf[star]), textcoords="offset points", xytext=(12, -48),
                color=GREEN, fontsize=10, fontweight="bold")
    ax.set_xlim(taus[0] - 0.02, taus[-1] + 0.16)
    ax.set_ylim(0, max(1.0, base_uf + 0.05))
    ax.set_xlabel("tau  (ridge threshold for a membrane wall)", fontsize=11, color="#222")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0, 1.06, f"Autofill tau sweep: underfill falls, bleed rises  (n={len(samples)} masks)",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color="#222")
    fig.tight_layout()
    out = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\autofill-demo\tau_sweep.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"base underfill {base_uf:.3f}; best tau {taus[star]} "
          f"(underfill {mean_uf[star]:.3f}, bleed {bleed_rate[star]:.3f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
