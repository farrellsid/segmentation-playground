"""Per-slice jitter strip: one neuron's per-slice masks on a few consecutive
slices, side by side, to show the janky boundary that per-slice can produce even
when each frame is metrically fine. CPU only, disk masks.

Run: py -3 experiments/perslice_jitter_strip.py [--neuron URAVL --chain 5]
Out: F:\ZhenLab\Data\repo_offload\presentation_figures\autofill-demo\perslice-jitter.png  (public/images too)
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

import pipeline

RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
TREE = RE / "target_perslice_only_guard_sam3_merged"
SCALE = 8
N = 5


def pick_chain(neuron=None, chain=None):
    if neuron is not None and chain is not None:
        return neuron, chain, pipeline.chain_masks_in_sam(TREE / neuron / f"chain_{chain:02d}")
    for nd in sorted(d for d in TREE.iterdir() if d.is_dir()):
        for cd in sorted(nd.glob("chain_*")):
            m = pipeline.chain_masks_in_sam(cd)
            if len(m) >= N + 2 and 300 <= int(next(iter(m.values()))[0].sum()) <= 4000:
                return nd.name, int(cd.name.split("_")[1]), m
    raise SystemExit("no suitable chain found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuron", default=None)
    ap.add_argument("--chain", type=int, default=None)
    ap.add_argument("--pad", type=int, default=24)
    args = ap.parse_args()

    neuron, chain, masks = pick_chain(args.neuron, args.chain)
    zs = sorted(masks)
    mid = len(zs) // 2
    zsel = zs[max(0, mid - N // 2): max(0, mid - N // 2) + N]

    # common window across the selected slices
    xs0 = ys0 = 10 ** 9; xs1 = ys1 = 0
    for z in zsel:
        m, x0, y0 = masks[z]
        yy, xx = np.where(m)
        xs0 = min(xs0, xx.min() + x0); ys0 = min(ys0, yy.min() + y0)
        xs1 = max(xs1, xx.max() + x0); ys1 = max(ys1, yy.max() + y0)
    p = args.pad
    x0w, y0w, x1w, y1w = xs0 - p, ys0 - p, xs1 + p, ys1 + p

    fig, ax = plt.subplots(1, len(zsel), figsize=(2.3 * len(zsel), 2.6))
    for a, z in zip(ax, zsel):
        em, _ = pipeline.load_frame_sam(int(z), scale=SCALE)
        em = em.mean(axis=2) if em.ndim == 3 else em
        H, W = em.shape[:2]
        xa, ya, xb, yb = max(0, x0w), max(0, y0w), min(W, x1w), min(H, y1w)
        m, mx, my = masks[z]
        full = np.zeros((H, W), bool); full[my:my + m.shape[0], mx:mx + m.shape[1]] = m
        a.imshow(em[ya:yb, xa:xb], cmap="gray")
        ov = np.zeros((yb - ya, xb - xa, 4))
        ov[full[ya:yb, xa:xb]] = (0.90, 0.62, 0.0, 0.5)
        a.imshow(ov)
        a.set_title(f"z={z}", fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Per-slice masks on consecutive slices: {neuron} chain_{chain:02d} "
                 f"(the boundary jumps)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for out in (Path("public/images/perslice-jitter.png"),
                Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\autofill-demo\perslice-jitter.png")):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"{neuron}/chain_{chain:02d} z={zsel}  wrote perslice-jitter.png")


if __name__ == "__main__":
    main()
