"""Quick, laptop-only prototype: grow an underfilled per-slice mask out to the
membrane using the Sato ridge map (the "autofill" idea). CPU only, one frame.

For one chain/frame: load the per-slice mask (canonical _sam grid), crop the EM
around it, compute the membrane ridge map, then watershed-grow the mask from its
interior out to the ridge walls. Renders EM + original mask, the ridge map, and
EM + grown mask, and prints underfill before/after.

Run:
    py -3 experiments/membrane_autofill_demo.py                       # URAVL/5/1417
    py -3 experiments/membrane_autofill_demo.py --neuron AIZL --chain 54 --z 1408
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
from scipy import ndimage
from skimage.segmentation import watershed

import pipeline
from sam2_utils import membrane

RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
TREE = "target_perslice_only_guard_sam3_merged"
SCALE = 8


def grow_to_membrane(mask_win, mem, cap=5.0):
    """Watershed the mask interior out to the ridge walls. Returns (grown, capped)."""
    interior = ndimage.binary_erosion(mask_win, iterations=1)
    if interior.sum() == 0:
        interior = mask_win
    markers = np.zeros(mask_win.shape, np.int32)
    markers[interior] = 1
    border = np.zeros(mask_win.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    markers[border & ~mask_win] = 2
    grown = watershed(mem, markers) == 1
    if grown.sum() > cap * max(int(mask_win.sum()), 1):   # runaway: walls leaked
        return mask_win, True
    return grown, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuron", default="URAVL")
    ap.add_argument("--chain", type=int, default=5)
    ap.add_argument("--z", type=int, default=1417)
    ap.add_argument("--pad", type=int, default=28)
    args = ap.parse_args()

    cdir = RE / TREE / args.neuron / f"chain_{args.chain:02d}"
    masks = pipeline.chain_masks_in_sam(cdir)
    if args.z not in masks:
        print(f"z={args.z} not in chain; available z: {sorted(masks)[:12]} ...")
        return
    mask, x0, y0 = masks[args.z]

    em, _ = pipeline.load_frame_sam(int(args.z), scale=SCALE)
    em = em.mean(axis=2) if em.ndim == 3 else em
    H, W = em.shape[:2]

    full = np.zeros((H, W), bool)
    mh, mw = mask.shape
    full[y0:y0 + mh, x0:x0 + mw] = mask
    ys, xs = np.where(full)
    p = args.pad
    y1, y2 = max(0, ys.min() - p), min(H, ys.max() + p)
    x1, x2 = max(0, xs.min() - p), min(W, xs.max() + p)

    em_win = em[y1:y2, x1:x2].astype(np.float32)
    mask_win = full[y1:y2, x1:x2]
    mem = membrane.membrane_map(em_win)

    grown, capped = grow_to_membrane(mask_win, mem)

    uf_before = membrane.underfill_fraction(mask_win, mem)
    uf_after = membrane.underfill_fraction(grown, mem)
    print(f"{args.neuron}/chain_{args.chain:02d} z={args.z}")
    print(f"  area  {int(mask_win.sum())} -> {int(grown.sum())} px  (capped={capped})")
    print(f"  underfill  {uf_before:.3f} -> {uf_after:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    ax[0].imshow(em_win, cmap="gray")
    ov = np.zeros((*mask_win.shape, 4)); ov[mask_win] = (0.90, 0.62, 0.0, 0.45)
    ax[0].imshow(ov); ax[0].set_title(f"original per-slice mask\nunderfill {uf_before:.2f}")
    ax[1].imshow(mem, cmap="magma"); ax[1].set_title("membrane ridge map")
    ax[2].imshow(em_win, cmap="gray")
    ov2 = np.zeros((*grown.shape, 4)); ov2[grown] = (0.0, 0.62, 0.0, 0.45)
    ax[2].imshow(ov2); ax[2].set_title(f"grown to membrane\nunderfill {uf_after:.2f}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Membrane autofill: {args.neuron} chain_{args.chain:02d} z={args.z}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("docs/figures/presentation/autofill-demo")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.neuron}_chain{args.chain:02d}_z{args.z}.png"
    fig.savefig(f, dpi=140); plt.close(fig)
    print(f"  wrote {f}")


if __name__ == "__main__":
    main()
