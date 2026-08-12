"""Three real failure modes for the presentation's "where it still breaks" slide.

All three panels are read FROM DISK (no model, no GPU) and all three are cases the
merge metric scores as CLEAN: the mask contains its own skeleton node and no foreign
node. That is the point of the slide. The metric is a bleed-and-dropout floor, so a
mask can pass it while still being the wrong object.

    a) nucleus capture: the mask fills the nucleus and stops at the nuclear envelope,
       leaving a rim of cytoplasm and the real cell membrane outside it. Read from the
       local dense labelmaps (docs/figures/sam3-bakeoff/dense-overlay/*.npz).
    b) fragment: the mask is a speck at the skeleton node inside a much larger cell.
       Selected as a very high underfill_fraction frame in the per-slice tree.
    c) wrong object: the mask has locked onto a round organelle next to the node
       rather than the cell. Selected from the tree's rare dropout frames.

Panels b and c read per-chain masks from the scored tree on F:, the same grid
eval.merge_metric scores, so the picture and the CSV agree by construction.

Run: py -3 experiments/failure_modes_figure.py
Out: docs/figures/presentation/failure-modes/failure_modes.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline
from eval.merge_metric import DEFAULT_RADIUS, load_node_table, nodes_by_z
from pipeline.predict import _point_in_mask

RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
TREE = "original_perslice_only_guard_merged"
DENSE = Path("docs/figures/sam3-bakeoff/dense-overlay")
SCALE = 8

MASK_RGBA = (0.0, 0.45, 0.70, 0.42)      # Okabe-Ito blue, matching the other figures
OWN_C = "#00C000"
INK = "#222222"

# panel a: a captured nucleus in the dense labelmap (large, round, well filled)
NUCLEUS = dict(z=1468, lid=44, pad=34)
# panel b: mask reduced to a speck inside a much larger cell
FRAGMENT = dict(neuron="AIYL", chain=18, z=1457, pad=46)
# panel c: mask locked onto a round organelle beside the node
WRONG = dict(neuron="RIH", chain=60, z=1534, pad=52)


def em_frame(z):
    em, _ = pipeline.load_frame_sam(z, scale=SCALE)
    return em


def window(cx, cy, half, shape):
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(shape[1], int(cx + half)), min(shape[0], int(cy + half))
    return x0, y0, x1, y1


def metric_status(mask, mx0, my0, z, neuron):
    """What eval.merge_metric would say about this frame, computed here so the caption
    cannot drift from the truth: own-node containment and foreign-node count."""
    own_in, foreign_in = False, 0
    for x, y, cell, _nid in nodes_by_z(load_node_table(), SCALE).get(z, []):
        inside = _point_in_mask(mask, x - mx0, y - my0, DEFAULT_RADIUS)
        if cell == neuron:
            own_in = own_in or inside
        elif inside:
            foreign_in += 1
    if not own_in:
        return "metric: DROPOUT (caught)"
    if foreign_in:
        return f"metric: MERGE, {foreign_in} foreign node(s) (caught)"
    return "metric: clean (invisible to it)"


def show(ax, crop, mask_in_crop, title, sub, own_xy=None, status=None):
    hw = crop.shape[:2]
    ax.imshow(crop, cmap="gray")
    rgba = np.zeros((*hw, 4))
    rgba[mask_in_crop] = MASK_RGBA
    ax.imshow(rgba)
    # mask outline, so the boundary the model chose is unmistakable
    ax.contour(mask_in_crop.astype(float), levels=[0.5], colors=["#0072B2"], linewidths=1.6)
    if own_xy is not None:
        ax.plot(*own_xy, "*", color=OWN_C, ms=15, mec="black", mew=0.7)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", pad=6)
    label = sub if status is None else f"{sub}\n{status}"
    ax.set_xlabel(label, fontsize=10.5, color=INK, labelpad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def panel_nucleus(ax):
    z, lid, pad = NUCLEUS["z"], NUCLEUS["lid"], NUCLEUS["pad"]
    d = np.load(DENSE / f"labelmap_z{z}.npz")
    lab = d[list(d.keys())[0]]
    m = lab == lid
    if not m.any():
        raise SystemExit(f"label {lid} absent at z{z}")
    ys, xs = np.where(m)
    half = max(np.ptp(xs), np.ptp(ys)) / 2 + pad
    em = em_frame(z)
    x0, y0, x1, y1 = window(xs.mean(), ys.mean(), half, em.shape)

    # the dense legend is name -> label id, so invert it to name this mask's neuron
    legend = json.loads((DENSE / "_legend.json").read_text())
    name = next((n for n, i in legend.items() if int(i) == lid), None)
    own, status = None, None
    if name:
        for x, y, cell, _nid in nodes_by_z(load_node_table(), SCALE).get(z, []):
            if cell == name and x0 <= x < x1 and y0 <= y < y1:
                own = (x - x0, y - y0)
                break
        # the dense labelmap is a full-frame array, so its origin is (0, 0)
        status = metric_status(m, 0, 0, z, name)

    show(ax, em[y0:y1, x0:x1], m[y0:y1, x0:x1],
         "Nucleus capture",
         "the mask fills the nucleus and stops at the\nnuclear envelope, not the cell membrane",
         own_xy=own, status=status)


def _chain_panel(ax, spec, title, sub):
    neuron, ci, z, pad = spec["neuron"], spec["chain"], spec["z"], spec["pad"]
    cdir = RE / TREE / neuron / f"chain_{ci:02d}"
    if not cdir.exists():
        raise SystemExit(f"missing {cdir}")
    masks = pipeline.chain_masks_in_sam(cdir)
    if z not in masks:
        raise SystemExit(f"no mask for {neuron}/chain_{ci:02d} at z{z}")
    m, mx0, my0 = masks[z]
    ys, xs = np.where(m)
    cx, cy = mx0 + xs.mean(), my0 + ys.mean()
    em = em_frame(z)
    x0, y0, x1, y1 = window(cx, cy, pad + max(np.ptp(xs), np.ptp(ys)) / 2, em.shape)
    crop = em[y0:y1, x0:x1]

    placed = np.zeros(crop.shape[:2], bool)
    h, w = m.shape
    ax0, ay0 = max(mx0, x0), max(my0, y0)
    ax1, ay1 = min(mx0 + w, x1), min(my0 + h, y1)
    if ax1 > ax0 and ay1 > ay0:
        placed[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = \
            m[ay0 - my0:ay1 - my0, ax0 - mx0:ax1 - mx0]

    own = None
    for x, y, cell, _nid in nodes_by_z(load_node_table(), SCALE).get(z, []):
        if cell == neuron and x0 <= x < x1 and y0 <= y < y1:
            own = (x - x0, y - y0)
            break
    show(ax, crop, placed, title, sub, own_xy=own,
         status=metric_status(m, mx0, my0, z, neuron))


fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.6), dpi=170)
fig.patch.set_facecolor("white")

panel_nucleus(axes[0])
_chain_panel(axes[1], FRAGMENT, "Fragment",
             "the mask is a speck at the skeleton node\ninside a much larger cell")
_chain_panel(axes[2], WRONG, "Wrong object",
             "the mask locked onto a round organelle\nbeside the node")

fig.suptitle("Three ways a mask is wrong, and what the metric sees",
             fontsize=14.5, color=INK, y=0.985)
fig.tight_layout(rect=(0, 0.13, 1, 0.945))
fig.text(0.5, 0.028,
         "green star = the chain's own skeleton node      blue outline = the predicted mask",
         ha="center", fontsize=10.5, color="#666666")

out = Path("docs/figures/presentation/failure-modes/failure_modes.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
