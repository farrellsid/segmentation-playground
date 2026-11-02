"""Schematic teaching cartoon for the GT-free merge-metric (ADR 0015).

This is a conceptual explainer for a presentation slide, NOT a real EM image.
It draws three labelled cases (GOOD, MERGE, DROPOUT) of two adjacent neuron
cross-sections sharing a membrane, each seeded from its OWN CATMAID skeleton
node:

  - GOOD    : the mask covers its own node and contains no foreign node.
  - MERGE   : the mask bleeds across the membrane and swallows a neighbour's
              node (a foreign node inside = unambiguous bleed).
  - DROPOUT : the mask misses its own node (own node not covered = omission).

Run:  py -3 experiments/merge_metric_diagram.py
Out:  F:\ZhenLab\Data\repo_offload\presentation_figures\merge-metric-diagram\merge-metric-diagram.{png,svg}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

# Okabe-Ito colourblind-safe palette. Node roles are also distinguished by
# marker shape (circle vs triangle) so the figure survives grayscale printing.
OWN_GREEN = "#009E73"      # bluish-green: the mask's OWN skeleton node
FOREIGN_RED = "#D55E00"    # vermillion: ANOTHER neuron's node
MASK_BLUE = "#0072B2"      # blue: the segmentation mask (semi-transparent fill)
MEMBRANE = "#1A1A1A"       # near-black membranes
CELL_FACE = "#F2F2F2"      # pale cell interior
CELL_EDGE = "#4D4D4D"      # cell outline

MASK_ALPHA = 0.42
NODE_SIZE = 430
CELL_W, CELL_H = 2.55, 3.2  # cell footprint in data units


def rounded(ax, x, y, w, h, *, face, edge, lw, alpha=1.0, ls="-", zorder=1):
    """A rounded-rectangle blob standing in for a neuron cross-section."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.7",
        linewidth=lw, edgecolor=edge, facecolor=face,
        alpha=alpha, linestyle=ls, zorder=zorder,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    return patch


def draw_cells(ax, x0, y0):
    """Two adjacent neuron cross-sections sharing a dark membrane."""
    gap = 0.02
    lx = x0
    rx = x0 + CELL_W + gap
    rounded(ax, lx, y0, CELL_W, CELL_H, face=CELL_FACE, edge=CELL_EDGE, lw=2.5, zorder=1)
    rounded(ax, rx, y0, CELL_W, CELL_H, face=CELL_FACE, edge=CELL_EDGE, lw=2.5, zorder=1)
    # The shared membrane: a single thick dark line down the join.
    mid = (lx + CELL_W + rx) / 2.0
    ax.plot([mid, mid], [y0 + 0.18, y0 + CELL_H - 0.18],
            color=MEMBRANE, lw=5.0, solid_capstyle="round", zorder=2)
    return lx, rx, mid


def own_node(ax, x, y):
    ax.scatter([x], [y], s=NODE_SIZE, marker="o", facecolor=OWN_GREEN,
               edgecolor="white", linewidth=2.0, zorder=6)


def foreign_node(ax, x, y):
    ax.scatter([x], [y], s=NODE_SIZE + 120, marker="^", facecolor=FOREIGN_RED,
               edgecolor="white", linewidth=2.0, zorder=6)


def mask_blob(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=MASK_BLUE, edge=MASK_BLUE,
            lw=0, alpha=MASK_ALPHA, zorder=4)


def bleed_mask(ax, lx, y0, mid):
    """A mask that fills the left cell and spills across the membrane."""
    inset = 0.28
    x = lx + inset
    y = y0 + inset
    # extend well past the membrane into the right cell
    w = (mid - x) + CELL_W * 0.62
    h = CELL_H - 2 * inset
    verts = [
        (x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=MASK_BLUE,
                           edgecolor=MASK_BLUE, lw=0, alpha=MASK_ALPHA, zorder=4))


def panel(ax, kind):
    x0, y0 = 0.0, 0.0
    lx, rx, mid = draw_cells(ax, x0, y0)

    # node positions: own node centred in the left cell, foreign in the right.
    own_x = lx + CELL_W / 2.0
    own_y = y0 + CELL_H / 2.0
    for_x = rx + CELL_W / 2.0
    for_y = y0 + CELL_H / 2.0

    if kind == "good":
        title = "GOOD"
        sub = "own node inside,\nno foreign node"
        mask_blob(ax, lx + 0.28, y0 + 0.28, CELL_W - 0.56, CELL_H - 0.56)
        own_node(ax, own_x, own_y)
        foreign_node(ax, for_x, for_y)
        tcol = OWN_GREEN
    elif kind == "merge":
        title = "MERGE (bleed)"
        sub = "mask contains a\nforeign node"
        bleed_mask(ax, lx, y0, mid)
        own_node(ax, own_x, own_y)
        foreign_node(ax, for_x, for_y)
        tcol = FOREIGN_RED
    else:  # dropout
        title = "DROPOUT"
        sub = "own node\nnot covered"
        # small mask tucked in a corner, missing the central own node
        mask_blob(ax, lx + 0.30, y0 + 0.30, CELL_W * 0.44, CELL_H * 0.40)
        own_node(ax, own_x, own_y)
        foreign_node(ax, for_x, for_y)
        tcol = "#8A5A00"

    ax.set_title(title, fontsize=25, fontweight="bold", color=tcol, pad=14)
    ax.text((lx + rx + CELL_W) / 2.0, y0 - 0.62, sub, ha="center", va="top",
            fontsize=16.5, color="#222222", linespacing=1.25)

    ax.set_xlim(x0 - 0.35, rx + CELL_W + 0.35)
    ax.set_ylim(y0 - 1.75, y0 + CELL_H + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")


def build():
    fig = plt.figure(figsize=(15.5, 6.2))
    fig.suptitle("GT-free merge metric: grade each mask against its own skeleton node",
                 fontsize=23, fontweight="bold", y=0.985)

    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.84, bottom=0.20, wspace=0.10)
    for i, kind in enumerate(("good", "merge", "dropout")):
        panel(fig.add_subplot(gs[0, i]), kind)

    # Shared legend along the bottom.
    lax = fig.add_axes([0.0, 0.0, 1.0, 0.14])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    y = 0.55
    items = [
        ("o", OWN_GREEN, "the mask's OWN skeleton node"),
        ("^", FOREIGN_RED, "another neuron's node (foreign)"),
        ("s", MASK_BLUE, "segmentation mask"),
        ("line", MEMBRANE, "shared membrane"),
    ]
    x = 0.055
    for marker, color, label in items:
        if marker == "line":
            lax.plot([x - 0.018, x + 0.018], [y, y], color=color, lw=5.0,
                     solid_capstyle="round", transform=lax.transAxes, clip_on=False)
        else:
            alpha = MASK_ALPHA if marker == "s" else 1.0
            ec = MASK_BLUE if marker == "s" else "white"
            lax.scatter([x], [y], s=430, marker=marker, facecolor=color,
                        edgecolor=ec, linewidth=1.8, alpha=alpha,
                        transform=lax.transAxes, clip_on=False)
        lax.text(x + 0.028, y, label, va="center", ha="left", fontsize=15.5,
                 transform=lax.transAxes)
        x += 0.255

    return fig


def main():
    out_dir = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\merge-metric-diagram")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build()
    png = out_dir / "merge-metric-diagram.png"
    svg = out_dir / "merge-metric-diagram.svg"
    fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
