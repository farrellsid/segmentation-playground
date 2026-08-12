"""Compose the slide-ready figures for the membrane / dense / overlap-resolution slides.

`dense_overlaps_figure.py` already writes the seven bare panel PNGs (one frame shown
several ways). That script's combined figure is a near-square 7-panel grid with an
orphan panel, which does not sit well on a 16:9 slide. This script reads those panels
back off disk (no model, no GPU, no mask reads) and lays them out twice, at aspect
ratios a slide can actually use:

    figure_ways_of_seeing.png   2x2   raw EM | Sato ridge | dense segmentation | automask
    figure_overlap_resolve.png  1x3   contested overlap | argmax | watershed on the ridge

The split follows the two things the supervisor asked to see: the ridge filter next to a
dense segmentation as a direct comparison (plus the automask output), and then the ridge
map used to illustrate argmax versus watershed.

Run: py -3 experiments/slide_overlap_panels.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SRC = Path("docs/figures/presentation/dense-overlaps")
INK = "#222222"

WAYS = [
    ("panel1_em.png", "1.  Raw EM"),
    ("panel2_ridge.png", "2.  Sato ridge filter: membranes as walls"),
    ("panel3_dense.png", "3.  Dense segmentation, 25 prompted neurons"),
    ("panel5_amg.png", "4.  Automask (SAM2 AMG), 28 unlabelled masks"),
]

RESOLVE = [
    ("panel4a_overlap_raw.png", "Contested pixels (white)\nwhere 5 masks overlap"),
    ("panel4b_argmax.png", "argmax: highest score\nwins the pixel"),
    ("panel4c_watershed.png", "watershed: grow to the\nridge walls"),
]


def load(name):
    p = SRC / name
    if not p.exists():
        raise SystemExit(f"missing panel {p}; run experiments/dense_overlaps_figure.py first")
    return mpimg.imread(p)


def compose(spec, nrows, ncols, out_name, suptitle, panel_w=4.6, title_size=12):
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_w * nrows + 0.55),
                             dpi=170)
    fig.patch.set_facecolor("white")
    for ax, (name, label) in zip(axes.ravel(), spec):
        ax.imshow(load(name))
        ax.set_title(label, fontsize=title_size, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    for ax in axes.ravel()[len(spec):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=title_size + 2, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = SRC / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


compose(WAYS, 2, 2, "figure_ways_of_seeing.png",
        "One frame, four ways of seeing it  (z=1456, 180x180 px crop at scale 8)")
compose(RESOLVE, 1, 3, "figure_overlap_resolve.png",
        "Where masks overlap, two ways to decide the pixel  (magenta = the two disagree)")
