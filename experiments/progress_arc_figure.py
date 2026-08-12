"""Progress-arc figure for the presentation: bleed rate down, coverage up across the
pipeline changes. Static, no model, no cluster. Numbers are the measured merge-metric
values on the 16-neuron / 629-chain set (the SAM3 per-slice point is its whole-set
foreign-frame-rate, essentially equal to its 15-neuron matched value of 0.087).

Two measures, both fractions on [0, 1], so they share ONE y-axis (no dual axis).
Okabe-Ito colours (colourblind-safe), thin marks, direct end-labels, recessive grid.

Run: py -3 experiments/progress_arc_figure.py
Out: F:\ZhenLab\Data\repo_offload\presentation_figures\progress-arc\progress_arc.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- data (measured; see eval/merge_metric.py output aggregated across configs) -------
stages = [
    "Whole-image\n(early)",
    "+ crop\n+ negatives",
    "Per-slice",
    "+ blow-up\nguard",
    "SAM3\nper-slice",
]
bleed = [0.453, 0.321, 0.136, 0.109, 0.087]      # foreign-frame-rate, lower is better
# (SAM3 point is its matched value on the shared neuron set; whole-worm is 0.088, same to 2 dp)
coverage = [0.681, 0.870, 0.998, 0.999, 0.999]   # own-node containment, higher is better

# --- colours (Okabe-Ito, CVD-safe) ---------------------------------------------------
BLUE = "#0072B2"     # bleed
ORANGE = "#E69F00"   # coverage
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

x = list(range(len(stages)))
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# recessive horizontal grid behind the data
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=1.0)
ax.xaxis.grid(False)

ax.plot(x, coverage, "-o", color=ORANGE, linewidth=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
ax.plot(x, bleed, "-o", color=BLUE, linewidth=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)

# direct series labels at the right end (no legend box)
ax.text(x[-1] + 0.10, coverage[-1], "Coverage\n(own node)", color=ORANGE,
        va="center", ha="left", fontsize=11, fontweight="bold")
ax.text(x[-1] + 0.10, bleed[-1], "Bleed rate\n(foreign frame)", color=BLUE,
        va="center", ha="left", fontsize=11, fontweight="bold")

# endpoint value labels only (selective, not every point); both placed above their point
for xi, yi in [(x[0], bleed[0]), (x[-1], bleed[-1])]:
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 12),
                ha="center", color=BLUE, fontsize=10, fontweight="bold")
for xi, yi, off in [(x[0], coverage[0], -16), (x[-1], coverage[-1], 10)]:
    ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, off),
                ha="center", color=ORANGE, fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10, color=INK)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], color=MUTED, fontsize=10)
ax.set_xlim(-0.35, len(stages) - 1 + 1.15)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#cccccc")

# No chart title: the slide title carries the headline. Keep only a small metric note.
ax.text(0, 1.03, "Merge metric on the 16-neuron set; lower bleed and higher coverage are better",
        transform=ax.transAxes, fontsize=11, color=MUTED)

fig.tight_layout()
out = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\progress-arc\progress_arc.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
