"""Real spill/overfill examples for the PI's documentation ask (roadmap item 16, section 5b):
not just the rate, real frames showing the mechanism, so a viewer can see the object clearly and
judge the scenario the spill happened in, with an eye toward identifying common causes.

Reads the 20-neuron membrane-scored subset from the 2026-08-09 overnight run
(`_merge_metric_membrane_subset.csv`). Renders four panels per example: raw EM, EM + mask + node
markers, the membrane/ridge map, all tightly windowed around the FIRST frame's own content, plus a
fourth panel showing a SECOND chosen frame from the same chain, independently windowed around ITS
own content (the two frames can be far apart in z, e.g. the anchor vs. a badly drifted frame, so
sharing one window would shrink both past the point of being readable).

Node markers: a green filled star is the neuron's OWN node when the mask actually contains it; a
hollow/outlined star in the same position is the own node when the mask does NOT contain it (the
mask missed its own target entirely, not just spilled into a neighbour, worth seeing even though
it will not get a "star" in the everyday sense of "found"). A red x is a foreign neighbour's node
the mask engulfed.

Two example kinds, found from real per-chain timelines (`eval.merge_metric`'s own
`anchor_catmaid_z` from each chain's state.json, cross-referenced against the already-scored
per-frame CSV), not hand-waved:

  bad_seed:  the chain's ANCHOR frame already contains a foreign node (own_contained may still be
             True; the seed itself was never clean). Panel 4 shows the anchor frame itself, so the
             comparison is "the origin" vs. "how far it grew", not two arbitrary adjacent frames.
  drift:     the anchor frame is genuinely clean (no foreign node, own node contained), and bleed
             appears only after some real distance of propagation. Panel 4 shows the LAST CLEAN
             frame immediately before the onset frame, so the comparison is the actual transition
             moment, not just "one more already-bad frame" (an earlier version of this script
             picked "the next chronological frame" for every example regardless of kind, which for
             an already-bad-at-the-seed chain just showed more of the same and was not useful).

    py -3 experiments/spill_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from eval.merge_metric import DEFAULT_RADIUS, load_node_table, nodes_by_z

SCALE = 8
TREE = Path(r"F:\ZhenLab\Data\output_masks\target_tier2_s1forced_neg_sam3_merged")
OUT_DIR = Path("docs/figures/presentation/spill-overfill-examples")
PAD = 30

# (neuron, chain_idx, z, z2, kind, label). z2 is the deliberately-chosen second frame (see the
# module docstring); it is NOT auto-derived as "the next frame" any more, since that gave a
# misleading pair for a chain that was already bad at its own seed.
EXAMPLES = [
    ("SAAVL", 17, 1568, 1460, "bad_seed",
     "z=1568 (9 foreign) vs its OWN ANCHOR z=1460 (1 foreign already at the seed): "
     "a small seed-level contamination that compounds far from the anchor"),
    ("AVJR", 3, 1607, 1610, "bad_seed",
     "z=1607 (8 foreign) vs its OWN ANCHOR z=1610 (6 foreign already at the seed): "
     "bad from the start, not a clean pipeline that drifted"),
    ("SABD", 25, 1330, 1331, "drift",
     "z=1330, the FIRST bled frame (1 foreign), vs z=1331, the LAST CLEAN frame: "
     "77 clean propagated frames from a clean anchor (z=1407) before this onset"),
    ("SIADR", 6, 1427, 1426, "drift",
     "z=1427, the FIRST bled frame (1 foreign), vs z=1426, the LAST CLEAN frame: "
     "23 clean propagated frames from a clean anchor (z=1404) before this onset"),
]


def own_and_foreign_nodes(mask, x0, y0, nodes, neuron, radius):
    """(own list of (xy, contained), foreign_xy list).

    A neuron can have more than one of its OWN nodes at a single z (real branch
    points, or virtual/interpolated nodes from separate chains sharing the same
    cell_name), same as eval.merge_metric.score_chain's own_contained check
    ("any of this neuron's nodes at this z"), not exactly one. Returning all of
    them, each with its own containment flag, keeps the figure honest instead of
    silently picking one (an earlier version of this script kept only the LAST
    same-named node found, which for a multi-branch neuron could show the wrong
    node's containment status)."""
    own = []
    foreign = []
    for x, y, cell, _nid in nodes:
        inside = pipeline._point_in_mask(mask, x - x0, y - y0, radius)
        if cell == neuron:
            own.append(((x, y), inside))
        elif inside:
            foreign.append((x, y))
    return own, foreign


def _draw_nodes(ax, own, foreign, wx0, wy0):
    for (ox, oy), contained in own:
        if contained:
            ax.scatter([ox - wx0], [oy - wy0], s=150, marker="*", c="#00e000",
                      edgecolors="black", linewidths=0.9, zorder=5)
        else:
            ax.scatter([ox - wx0], [oy - wy0], s=170, marker="*", facecolors="none",
                      edgecolors="#00e000", linewidths=2.2, zorder=5)
    for fx, fy in foreign:
        ax.scatter([fx - wx0], [fy - wy0], s=90, marker="x", c="red", linewidths=2.2, zorder=6)


def _frame_window(mask, x0, y0, foreign, frame_hw):
    """A tight window around ONE frame's own mask + its FOREIGN flagged nodes, padded.
    Own nodes deliberately do NOT expand this window: a neuron can have another,
    unrelated branch node far away at the same z (a real case, see the module
    docstring), and including it dragged the window out so far the actual mask
    became a speck. _draw_nodes still draws any own node that happens to land
    inside the resulting window; a far-away one is simply outside it, which is
    correct, it is not part of this spill's local picture.

    Independent per frame: two frames from the same chain can sit far apart in z (an
    anchor vs. a badly drifted frame), so a shared window would shrink both past the
    point of being readable."""
    ys, xs = np.where(mask)
    xs0, ys0 = [int(xs.min()) + x0], [int(ys.min()) + y0]
    xs1, ys1 = [int(xs.max()) + x0], [int(ys.max()) + y0]
    for nx, ny in foreign:
        xs0.append(nx); ys0.append(ny); xs1.append(nx); ys1.append(ny)
    H, W = frame_hw
    wx0, wy0 = max(0, int(min(xs0)) - PAD), max(0, int(min(ys0)) - PAD)
    wx1, wy1 = min(W, int(max(xs1)) + PAD), min(H, int(max(ys1)) + PAD)
    return wx0, wy0, wx1, wy1


def render_example(neuron, chain_idx, z, z2, kind, label, annotate_df, nbz, out_dir):
    from sam2_utils import membrane as membrane_mod

    chain_dir = TREE / neuron / f"chain_{chain_idx:02d}"
    masks = pipeline.chain_masks_in_sam(chain_dir)
    if z not in masks:
        print(f"[spill] skip {neuron} chain_{chain_idx:02d} z={z}: not in chain")
        return
    mask, x0, y0 = masks[z]

    em, full_hw = pipeline.load_frame_sam(z, scale=SCALE)
    em_gray = (em.mean(axis=2) if em.ndim == 3 else em).astype(np.float32)
    own, foreign = own_and_foreign_nodes(mask, x0, y0, nbz.get(z, []), neuron, DEFAULT_RADIUS)
    wx0, wy0, wx1, wy1 = _frame_window(mask, x0, y0, foreign, full_hw)
    em_win = em_gray[wy0:wy1, wx0:wx1]
    mem_win = membrane_mod.membrane_map(em_win)

    h, w = mask.shape
    mask_full = np.zeros(full_hw, bool); mask_full[y0:y0 + h, x0:x0 + w] = mask
    mask_win = mask_full[wy0:wy1, wx0:wx1]

    z2_ok = z2 in masks
    panel4_title = f"z={z2} not in this chain"
    if z2_ok:
        mask2, x02, y02 = masks[z2]
        own2, foreign2 = own_and_foreign_nodes(
            mask2, x02, y02, nbz.get(z2, []), neuron, DEFAULT_RADIUS)
        w2x0, w2y0, w2x1, w2y1 = _frame_window(mask2, x02, y02, foreign2, full_hw)
        em2, _ = pipeline.load_frame_sam(z2, scale=SCALE)
        em2_gray = (em2.mean(axis=2) if em2.ndim == 3 else em2).astype(np.float32)
        em2_win = em2_gray[w2y0:w2y1, w2x0:w2x1]
        h2, w2wd = mask2.shape
        mask2_full = np.zeros(full_hw, bool); mask2_full[y02:y02 + h2, x02:x02 + w2wd] = mask2
        mask2_win = mask2_full[w2y0:w2y1, w2x0:w2x1]
        panel4_title = (f"anchor z={z2} ({len(foreign2)} foreign)" if kind == "bad_seed"
                        else f"last clean frame z={z2}")

    fig, ax = plt.subplots(1, 4, figsize=(21, 6))
    ax[0].imshow(em_win, cmap="gray")
    ax[0].set_title(f"raw EM (z={z})", fontsize=10)

    ax[1].imshow(em_win, cmap="gray")
    if mask_win.any():
        ov = np.zeros((*mask_win.shape, 4)); ov[mask_win] = (0.0, 0.45, 0.90, 0.45)
        ax[1].imshow(ov)
    win_h, win_w = em_win.shape[:2]
    ax[1].set_xlim(0, win_w); ax[1].set_ylim(win_h, 0)   # lock BEFORE scatter: a node
    # marker outside the window (own nodes are drawn even when outside, see
    # _draw_nodes) would otherwise auto-expand the whole axes and shrink the image
    # to a speck, exactly the bug an earlier version of this script had.
    _draw_nodes(ax[1], own, foreign, wx0, wy0)
    own_status = "contained" if any(c for _xy, c in own) else ("MISSED" if own else "n/a")
    ax[1].set_title(f"mask + nodes ({len(foreign)} foreign, own {own_status})", fontsize=10)

    ax[2].imshow(mem_win, cmap="magma")
    ax[2].set_title("membrane / ridge map", fontsize=10)

    if z2_ok:
        ax[3].imshow(em2_win, cmap="gray")
        if mask2_win.any():
            ov2 = np.zeros((*mask2_win.shape, 4)); ov2[mask2_win] = (0.90, 0.45, 0.0, 0.45)
            ax[3].imshow(ov2)
        win2_h, win2_w = em2_win.shape[:2]
        ax[3].set_xlim(0, win2_w); ax[3].set_ylim(win2_h, 0)
        _draw_nodes(ax[3], own2, foreign2, w2x0, w2y0)
    else:
        ax[3].imshow(np.zeros_like(em_win), cmap="gray", vmin=0, vmax=1)
    ax[3].set_title(panel4_title, fontsize=10)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{neuron} chain_{chain_idx:02d}\n{label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"spill_{neuron}_chain{chain_idx:02d}_z{z}.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[spill] wrote {f}")


def main():
    annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, SCALE)
    for neuron, chain_idx, z, z2, kind, label in EXAMPLES:
        render_example(neuron, chain_idx, z, z2, kind, label, annotate_df, nbz, OUT_DIR)


if __name__ == "__main__":
    main()
