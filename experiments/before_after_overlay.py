"""Before/after mask-overlay figures that make a measured merge fix VISUALLY obvious.

Reads per-chain masks FROM DISK (no model runs, no GPU) from an OLDER, bleedier
merged tree and the NEWER SAM3 per-slice tree, finds a frame where the OLD mask
merged into a neighbour (the metric counts a foreign skeleton node inside the mask)
and the NEW mask stays clean, and renders them side by side on the SAME EM crop:

    left  = OLD config mask (blue)   title carries its n_foreign for that frame
    right = NEW config mask (orange) title carries n_foreign == 0

    green *  = the chain's own centreline node (should be INSIDE both masks)
    red   x  = the foreign neighbour node(s) the metric flagged (INSIDE old = bleed)

Everything is on the canonical _sam grid (scale 8): pipeline.chain_masks_in_sam
rebuilds each chain's crop_window into full-_sam-frame placement (mask, x0, y0),
exactly what eval.merge_metric scores, and eval.merge_metric.nodes_by_z /
foreign_hits define which node is foreign, so the picture and the CSV number line
up by construction. The EM backdrop is pipeline.load_frame_sam(z, scale=8).

Candidate search reuses each tree's _merge_metric.csv: join OLD and NEW on
(neuron, chain_idx, z), keep frames where OLD n_foreign >= 1 and NEW n_foreign == 0,
restricted to the neurons present in BOTH trees. A shortlist is then scored on disk
for visual clarity (single clearly-contained foreign node, deep inside the old
mask, well separated from the own node, reasonably large mask) and the top few are
rendered.

Run:

    py -3 experiments/before_after_overlay.py                    # auto-pick top 4
    py -3 experiments/before_after_overlay.py --old tier2_prop --top 5
    py -3 experiments/before_after_overlay.py --pick AIAL:0:1554 # a specific frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from pipeline.predict import _point_in_mask
from eval.merge_metric import load_node_table, nodes_by_z, DEFAULT_RADIUS

SCALE = 8                       # the _sam grid these trees were scored on
MIN_NEW_AREA = 100              # _sam px: below this the "clean" mask is an invisible speck
OLD_C = (0.0, 0.45, 0.70)       # OLD mask: blue (matches sam3_overlay_disk SAM2 colour)
NEW_C = (0.90, 0.62, 0.0)       # NEW mask: orange
RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")

# Human labels for the panel titles keep the picture tied to the config name.
OLD_TREES = {
    "tier2_prop": ("original_tier2_s1forced_neg_merged", "SAM2 propagation (tier2 s1forced_neg)"),
    "perslice_guard": ("original_perslice_only_guard_merged", "SAM2 per-slice (guard)"),
}
NEW_TREE = ("target_perslice_only_guard_sam3_merged", "SAM3 per-slice (guard)")


# =============================================================================
# Candidate search from the per-frame CSVs
# =============================================================================

def load_candidates(old_dir: str, new_dir: str) -> pd.DataFrame:
    """(neuron, chain_idx, z) frames where OLD merged (n_foreign>=1) and NEW did not.

    Restricted to neurons present in BOTH trees (the shared subset). Includes the
    CSV's own_contained / foreign_ids / bled_fraction so the shortlist can be ranked
    cheaply before any mask is read.
    """
    old = pd.read_csv(RE / old_dir / "_merge_metric.csv")
    new = pd.read_csv(RE / new_dir / "_merge_metric.csv")
    shared = set(old["neuron"].unique()) & set(new["neuron"].unique())
    old = old[old["neuron"].isin(shared)]
    new = new[new["neuron"].isin(shared)]
    key = ["neuron", "chain_idx", "z"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"))
    cand = m[(m["n_foreign_old"] >= 1) & (m["n_foreign_new"] == 0)].copy()
    cand["own_both"] = cand["own_contained_old"].astype(bool) & cand["own_contained_new"].astype(bool)
    return cand


# =============================================================================
# Disk masks (canonical _sam grid) + node geometry, reusing the metric's own defs
# =============================================================================

def chain_masks(tree_dir: str, neuron: str, ci: int) -> dict:
    """{catmaid_z: (mask_sam bool, x0_sam, y0_sam)} for one chain, or {} if absent."""
    cdir = RE / tree_dir / neuron / f"chain_{ci:02d}"
    if not cdir.exists():
        return {}
    return pipeline.chain_masks_in_sam(cdir)


def flagged_nodes(mask, x0, y0, nodes, neuron, radius):
    """(own_xy list, foreign_xy list) for this frame in _sam coords.

    A foreign node is one whose cell_name differs from `neuron` and that falls
    inside `mask` at `radius` (eval.merge_metric.foreign_hits, inlined to keep the
    coordinates, not just the ids)."""
    own, foreign = [], []
    for x, y, cell, _nid in nodes:
        inside = _point_in_mask(mask, x - x0, y - y0, radius)
        if cell == neuron:
            if inside:
                own.append((x, y))
        elif inside:
            foreign.append((x, y))
    return own, foreign


def _containment_depth(mask, x0, y0, xy) -> float:
    """Distance (px) from a node to the nearest mask edge, i.e. how deep inside the
    mask it sits. 0 means it is on/outside the boundary (a borderline touch); a large
    value means the mask clearly swallowed it. Uses the EDT of the mask interior."""
    from scipy import ndimage
    x, y = xy
    xi, yi = int(round(x - x0)), int(round(y - y0))
    h, w = mask.shape
    if not (0 <= yi < h and 0 <= xi < w) or not mask[yi, xi]:
        return 0.0
    return float(ndimage.distance_transform_edt(mask)[yi, xi])


def _bbox(mask, x0, y0):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return (int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0, int(ys.max()) + y0)


# =============================================================================
# Scoring a candidate frame for visual clarity (reads the two masks)
# =============================================================================

def score_frame(masks_old, masks_new, z, nodes, neuron, radius):
    """Metrics for one candidate frame, or None if it is not renderable/clean.

    Returns a dict with the flagged nodes, the deepest foreign containment, the
    own<->foreign separation, and the old mask area. Rejects a frame when the NEW
    mask still contains the flagged foreign node (CSV drift), or when no own node is
    contained in BOTH masks: chain_idx is not a stable identity across the SAM2 and
    SAM3 trees (they hold 629 vs 3899 chains), so requiring a shared own node is what
    guarantees the two panels track the SAME neurite and the difference is only the
    neighbour bleed, not two different branches.
    """
    if z not in masks_old or z not in masks_new:
        return None
    mo, xo, yo = masks_old[z]
    mn, xn, yn = masks_new[z]
    own, foreign = flagged_nodes(mo, xo, yo, nodes, neuron, radius)
    if not foreign:
        return None
    # the NEW mask must be clean on exactly these foreign nodes
    own_n, foreign_n = flagged_nodes(mn, xn, yn, nodes, neuron, radius)
    if foreign_n:
        return None
    # both masks must cover a common own node (same neurite, not a different branch)
    own_shared = [o for o in own if o in own_n]
    if not own_shared:
        return None
    own = own_shared
    # the NEW mask has to be a VISIBLE clean segmentation, not a near-dropout speck:
    # a handful of pixels reads as "the model gave up", not "the merge was fixed".
    if int(mn.sum()) < MIN_NEW_AREA:
        return None
    depths = [_containment_depth(mo, xo, yo, f) for f in foreign]
    best = int(np.argmax(depths))
    sep = 0.0
    if own:
        fx, fy = foreign[best]
        sep = min(float(np.hypot(fx - ox, fy - oy)) for ox, oy in own)
    return {
        "z": z, "own": own, "foreign": foreign,
        "depth": depths[best], "separation": sep,
        "old_area": int(mo.sum()), "new_area": int(mn.sum()),
        "n_foreign": len(foreign),
        "old_bbox": _bbox(mo, xo, yo), "new_bbox": _bbox(mn, xn, yn),
    }


# =============================================================================
# Rendering
# =============================================================================

def _paste(mask, x0, y0, wx0, wy0, ww, wh):
    """Place a full-_sam-frame mask (occupying [y0:y0+h, x0:x0+w]) into a window."""
    out = np.zeros((wh, ww), bool)
    h, w = mask.shape
    ax0, ay0 = max(x0, wx0), max(y0, wy0)
    ax1, ay1 = min(x0 + w, wx0 + ww), min(y0 + h, wy0 + wh)
    if ax1 <= ax0 or ay1 <= ay0:
        return out
    out[ay0 - wy0:ay1 - wy0, ax0 - wx0:ax1 - wx0] = mask[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    return out


def _window(sc, pad, frame_hw):
    """Tight _sam window (wx0, wy0, ww, wh) around both masks and the flagged nodes."""
    boxes = [b for b in (sc["old_bbox"], sc["new_bbox"]) if b is not None]
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]; ys1 = [b[3] for b in boxes]
    for x, y in list(sc["own"]) + list(sc["foreign"]):
        xs0.append(x); ys0.append(y); xs1.append(x); ys1.append(y)
    H, W = frame_hw
    wx0 = max(0, int(min(xs0)) - pad); wy0 = max(0, int(min(ys0)) - pad)
    wx1 = min(W, int(max(xs1)) + pad); wy1 = min(H, int(max(ys1)) + pad)
    return wx0, wy0, wx1 - wx0, wy1 - wy0


def _panel(ax, em_win, mask_win, own, foreign, wx0, wy0, color, title):
    ax.imshow(em_win, cmap="gray")
    if mask_win.any():
        ov = np.zeros((*mask_win.shape, 4)); ov[mask_win] = (*color, 0.45)
        ax.imshow(ov)
    for ox, oy in own:
        ax.scatter([ox - wx0], [oy - wy0], s=150, marker="*", c="#00e000",
                   edgecolors="black", linewidths=0.9, zorder=5)
    for fx, fy in foreign:
        ax.scatter([fx - wx0], [fy - wy0], s=90, marker="x", c="red", linewidths=2.2, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def render(neuron, ci, sc, masks_old, masks_new, old_label, new_label, out_dir, pad):
    z = sc["z"]
    em, full_hw = pipeline.load_frame_sam(int(z), scale=SCALE)
    em = em.mean(axis=2) if em.ndim == 3 else em
    wx0, wy0, ww, wh = _window(sc, pad, em.shape[:2])
    em_win = em[wy0:wy0 + wh, wx0:wx0 + ww]

    mo, xo, yo = masks_old[z]; mn, xn, yn = masks_new[z]
    mo_win = _paste(mo, xo, yo, wx0, wy0, ww, wh)
    mn_win = _paste(mn, xn, yn, wx0, wy0, ww, wh)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), squeeze=True)
    _panel(axes[0], em_win, mo_win, sc["own"], sc["foreign"], wx0, wy0, OLD_C,
           f"OLD  {old_label}\nn_foreign = {sc['n_foreign']}  (merge)")
    _panel(axes[1], em_win, mn_win, sc["own"], sc["foreign"], wx0, wy0, NEW_C,
           f"NEW  {new_label}\nn_foreign = 0  (clean)")
    fig.suptitle(f"{neuron} chain_{ci:02d}  z={z}   "
                 f"green * = own node (kept),  red x = neighbour node (bleed if covered)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{neuron}_chain{ci:02d}_z{z}.png"
    fig.savefig(out_dir / fname, dpi=140)
    plt.close(fig)
    return fname


# =============================================================================
# Driver
# =============================================================================

def gather_and_render(old_key, top, out_dir, pad, shortlist_chains, radius, explicit=None):
    old_dir, old_label = OLD_TREES[old_key]
    new_dir, new_label = NEW_TREE
    annotate_df = load_node_table()
    nbz = nodes_by_z(annotate_df, SCALE)

    results, skipped_null = [], []

    if explicit:
        neuron, ci, z = explicit
        chains = [((neuron, ci), pd.DataFrame([{"z": z}]))]
    else:
        cand = load_candidates(old_dir, new_dir)
        # rank chains by how many single-clean-foreign candidate frames they offer,
        # so the shortlist spreads across chains/neurons rather than one busy chain.
        single = cand[cand["own_both"] & (cand["n_foreign_old"] == 1)]
        order = (single.groupby(["neuron", "chain_idx"]).size()
                 .sort_values(ascending=False).index.tolist())
        chains = []
        for neuron, ci in order[:shortlist_chains]:
            zs = single[(single["neuron"] == neuron) & (single["chain_idx"] == ci)]
            chains.append(((neuron, ci), zs))

    scored = []
    for (neuron, ci), zs in chains:
        masks_old = chain_masks(old_dir, neuron, ci)
        masks_new = chain_masks(new_dir, neuron, ci)
        if not masks_old or not masks_new:
            skipped_null.append((neuron, ci, "missing masks on one side"))
            continue
        for z in zs["z"].astype(int):
            nodes = nbz.get(int(z), [])
            sc = score_frame(masks_old, masks_new, int(z), nodes, neuron, radius)
            if sc is None:
                continue
            sc.update(neuron=neuron, ci=ci)
            scored.append(sc)

    # visual-clarity rank: single foreign, then deep containment, then separation, then area
    scored.sort(key=lambda s: (s["n_foreign"] == 1, s["depth"], s["separation"], s["old_area"]),
                reverse=True)
    # one frame per chain, so the set shows variety rather than adjacent z of one neurite
    seen, deduped = set(), []
    for s in scored:
        k = (s["neuron"], s["ci"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(s)
    picks = scored if explicit else deduped[:top]

    for sc in picks:
        neuron, ci = sc["neuron"], sc["ci"]
        fname = render(neuron, ci, sc,
                       chain_masks(old_dir, neuron, ci), chain_masks(new_dir, neuron, ci),
                       old_label, new_label, out_dir, pad)
        clarity = ("clear" if sc["depth"] >= 6 and sc["separation"] >= 20 and sc["new_area"] >= 200
                   else "moderate" if sc["depth"] >= 3 and sc["new_area"] >= MIN_NEW_AREA
                   else "borderline")
        results.append({
            "file": fname, "neuron": neuron, "chain_idx": ci, "z": sc["z"],
            "old_key": old_key, "n_foreign_old": sc["n_foreign"], "n_foreign_new": 0,
            "depth_px": round(sc["depth"], 1), "separation_px": round(sc["separation"], 1),
            "old_area": sc["old_area"], "new_area": sc["new_area"], "clarity": clarity,
        })
    return results, skipped_null


def write_index(all_results, out_dir):
    html = ("<!doctype html><meta charset=utf-8><title>Before/after merge fix</title>"
            "<body style='font-family:sans-serif;background:#111;color:#eee;max-width:1000px;margin:auto'>"
            "<h2>Before / after: a measured merge fix, made visible</h2>"
            "<p>Left = OLD config (blue mask), right = NEW SAM3 per-slice (orange mask), same EM "
            "crop and same nodes. green star = the chain's own skeleton node (kept in both); "
            "red x = a neighbour neuron's node that the OLD mask swallowed (a merge the metric "
            "counts) and the NEW mask leaves alone. Rendered from saved masks, no model runs.</p>")
    for r in all_results:
        html += (f"<h3>{r['neuron']} chain_{r['chain_idx']:02d}  z={r['z']}  "
                 f"[old={r['old_key']}]</h3>"
                 f"<p>OLD n_foreign = {r['n_foreign_old']} -> NEW n_foreign = {r['n_foreign_new']}; "
                 f"foreign node {r['depth_px']} px inside old mask, {r['separation_px']} px from own "
                 f"node; clarity: <b>{r['clarity']}</b></p>"
                 f"<img src='{r['file']}' style='max-width:100%'>")
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main(argv=None):
    global MIN_NEW_AREA
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="both", choices=[*OLD_TREES, "both"],
                    help="which OLD/bleedier tree to compare against SAM3 (default both)")
    ap.add_argument("--top", type=int, default=4, help="candidates to render per OLD tree")
    ap.add_argument("--shortlist-chains", type=int, default=25,
                    help="chains to read from disk when ranking candidates")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                    help="node-in-mask radius (matches the merge metric)")
    ap.add_argument("--pad", type=int, default=45, help="_sam px padding around the crop window")
    ap.add_argument("--min-new-area", type=int, default=MIN_NEW_AREA,
                    help="drop candidates whose NEW mask is smaller than this (near-dropout speck)")
    ap.add_argument("--pick", default=None, help="render a specific NEURON:chain_idx:z (needs --old)")
    ap.add_argument("--out", default="docs/figures/sam3-bakeoff/before-after")
    args = ap.parse_args(argv)

    MIN_NEW_AREA = args.min_new_area
    out_dir = Path(args.out)
    explicit = None
    if args.pick:
        n, c, z = args.pick.split(":")
        explicit = (n, int(c), int(z))
        if args.old == "both":
            ap.error("--pick needs an explicit --old tree")

    old_keys = list(OLD_TREES) if args.old == "both" else [args.old]
    all_results, all_skipped = [], []
    for ok in old_keys:
        res, skipped = gather_and_render(ok, args.top, out_dir, args.pad,
                                         args.shortlist_chains, args.radius, explicit)
        all_results.extend(res)
        all_skipped.extend((ok, *s) for s in skipped)

    if all_results:
        write_index(all_results, out_dir)

    print(f"\n[before-after] wrote {len(all_results)} figures to {out_dir}")
    for r in all_results:
        print(f"  {r['old_key']:<15} {r['neuron']}/chain_{r['chain_idx']:02d} z={r['z']:<5} "
              f"n_foreign {r['n_foreign_old']}->0  depth={r['depth_px']}px "
              f"sep={r['separation_px']}px area {r['old_area']}->{r['new_area']}  [{r['clarity']}]")
    if all_skipped:
        print(f"[before-after] skipped {len(all_skipped)} chains:")
        for s in all_skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
