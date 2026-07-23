"""Side-by-side SAM2-vs-SAM3 per-slice overlays rendered FROM DISK (no model runs).

Reads the extracted masks from two merged trees, reconstructs the EM crop from the
raw tifs (the frames themselves were not saved), overlays each mask, and marks the
skeleton nodes:

    green *  = the chain's own centreline node at that z (should be INSIDE the mask)
    red   x  = other skeleton nodes at that z (should be OUTSIDE; inside == bleed)

SAM2 mask blue, SAM3 mask orange. One PNG per chain + index.html under --out.
No SAM2/SAM3 inference: it only reads mask PNGs, state.json crop windows, the node
table, and the raw EM. Run:

    py -3 experiments/sam3_overlay_disk.py --chains AIAL:0,AIAL:5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from sam2_utils import review, alignment, config
from pipeline.predict import centreline_by_z, build_prompts
from pipeline.propagate import _node_id_at
from experiments.sam3_bakeoff import load_node_table

# match the per-slice preset (original_perslice_only_guard) so the "neighbours" shown are
# exactly the negatives the pipeline used
_SCALE, _KNEG, _NEGR = 8, 3, 150


def frame_negs_tif(z, x_tif, y_tif, annotate_df):
    """The k nearest same-z neighbour nodes (build_prompts' negatives) in _tif px."""
    nid = _node_id_at(annotate_df, int(z), float(x_tif), float(y_tif))
    if nid is None:
        return np.empty((0, 2))
    p = build_prompts(nid, int(z), annotate_df, scale=_SCALE, k_max_neg=_KNEG, neg_radius=_NEGR)
    labs = np.asarray(p.labels)
    negs_sam = np.asarray(p.points_sam, dtype=float)[labs == 0]
    return negs_sam * _SCALE  # _sam -> _tif

SAM2_C = (0.0, 0.45, 0.70)
SAM3_C = (0.90, 0.62, 0.0)
RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
SAM2_TREE = RE / "original_perslice_only_guard_merged"
SAM3_TREE = RE / "target_perslice_only_guard_sam3_merged"


def load_tree_chain(tree: Path, neuron: str, ci: int):
    cdir = tree / neuron / f"chain_{ci:02d}"
    rd = review.load_chain(cdir, verbose=False)
    cwd = json.loads((cdir / "state.json").read_text()).get("crop_window")
    cw = None
    if cwd:
        cw = alignment.CropWindow(origin_tif=tuple(cwd["origin_tif"]), size_tif=tuple(cwd["size_tif"]),
                                  crop_scale=int(cwd["crop_scale"]), sam_scale=int(cwd["sam_scale"]))
    return rd, cw


def em_crop(z: int, cw):
    img, _ = pipeline.load_frame_sam(int(z), scale=1)
    return img[cw.slice_tif()] if cw is not None else img


def _mask_at(rd, z):
    inv = {v: k for k, v in rd.frame_to_z.items()}
    fi = inv.get(int(z))
    if fi is None or fi not in rd.video_segments:
        return None
    d = rd.video_segments[fi]
    return d.get(rd.obj_id, next(iter(d.values())))


def _overlay(ax, em, mask, own_xy, neg_xy, color, title):
    ax.imshow(em)
    if mask is not None and mask.any():
        import cv2
        m = cv2.resize(mask.astype(np.uint8), (em.shape[1], em.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        ov = np.zeros((*m.shape, 4)); ov[m] = (*color, 0.45)
        ax.imshow(ov)
    if own_xy is not None:
        ax.scatter([own_xy[0]], [own_xy[1]], s=110, marker="*", c="#00e000",
                   edgecolors="black", linewidths=0.8, zorder=5)
    if len(neg_xy):
        ax.scatter(neg_xy[:, 0], neg_xy[:, 1], s=45, marker="x", c="red", linewidths=1.5, zorder=5)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def render_chain(neuron, ci, chain, annotate_df, out_dir):
    rd2, cw2 = load_tree_chain(SAM2_TREE, neuron, ci)
    rd3, cw3 = load_tree_chain(SAM3_TREE, neuron, ci)
    cl = centreline_by_z(chain, annotate_df)
    # frames present in BOTH trees, pick anchor + a spread
    common_z = sorted(set(rd2.frame_to_z.values()) & set(rd3.frame_to_z.values()) & set(cl))
    if not common_z:
        print(f"[disk-overlay] {neuron}/chain_{ci:02d}: no common frames; skip"); return None
    anchor_z = rd3.frame_to_z.get(rd3.anchor_idx) if rd3.anchor_idx is not None else common_z[len(common_z)//2]
    picks = sorted({anchor_z, *[common_z[int(f * (len(common_z) - 1))] for f in (0.15, 0.4, 0.65, 0.9)]})
    picks = [z for z in picks if z in common_z][:5]

    fig, axes = plt.subplots(len(picks), 2, figsize=(7, 3.3 * len(picks)), squeeze=False)
    for r, z in enumerate(picks):
        own = np.asarray(cl[z], float)
        others = frame_negs_tif(z, own[0], own[1], annotate_df)
        for c, (rd, cw, color, tag) in enumerate([(rd2, cw2, SAM2_C, "SAM2"), (rd3, cw3, SAM3_C, "SAM3")]):
            em = em_crop(z, cw)
            o = np.asarray(cw.origin_tif, float) if cw is not None else np.zeros(2)
            own_e = own - o
            neg_e = (others - o) if len(others) else others
            if len(neg_e):
                inb = (neg_e[:, 0] >= 0) & (neg_e[:, 0] < em.shape[1]) & (neg_e[:, 1] >= 0) & (neg_e[:, 1] < em.shape[0])
                neg_e = neg_e[inb]
            at = " (anchor)" if z == anchor_z else ""
            _overlay(axes[r][c], em, _mask_at(rd, z), own_e, neg_e, color, f"{tag}  z={z}{at}")
    fig.suptitle(f"{neuron} chain_{ci:02d}  |  green* = own node (cover), red x = neighbour (bleed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(out_dir) / f"{neuron}_chain{ci:02d}_disk.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[disk-overlay] wrote {out}")
    return out.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default="AIAL:0,AIAL:5")
    ap.add_argument("--out", default="docs/figures/sam3-bakeoff/node-overlays")
    args = ap.parse_args(argv)

    annotate_df = load_node_table()
    with open(config.CHAINS_PATH) as f:
        chains = json.load(f)
    from experiments.sam3_bakeoff import enumerate_chains, parse_chains
    lookup = {(n, i): c for n, i, c in enumerate_chains(chains, neurons=None)}
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for neuron, ci in parse_chains(args.chains):
        if (neuron, ci) not in lookup:
            print(f"[disk-overlay] {neuron}/chain_{ci:02d} not in chains.json; skip"); continue
        if not (SAM2_TREE / neuron).exists():
            print(f"[disk-overlay] {neuron} not in SAM2 baseline tree (16-neuron subset); skip"); continue
        nm = render_chain(neuron, ci, lookup[(neuron, ci)], annotate_df, out_dir)
        if nm:
            names.append(nm)

    if names:
        html = ("<!doctype html><meta charset=utf-8><title>SAM2 vs SAM3 node overlays</title>"
                "<body style='font-family:sans-serif;background:#111;color:#eee'>"
                "<h2>SAM2 (blue) vs SAM3 (orange) per-slice, zoomed, with skeleton nodes</h2>"
                "<p>green star = this chain's node (should be covered); red x = neighbour nodes "
                "(bleed if covered). Rendered from the saved masks, no model runs.</p>")
        for nm in names:
            html += f"<h3>{nm}</h3><img src='{nm}' style='max-width:100%'>"
        (out_dir / "index_disk.html").write_text(html, encoding="utf-8")
        print(f"[disk-overlay] index -> {out_dir/'index_disk.html'}")


if __name__ == "__main__":
    main()
