"""Zoomed, node-annotated SAM2-vs-SAM3 per-slice overlays for a few chains.

Reuses the bake-off per-slice machinery so the two masks are aligned (same anchor,
same frames, cw=None whole-frame scale-8). For each picked frame it crops the view
to the neuron (mask + node bbox) so the object fills the panel, and marks the
skeleton nodes:

    green star = this chain's own centreline node at that z (should be INSIDE the mask)
    red x      = same-z neighbour nodes (should be OUTSIDE; inside == bleed)

SAM2 mask is drawn blue, SAM3 mask orange (matching the earlier plotly). Output is
one PNG per chain plus an index.html, under --out.

Run (GPU + EM on WORM_PATH + the SAM3 checkpoint):
    py -3 experiments/sam3_overlay_nodes.py --chains AIAL:0,AIAL:5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on the path so `pipeline`, `sam2_utils`, and `experiments.*` resolve whether
# this is run as a script (py -3 experiments/sam3_overlay_nodes.py) or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from sam2_utils import alignment, setup, config
from pipeline.config import PipelineConfig
from pipeline.predict import build_prompts
from pipeline.propagate import _node_id_at
from sam2_utils.sam3_backend import DEFAULT_CHECKPOINT_DIR

from experiments.sam3_bakeoff import (
    build_chain_inputs, run_sam2_perslice, run_sam3_perslice,
    load_node_table, enumerate_chains, parse_chains, _free_gpu,
)

SAM2_C = (0.0, 0.45, 0.70)    # Okabe-Ito blue
SAM3_C = (0.90, 0.62, 0.0)    # Okabe-Ito orange


def frame_nodes_sam(z, centreline_tif, annotate_df, cfg):
    """(own (2,), negs (M,2)) skeleton nodes at catmaid z, in _sam, exactly the ones
    segment_per_slice uses: the chain's centreline point plus its same-z neighbour
    negatives."""
    x_tif, y_tif = centreline_tif[z]
    own = alignment.tif_to_sam([x_tif, y_tif], cfg.scale).reshape(2)
    negs = np.empty((0, 2))
    if cfg.k_max_neg > 0:
        nid = _node_id_at(annotate_df, z, x_tif, y_tif)
        if nid is not None:
            p = build_prompts(nid, z, annotate_df, scale=cfg.scale,
                              k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)
            labs = np.asarray(p.labels)
            pts = np.asarray(p.points_sam, dtype=float)
            negs = pts[labs == 0]
    return own, negs


def _mask(seg, i, obj_id):
    d = seg.get(i)
    if not d:
        return None
    return d.get(obj_id, next(iter(d.values())))


def pick_frames(seg2, seg3, frame_to_z, anchor_idx, obj_id, k=4):
    """anchor + the frames where SAM2 and SAM3 masks disagree most (by area), so the
    panel shows both a typical slice and the slices where they diverge."""
    diffs = []
    for i in frame_to_z:
        m2, m3 = _mask(seg2, i, obj_id), _mask(seg3, i, obj_id)
        if m2 is None or m3 is None:
            continue
        diffs.append((abs(int(m2.sum()) - int(m3.sum())), i))
    diffs.sort(reverse=True)
    picks = [anchor_idx] + [i for _, i in diffs if i != anchor_idx]
    seen, out = set(), []
    for i in picks:
        if i not in seen and i in frame_to_z:
            seen.add(i); out.append(i)
        if len(out) == k:
            break
    return sorted(out)


def bbox(masks, pts, hw, pad=18):
    ys, xs = [], []
    for m in masks:
        if m is not None and m.any():
            yy, xx = np.where(m); ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    for p in pts:
        if len(p):
            xs += list(p[:, 0]); ys += list(p[:, 1])
    H, W = hw
    if not xs:
        return 0, 0, W, H
    x0 = max(int(min(xs)) - pad, 0); y0 = max(int(min(ys)) - pad, 0)
    x1 = min(int(max(xs)) + pad, W); y1 = min(int(max(ys)) + pad, H)
    return x0, y0, x1, y1


def _panel(ax, em, mask, own, negs, color, title, x0, y0):
    ax.imshow(em, cmap="gray" if em.ndim == 2 else None)
    if mask is not None and mask.any():
        ov = np.zeros((*mask.shape, 4))
        ov[mask] = (*color, 0.45)
        ax.imshow(ov)
    ax.scatter([own[0] - x0], [own[1] - y0], s=90, marker="*",
               c="#00e000", edgecolors="black", linewidths=0.7, zorder=5)
    if len(negs):
        ax.scatter(negs[:, 0] - x0, negs[:, 1] - y0, s=40, marker="x",
                   c="red", linewidths=1.4, zorder=5)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def render_chain(neuron, chain_idx, seg2, seg3, inputs, annotate_df, cfg, out_dir):
    obj_id = inputs.obj_id
    picks = pick_frames(seg2, seg3, inputs.frame_to_z, inputs.anchor_frame_idx, obj_id)
    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.3 * n), squeeze=False)
    for r, i in enumerate(picks):
        z = inputs.frame_to_z[i]
        em = cv2.cvtColor(cv2.imread(str(Path(inputs.frames_dir) / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        m2, m3 = _mask(seg2, i, obj_id), _mask(seg3, i, obj_id)
        own, negs = frame_nodes_sam(z, inputs.centreline_tif, annotate_df, cfg)
        x0, y0, x1, y1 = bbox([m2, m3], [own.reshape(1, 2), negs], em.shape[:2])
        crop = lambda a: a[y0:y1, x0:x1] if a is not None else None
        tag = " (anchor)" if i == inputs.anchor_frame_idx else ""
        _panel(axes[r][0], crop(em), crop(m2), own, negs, SAM2_C, f"SAM2  z={z}{tag}", x0, y0)
        _panel(axes[r][1], crop(em), crop(m3), own, negs, SAM3_C, f"SAM3  z={z}{tag}", x0, y0)
    fig.suptitle(f"{neuron} chain_{chain_idx:02d}  |  green* = own node (cover), red x = neighbour (bleed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(out_dir) / f"{neuron}_chain{chain_idx:02d}_nodes.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[overlay] wrote {out}")
    return out.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default="AIAL:0,AIAL:5")
    ap.add_argument("--root", default=str(config.OUTPUT_ROOT))
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--out", default="docs/figures/sam3-bakeoff/node-overlays")
    args = ap.parse_args(argv)

    cfg = PipelineConfig(scale=8, save_downscale=8, k_max_neg=3, neg_radius=150, box_margin=10,
                         output_root=Path(args.root), frames_root=config.FRAMES_ROOT)
    annotate_df = load_node_table()
    import json
    with open(config.CHAINS_PATH) as f:
        chains = json.load(f)
    lookup = {(n, i): c for n, i, c in enumerate_chains(chains, neurons=None)}
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for neuron, chain_idx in parse_chains(args.chains):
        key = (neuron, chain_idx)
        if key not in lookup:
            print(f"[overlay] {neuron}/chain_{chain_idx:02d} not in chains.json; skip"); continue
        anchor_ip, _ = setup.build_predictor(size=cfg.model_size, kind="image", image_size=cfg.image_size)
        try:
            inputs = build_chain_inputs(neuron, chain_idx, lookup[key], annotate_df, cfg, anchor_ip)
        finally:
            del anchor_ip
            _free_gpu()
        seg2 = run_sam2_perslice(inputs, cfg, annotate_df)
        _free_gpu()                              # release SAM2 VRAM before SAM3 loads (6GB card)
        seg3 = run_sam3_perslice(inputs, cfg, annotate_df, args.checkpoint)
        _free_gpu()
        names.append(render_chain(neuron, chain_idx, seg2, seg3, inputs, annotate_df, cfg, out_dir))
        _free_gpu()

    if names:
        html = "<!doctype html><meta charset=utf-8><title>SAM2 vs SAM3 node overlays</title>"
        html += "<body style='font-family:sans-serif;background:#111;color:#eee'>"
        html += "<h2>SAM2 (blue) vs SAM3 (orange) per-slice, zoomed, with skeleton nodes</h2>"
        html += "<p>green star = this chain's node (should be covered); red x = neighbour nodes (bleed if covered)</p>"
        for nm in names:
            html += f"<h3>{nm}</h3><img src='{nm}' style='max-width:100%'>"
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        print(f"[overlay] index -> {out_dir/'index.html'}")


if __name__ == "__main__":
    main()
