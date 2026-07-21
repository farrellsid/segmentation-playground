"""Per-frame neuron segmentation driver. Approach 1 (prompt-based) here; Approach 2 (AMG)
in a later change. Segments every node-bearing cell in a frame, resolves overlaps
membrane-aware, scores with eval.perframe_score, and writes results/montages. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md

This is a DRIVER (like batch.py / run_aval.py): it may import the library (pipeline,
sam2_utils) and eval freely. The library must never import this file back
(tests/test_import_direction.py enforces that direction).

Run it directly, e.g.:
    py -3 run_perframe.py --approach prompt --frames 1400 1420 --negatives on \\
        --selection metric --resolver argmax --scale 8 --model-size tiny \\
        --out results/perframe/smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import perframe as pf, membrane as mb, setup
from eval.perframe_score import score_frame


@dataclass
class PerframeCfg:
    scale: int = 8
    radius: int = 3
    tau: float = 0.5
    k_max_neg: int = 3
    box_margin: int = 10


def segment_frame_prompt(image_predictor, frame_sam, node_index, membrane_map, *,
                         negatives: bool, selection: str, resolver: str, cfg: PerframeCfg):
    """One frame, prompt mode. For each node: set_image, predict with a positive point (+
    box, + the OTHER cells' nodes as negatives when `negatives`), take SAM2's 3 candidates,
    pick one by `selection` (pred_iou | generous | metric), collect labelled masks, resolve
    overlaps by `resolver` (argmax | watershed), score. Returns (cell_masks, label_map, score).

    negatives / selection / foreign nodes are all computed per node from `node_index`
    (a list of (x, y, cell_name, node_id) in `frame_sam`'s space, e.g. pf.nodes_in_frame's
    output); "foreign" means a node belonging to a DIFFERENT cell, never a second node of
    the same cell. When several nodes share a cell (a branch point in this frame), each
    node gets its own chosen mask and the cell's final mask is their union.
    """
    if selection not in ("pred_iou", "generous", "metric"):
        raise ValueError(f"unknown selection {selection!r}")
    if resolver not in ("argmax", "watershed"):
        raise ValueError(f"unknown resolver {resolver!r}")

    h, w = frame_sam.shape[:2]
    image_predictor.set_image(frame_sam)

    masks_in_order: list[np.ndarray] = []
    node_xy_in_order: list[tuple[float, float]] = []
    cell_masks: dict[str, np.ndarray] = {}

    for (x, y, cell, _node_id) in node_index:
        foreign_xy = [(fx, fy) for (fx, fy, fc, _fn) in node_index if fc != cell]

        pos = np.array([[float(x), float(y)]], dtype=float)
        pos_labels = np.array([1], dtype=int)
        if negatives and foreign_xy:
            neg = np.asarray(foreign_xy, dtype=float)
            if len(neg) > cfg.k_max_neg:
                d = (neg[:, 0] - x) ** 2 + (neg[:, 1] - y) ** 2
                neg = neg[np.argsort(d)[:cfg.k_max_neg]]
            pts = np.concatenate([pos, neg], axis=0)
            labs = np.concatenate([pos_labels, np.zeros(len(neg), dtype=int)])
        else:
            pts, labs = pos, pos_labels

        # first-pass single mask on the positive point alone, to size a box (mirrors
        # pipeline.box_from_mask's normal use: box AFTER a raw anchor mask, not before).
        m0, _s0, _l0 = image_predictor.predict(
            point_coords=pos, point_labels=pos_labels, box=None, multimask_output=False)
        mask0 = np.asarray(m0[0]).astype(bool)
        box = pipeline.box_from_mask(mask0, margin=cfg.box_margin, image_hw_sam=(h, w))

        masks, scores, _logits = image_predictor.predict(
            point_coords=pts, point_labels=labs, box=box, multimask_output=True)
        cands = [np.asarray(m).astype(bool) for m in masks]
        scores = np.asarray(scores, dtype=float).ravel()

        if selection == "pred_iou":
            idx = int(np.argmax(scores)) if scores.size else 0
        elif selection == "generous":
            containing = [i for i, m in enumerate(cands)
                         if pipeline._point_in_mask(m, x, y, cfg.radius)]
            idx = (max(containing, key=lambda i: int(cands[i].sum()))
                  if containing else int(np.argmax(scores)) if scores.size else 0)
        else:  # metric
            idx = pf.select_by_metric(cands, (x, y), foreign_xy, membrane_map,
                                      radius=cfg.radius, tau=cfg.tau)
            if idx < 0:
                idx = int(np.argmax(scores)) if scores.size else 0

        chosen = cands[idx]
        masks_in_order.append(chosen)
        node_xy_in_order.append((x, y))
        cell_masks[cell] = (cell_masks[cell] | chosen) if cell in cell_masks else chosen

    image_predictor.reset_predictor()

    if resolver == "argmax":
        label_map = pf.resolve_overlaps_argmax(masks_in_order, node_xy_in_order, membrane_map)
    else:
        label_map = pf.resolve_overlaps_watershed(masks_in_order, node_xy_in_order, membrane_map)

    score = score_frame(cell_masks, node_index, membrane_map, radius=cfg.radius, tau=cfg.tau)
    return cell_masks, label_map, score


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _write_montage(path: Path, frame_sam: np.ndarray, label_map: np.ndarray,
                   membrane_map: np.ndarray, node_index) -> None:
    """EM | coloured label map | membrane overlay, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    gray = frame_sam.mean(axis=2) if frame_sam.ndim == 3 else frame_sam

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("EM")
    for (x, y, cell, _n) in node_index:
        axes[0].plot(x, y, "rx", markersize=4)
        axes[0].annotate(cell, (x, y), color="red", fontsize=6)

    axes[1].imshow(gray, cmap="gray")
    masked = np.ma.masked_where(label_map == 0, label_map)
    axes[1].imshow(masked, cmap="tab20", alpha=0.6, interpolation="nearest")
    axes[1].set_title("labelled instances")

    axes[2].imshow(gray, cmap="gray")
    axes[2].imshow(membrane_map, cmap="hot", alpha=0.5, interpolation="nearest")
    axes[2].set_title("membrane map")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _append_experiment_log(out_dir: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    log_path = Path("docs/explanation/perframe-experiments.md")
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "# Per-frame segmentation experiments\n\n"
            "A running log of run_perframe.py sweeps: config, headline scores, and a "
            "pointer to the montages. Design:\n"
            "docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md\n\n")

    if rows:
        mean_cov = float(np.mean([r["own_coverage"] for r in rows]))
        mean_overlap = float(np.mean([r["overlap_fraction"] for r in rows]))
        mean_foreign = float(np.mean([r["total_foreign"] for r in rows]))
    else:
        mean_cov = mean_overlap = mean_foreign = float("nan")

    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = (
        f"- {when} `{out_dir.as_posix()}`: approach={args.approach} negatives={args.negatives} "
        f"selection={args.selection} resolver={args.resolver} scale={args.scale} "
        f"model_size={args.model_size} frames={args.frames} -> "
        f"own_coverage={mean_cov:.3f} overlap_fraction={mean_overlap:.4f} "
        f"total_foreign={mean_foreign:.2f} (montages: `{(out_dir / 'montages').as_posix()}`)\n"
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry)


def _run(args: argparse.Namespace) -> None:
    from eval import merge_metric

    cfg = PerframeCfg(scale=args.scale, radius=args.radius, tau=args.tau,
                      k_max_neg=args.k_max_neg, box_margin=args.box_margin)
    negatives = args.negatives == "on"

    out_dir = Path(args.out)
    montage_dir = out_dir / "montages"
    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)

    image_predictor, device = setup.build_predictor(size=args.model_size, kind="image")
    annotate_df = merge_metric.load_node_table()

    rows: list[dict] = []
    for z in args.frames:
        frame_sam, _full_hw = pipeline.load_frame_sam(int(z), scale=cfg.scale)
        gray = frame_sam.mean(axis=2) if frame_sam.ndim == 3 else frame_sam
        mem = mb.membrane_map(gray)
        node_index = pf.nodes_in_frame(annotate_df, int(z), cfg.scale)
        if not node_index:
            print(f"[run_perframe] z={z}: no nodes in frame, skipping")
            continue

        _cell_masks, label_map, score = segment_frame_prompt(
            image_predictor, frame_sam, node_index, mem,
            negatives=negatives, selection=args.selection, resolver=args.resolver, cfg=cfg)

        print(f"[run_perframe] z={z} n_cells={score['n_cells']} "
             f"own_coverage={score['own_coverage']:.3f} "
             f"total_foreign={score['total_foreign']} "
             f"overlap_fraction={score['overlap_fraction']:.4f}")

        rows.append({
            "z": int(z),
            "n_cells": score["n_cells"],
            "own_coverage": score["own_coverage"],
            "foreign_frame_rate": score["foreign_frame_rate"],
            "total_foreign": score["total_foreign"],
            "overlap_fraction": score["overlap_fraction"],
            "mean_boundary_on_membrane": score["mean_boundary_on_membrane"],
            "spanning_rate": score["spanning_rate"],
            "mean_underfill": score["mean_underfill"],
        })
        _write_montage(montage_dir / f"{z}.png", frame_sam, label_map, mem, node_index)

    pd.DataFrame(rows).to_csv(out_dir / "scores.csv", index=False)

    config_json = {
        "approach": args.approach,
        "frames": list(args.frames),
        "negatives": args.negatives,
        "selection": args.selection,
        "resolver": args.resolver,
        "scale": cfg.scale,
        "model_size": args.model_size,
        "radius": cfg.radius,
        "tau": cfg.tau,
        "k_max_neg": cfg.k_max_neg,
        "box_margin": cfg.box_margin,
        "device": str(device),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    _append_experiment_log(out_dir, args, rows)
    print(f"[run_perframe] wrote {out_dir / 'config.json'}, {out_dir / 'scores.csv'}, "
         f"{len(rows)} montage(s) under {montage_dir}")


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-frame neuron segmentation. Approach 1 (prompt-based): image-mode "
                    "SAM2 once per node, membrane-aware overlap resolution, F2 scoring.")
    ap.add_argument("--approach", choices=["prompt"], default="prompt",
                    help="segmentation approach (only 'prompt' exists so far)")
    ap.add_argument("--frames", nargs="+", type=int, required=True,
                    help="CATMAID z's to segment")
    ap.add_argument("--negatives", choices=["on", "off"], default="on",
                    help="pass other cells' nodes as negative points")
    ap.add_argument("--selection", choices=["pred_iou", "generous", "metric"], default="metric",
                    help="how to pick among SAM2's 3 multimask candidates")
    ap.add_argument("--resolver", choices=["argmax", "watershed"], default="argmax",
                    help="overlap-resolution method (F3)")
    ap.add_argument("--scale", type=int, default=8, help="downscale factor (_sam grid)")
    ap.add_argument("--model-size", default="tiny", help="SAM2 checkpoint size")
    ap.add_argument("--out", required=True, help="results dir, e.g. results/perframe/<run>")
    ap.add_argument("--radius", type=int, default=PerframeCfg.radius,
                    help="node-containment radius, px (_sam)")
    ap.add_argument("--tau", type=float, default=PerframeCfg.tau,
                    help="membrane threshold on the normalised [0, 1] map")
    ap.add_argument("--k-max-neg", type=int, default=PerframeCfg.k_max_neg,
                    help="cap on negative points per node (nearest first)")
    ap.add_argument("--box-margin", type=int, default=PerframeCfg.box_margin,
                    help="fixed px pad for the first-pass box")
    return ap.parse_args(argv)


if __name__ == "__main__":
    _run(_parse())
