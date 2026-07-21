"""Per-frame neuron segmentation driver. Approach 1 (prompt-based, image-mode SAM2 per
node) and Approach 2 (SAM2AutomaticMaskGenerator, match to nodes, keep the rest as
competitors) both live here. Segments every node-bearing cell in a frame, resolves
overlaps membrane-aware, scores with eval.perframe_score, and writes results/montages.
Design: docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md

This is a DRIVER (like batch.py / run_aval.py): it may import the library (pipeline,
sam2_utils) and eval freely. The library must never import this file back
(tests/test_import_direction.py enforces that direction).

Run it directly, e.g. Approach 1:
    py -3 run_perframe.py --approach prompt --frames 1400 1420 --negatives on \\
        --selection metric --resolver argmax --scale 8 --model-size tiny \\
        --out results/perframe/smoke

Or Approach 2 (auto-mask):
    py -3 run_perframe.py --approach amg --frames 1400 1420 --match metric \\
        --resolver argmax --scale 8 --model-size tiny --out results/perframe/amg_smoke

Or sweep the Approach-1 knob grid (12 combos) over the given frames, one subdirectory and one
experiments-log row per combo:
    py -3 run_perframe.py --approach prompt --sweep --frames 1400 1420 --scale 8 \\
        --model-size tiny --out results/perframe/sweep

Or tune Approach 2's AMG params (12-combo default grid, or --tune-grid) over the given frames,
keeping the trial that maximises eval.perframe_score.objective:
    py -3 run_perframe.py --approach amg --tune --frames 1400 1420 --scale 8 \\
        --model-size tiny --out results/perframe/tune
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pipeline
from sam2_utils import perframe as pf, membrane as mb, setup
from eval.perframe_score import score_frame, objective, pairwise_overlap_fraction

try:
    import torch
except ImportError:  # torch-free environments (e.g. the FakePred smoke test)
    torch = None


def _predict_ctx(image_predictor):
    """torch.inference_mode() around predict calls, skipped for torch-free/fake predictors.

    A real SAM2ImagePredictor holds its model at `.model`; the smoke test's FakePred
    does not, so this stays a no-op there and the torch-free path keeps working.
    """
    if torch is not None and hasattr(image_predictor, "model"):
        return torch.inference_mode()
    return nullcontext()


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
    overlaps by `resolver` (argmax | watershed), score. Returns
    (resolved_cell_masks, label_map, score).

    negatives / selection / foreign nodes are all computed per node from `node_index`
    (a list of (x, y, cell_name, node_id) in `frame_sam`'s space, e.g. pf.nodes_in_frame's
    output); "foreign" means a node belonging to a DIFFERENT cell, never a second node of
    the same cell. When several nodes share a cell (a branch point in this frame), each
    node gets its own chosen mask and the cell's raw (pre-resolution) mask is their union.

    Scoring contract (shared with segment_frame_amg, so the two approaches are comparable):
    `score_frame` grades the RESOLVED masks (`label_map == i+1` per cell, unioned over that
    cell's nodes), not the raw pre-resolution unions, so `resolver` actually moves
    own_coverage/foreign/boundary/spanning/underfill. `overlap_fraction` is then overridden
    with a pre-resolution diagnostic (the pairwise overlap of the raw per-cell union masks),
    since it would otherwise read ~0 by construction on the disjoint resolved masks.
    """
    if selection not in ("pred_iou", "generous", "metric"):
        raise ValueError(f"unknown selection {selection!r}")
    if resolver not in ("argmax", "watershed"):
        raise ValueError(f"unknown resolver {resolver!r}")

    h, w = frame_sam.shape[:2]
    with _predict_ctx(image_predictor):
        image_predictor.set_image(frame_sam)

    masks_in_order: list[np.ndarray] = []
    node_xy_in_order: list[tuple[float, float]] = []
    cell_in_order: list[str] = []
    raw_cell_masks: dict[str, np.ndarray] = {}

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

        # first-pass single mask on the FULL prompt set (positive + negatives, capped), to
        # size a box: mirrors pipeline/orchestrator.py's pattern of building the box-seeding
        # mask from the full prompt set, not the positive point alone (an oversized/bled box
        # in crowded frames would badly condition the real multimask predict below). Then
        # box_from_mask AFTER, exactly as pipeline.box_from_mask's normal use.
        with _predict_ctx(image_predictor):
            m0, _s0, _l0 = image_predictor.predict(
                point_coords=pts, point_labels=labs, box=None, multimask_output=False)
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
        cell_in_order.append(cell)
        raw_cell_masks[cell] = (raw_cell_masks[cell] | chosen) if cell in raw_cell_masks else chosen

    image_predictor.reset_predictor()

    if resolver == "argmax":
        label_map = pf.resolve_overlaps_argmax(masks_in_order, node_xy_in_order, membrane_map)
    else:
        label_map = pf.resolve_overlaps_watershed(masks_in_order, node_xy_in_order, membrane_map)

    # Resolved per-cell masks, per the shared scoring contract: label_map is indexed by
    # NODE (i+1 = masks_in_order[i]), so a cell with several nodes (a branch point) unions
    # its nodes' resolved slices back together. Every node-bearing cell keeps a key here
    # even if its node lost the whole overlap fight (an all-False mask), matching
    # segment_frame_amg's fairness fix for unmatched cells.
    resolved_cell_masks: dict[str, np.ndarray] = {cell: np.zeros((h, w), dtype=bool)
                                                  for cell in raw_cell_masks}
    for i, cell in enumerate(cell_in_order):
        resolved_cell_masks[cell] |= (label_map == (i + 1))

    score = score_frame(resolved_cell_masks, node_index, membrane_map,
                        radius=cfg.radius, tau=cfg.tau)
    # overlap_fraction on resolved (disjoint-by-construction) masks would read ~0
    # regardless of resolver, so override it with the pre-resolution diagnostic: how much
    # the raw per-cell union masks fought over the same pixels before resolution.
    score["overlap_fraction"] = pairwise_overlap_fraction(list(raw_cell_masks.values()))
    return resolved_cell_masks, label_map, score


# Approach 2's AMG defaults, matching the notebook's mask_generator_2. Overridable per run
# via --amg-params <json>.
DEFAULT_AMG_PARAMS = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.7,
    "stability_score_thresh": 0.92,
    "stability_score_offset": 0.7,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 25,
    "use_m2m": True,
}


def build_amg(sam2_model, **amg_params):
    """Thin wrapper: SAM2AutomaticMaskGenerator(sam2_model, **amg_params). Kept here (driver)
    so the library (sam2_utils/perframe.py) stays torch-free. amg_params: points_per_side,
    pred_iou_thresh, stability_score_thresh, stability_score_offset, box_nms_thresh,
    crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area, use_m2m.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(model=sam2_model, **amg_params)


def _match_amg_to_nodes_area(amg_masks, node_index, *, radius: int):
    """'area' matching: per node, the smallest AMG mask that contains it (ties broken by
    scan order), independent of the F2 composite. Returns (labels, leftover), same shape
    as pf.match_amg_to_nodes.

    Unlike pf.match_amg_to_nodes, this has no cross-node foreign exclusion: if one
    under-segmented AMG blob is the smallest containing mask for two different nodes,
    both cells match that same blob here, and the resolver later splits it by nearest
    seed. That yields a deceptively perfect own_coverage on what is really a merged
    blob, so treat 'area' results with suspicion and prefer 'metric' when the match
    itself needs to be trusted.
    """
    labels: dict[str, np.ndarray] = {}
    used = set()
    for (x, y, cell, _nid) in node_index:
        containing = [i for i, m in enumerate(amg_masks)
                     if pipeline._point_in_mask(m, float(x), float(y), radius)]
        if not containing:
            continue
        idx = min(containing, key=lambda i: int(amg_masks[i].sum()))
        labels[cell] = amg_masks[idx]
        used.add(idx)
    leftover = [m for i, m in enumerate(amg_masks) if i not in used]
    return labels, leftover


def segment_frame_amg(amg, frame_sam, node_index, membrane_map, *,
                      match: str, resolver: str, cfg):
    """One frame, AMG mode. Runs amg.generate(frame_sam), matches each node to one of the
    resulting masks, and keeps the rest as unlabelled competitors that still take part in
    overlap resolution (so they can push bleed off a cell mask) before being dropped to
    background in the returned label map.

    match: 'metric' uses pf.match_amg_to_nodes (the F2-composite matcher); 'area' picks,
    per node, the smallest AMG mask containing it. resolver: 'argmax' | 'watershed', same
    as segment_frame_prompt. Returns (resolved_cell_masks, label_map, score); the returned
    label_map is cell-only (0 background, competitor labels already zeroed out after
    resolution).

    Every distinct cell name in node_index is guaranteed a key in the returned cell_masks,
    even one AMG never matched: unmatched cells get an all-False mask of the frame shape,
    so score_frame counts them as uncovered rather than omitting them from own_coverage's
    mean. This keeps Approach 2 comparable to Approach 1, which attempts every node and
    already lands an empty prediction in cell_masks the same way.

    Scoring contract (shared with segment_frame_prompt, so the two approaches are
    comparable): `score_frame` grades these already-resolved masks (disjoint by
    construction, the resolve_overlaps_* output), so `resolver` moves own_coverage/foreign/
    boundary/spanning/underfill here too. `overlap_fraction` is overridden with a
    pre-resolution diagnostic (the pairwise overlap of the matched-but-not-yet-resolved
    per-cell masks, i.e. `labels` below), since it would otherwise read ~0 by construction.
    """
    if match not in ("area", "metric"):
        raise ValueError(f"unknown match {match!r}")
    if resolver not in ("argmax", "watershed"):
        raise ValueError(f"unknown resolver {resolver!r}")

    h, w = frame_sam.shape[:2]
    all_cells = {cell for (_x, _y, cell, _n) in node_index}
    anns = amg.generate(frame_sam)
    amg_masks = [np.asarray(a["segmentation"]).astype(bool) for a in anns]

    if not amg_masks:
        label_map = np.zeros((h, w), dtype=np.int32)
        cell_masks: dict[str, np.ndarray] = {c: np.zeros((h, w), dtype=bool) for c in all_cells}
        score = score_frame(cell_masks, node_index, membrane_map, radius=cfg.radius, tau=cfg.tau)
        score["overlap_fraction"] = 0.0
        return cell_masks, label_map, score

    if match == "metric":
        labels, leftover = pf.match_amg_to_nodes(amg_masks, node_index, membrane_map,
                                                  radius=cfg.radius, tau=cfg.tau)
    else:
        labels, leftover = _match_amg_to_nodes_area(amg_masks, node_index, radius=cfg.radius)

    # Pre-resolution overlap diagnostic, taken before the matched masks compete in
    # resolve_overlaps below: the same representation Approach 1 uses for its raw
    # per-cell union masks, so the two approaches' overlap_fraction means the same thing.
    raw_overlap_fraction = pairwise_overlap_fraction(list(labels.values()))

    # Resolution order: labelled cells first (seed = their node's xy), then competitors
    # (seed = their own mask centroid), so competitors take part in the fight for pixels
    # and can push bleed off a cell, but never appear as a named cell afterwards.
    order_masks: list[np.ndarray] = []
    order_xy: list[tuple[float, float]] = []
    order_names: list[Optional[str]] = []

    for cell, mask in labels.items():
        x, y = next((nx, ny) for (nx, ny, nc, _n) in node_index if nc == cell)
        order_masks.append(mask)
        order_xy.append((x, y))
        order_names.append(cell)

    for mask in leftover:
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        order_masks.append(mask)
        order_xy.append((float(xs.mean()), float(ys.mean())))
        order_names.append(None)

    if resolver == "argmax":
        label_map_full = pf.resolve_overlaps_argmax(order_masks, order_xy, membrane_map)
    else:
        label_map_full = pf.resolve_overlaps_watershed(order_masks, order_xy, membrane_map)

    cell_masks = {}
    label_map = np.zeros((h, w), dtype=np.int32)
    for i, name in enumerate(order_names):
        if name is None:
            continue
        m = label_map_full == (i + 1)
        cell_masks[name] = m
        label_map[m] = i + 1

    # Fairness fix: a cell with a node in this frame that AMG failed to match must still
    # show up in cell_masks, as an empty mask, so score_frame counts it as uncovered
    # (own_contained=False) instead of silently dropping it and inflating own_coverage.
    # Approach 1 has no such gap (it attempts every node, so an empty prediction there
    # already lands in cell_masks); this keeps the two approaches comparable. Never added
    # to label_map, since an empty mask contributes no pixels there anyway.
    for cell in all_cells:
        if cell not in cell_masks:
            cell_masks[cell] = np.zeros((h, w), dtype=bool)

    score = score_frame(cell_masks, node_index, membrane_map, radius=cfg.radius, tau=cfg.tau)
    score["overlap_fraction"] = raw_overlap_fraction
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


_EXPERIMENT_LOG_HEADER = (
    "# Per-frame segmentation experiments\n\n"
    "A running log of `run_perframe.py` runs, including sweeps over the Approach-1 knob grid. "
    "Each run writes its full output under `results/perframe/<run>/`: `config.json` (the exact "
    "knobs, git commit, and command line), `scores.csv` (one row per frame), and `montages/` "
    "(one EM / labelled-instance / membrane-overlay figure per frame). That directory is "
    "gitignored, since it is regenerable from the config; this table is the committed record of "
    "what each run tried and how it scored. Design:\n"
    "docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md\n\n"
    "Each table row summarises one run as the mean of its per-frame scores (`own_coverage`, "
    "`total_foreign`, `mean_boundary_on_membrane`, `overlap_fraction`; see "
    "`eval/perframe_score.py` for what each one measures). `notes` is for anything the numbers "
    "do not capture, filled in by hand.\n\n"
    "| run | approach | negatives | selection | resolver | frames | own_coverage | "
    "total_foreign | mean_boundary_on_membrane | overlap_fraction | notes |\n"
    "|-----|----------|-----------|-----------|----------|--------|---------------|"
    "----------------|----------------------------|-------------------|-------|\n"
)


_EXPERIMENT_LOG_PATH = Path("docs/explanation/perframe-experiments.md")


def _append_experiment_log(out_dir: Path, *, approach: str, negatives: str, selection: str,
                           resolver: str, frames, rows: list[dict], notes: str = "") -> None:
    """Append one summary row for this run to the committed experiments table
    (`_EXPERIMENT_LOG_PATH`, normally docs/explanation/perframe-experiments.md; a module
    constant so tests can point it at a scratch file instead of the real doc). The summary
    is the mean of the per-frame `rows`; `notes` defaults to blank for a human to fill in,
    but a caller (e.g. --tune) can pass one, such as a gameable-objective warning.
    """
    log_path = _EXPERIMENT_LOG_PATH
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(_EXPERIMENT_LOG_HEADER)

    if rows:
        mean_cov = float(np.mean([r["own_coverage"] for r in rows]))
        mean_overlap = float(np.mean([r["overlap_fraction"] for r in rows]))
        mean_foreign = float(np.mean([r["total_foreign"] for r in rows]))
        mean_boundary = float(np.mean([r["mean_boundary_on_membrane"] for r in rows]))
    else:
        mean_cov = mean_overlap = mean_foreign = mean_boundary = float("nan")

    frames_str = ",".join(str(z) for z in frames)
    row = (
        f"| {out_dir.as_posix()} | {approach} | {negatives} | {selection} | {resolver} | "
        f"{frames_str} | {mean_cov:.3f} | {mean_foreign:.2f} | {mean_boundary:.3f} | "
        f"{mean_overlap:.4f} | {notes} |\n"
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(row)


def _run_one(segment_fn, annotate_df, cfg: PerframeCfg, out_dir: Path, *, approach: str,
            frames, model_size: str, device, config_extra: dict) -> list[dict]:
    """Segment `frames` with one knob combination, writing config.json / scores.csv /
    montages under `out_dir`. `segment_fn(frame_sam, node_index, membrane_map) ->
    (cell_masks, label_map, score)` carries all the approach-specific logic (a closure
    over segment_frame_prompt or segment_frame_amg and its own knobs); `config_extra`
    holds those same knobs for config.json. Returns the per-frame score rows (the same
    rows written to scores.csv), so callers (a single run or a sweep step) can summarise
    them into the experiments log without redoing the segmentation.
    """
    montage_dir = out_dir / "montages"
    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for z in frames:
        frame_sam, _full_hw = pipeline.load_frame_sam(int(z), scale=cfg.scale)
        gray = frame_sam.mean(axis=2) if frame_sam.ndim == 3 else frame_sam
        mem = mb.membrane_map(gray)
        node_index = pf.nodes_in_frame(annotate_df, int(z), cfg.scale)
        if not node_index:
            print(f"[run_perframe] {out_dir.name} z={z}: no nodes in frame, skipping")
            continue

        _cell_masks, label_map, score = segment_fn(frame_sam, node_index, mem)

        print(f"[run_perframe] {out_dir.name} z={z} n_cells={score['n_cells']} "
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
        "approach": approach,
        "frames": list(frames),
        "scale": cfg.scale,
        "model_size": model_size,
        "radius": cfg.radius,
        "tau": cfg.tau,
        **config_extra,
        "device": str(device),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    print(f"[run_perframe] wrote {out_dir / 'config.json'}, {out_dir / 'scores.csv'}, "
         f"{len(rows)} montage(s) under {montage_dir}")
    return rows


def _build_sam2_model(model_size: str):
    """Build the raw SAM2 model for `model_size` (a checkpoint download/build, shared by a
    single --approach amg run and the --tune grid, which builds one AMG per trial on top of
    the same model instead of rebuilding it each time). Returns (sam2_model, device).
    """
    from sam2.build_sam import build_sam2

    device = setup.setup_device()
    ckpt, model_cfg = setup.ensure_checkpoint(model_size)
    sam2_model = build_sam2(model_cfg, str(ckpt), device=device)
    return sam2_model, device


def _build_amg_from_args(args: argparse.Namespace):
    """Build the SAM2AutomaticMaskGenerator for --approach amg, from --model-size and
    --amg-params (defaults: DEFAULT_AMG_PARAMS, overridden key-by-key by the --amg-params
    JSON if given). Returns (amg, device, amg_params), the last one so the caller can also
    record the resolved params in config.json.
    """
    sam2_model, device = _build_sam2_model(args.model_size)
    amg_params = dict(DEFAULT_AMG_PARAMS)
    if args.amg_params:
        amg_params.update(json.loads(args.amg_params))
    amg = build_amg(sam2_model, **amg_params)
    return amg, device, amg_params


def _make_segment_fn(args: argparse.Namespace, cfg: PerframeCfg):
    """Build the (segment_fn, device, config_extra) triple for --approach prompt|amg, so
    _run / _run_sweep stay approach-agnostic. segment_fn(frame_sam, node_index, mem) ->
    (cell_masks, label_map, score); config_extra is the approach-specific knobs recorded
    in config.json.
    """
    if args.approach == "prompt":
        image_predictor, device = setup.build_predictor(size=args.model_size, kind="image")

        def segment_fn(frame_sam, node_index, mem):
            return segment_frame_prompt(
                image_predictor, frame_sam, node_index, mem,
                negatives=args.negatives == "on", selection=args.selection,
                resolver=args.resolver, cfg=cfg)

        config_extra = {
            "negatives": args.negatives, "selection": args.selection,
            "resolver": args.resolver, "k_max_neg": cfg.k_max_neg,
            "box_margin": cfg.box_margin,
        }
        return segment_fn, device, config_extra

    if args.approach == "amg":
        amg, device, amg_params = _build_amg_from_args(args)

        def segment_fn(frame_sam, node_index, mem):
            return segment_frame_amg(
                amg, frame_sam, node_index, mem,
                match=args.match, resolver=args.resolver, cfg=cfg)

        config_extra = {
            "match": args.match, "resolver": args.resolver, "amg_params": amg_params,
        }
        return segment_fn, device, config_extra

    raise ValueError(f"unknown approach {args.approach!r}")


def _run(args: argparse.Namespace) -> None:
    from eval import merge_metric

    cfg = PerframeCfg(scale=args.scale, radius=args.radius, tau=args.tau,
                      k_max_neg=args.k_max_neg, box_margin=args.box_margin)
    out_dir = Path(args.out)

    segment_fn, device, config_extra = _make_segment_fn(args, cfg)
    annotate_df = merge_metric.load_node_table()

    rows = _run_one(segment_fn, annotate_df, cfg, out_dir, approach=args.approach,
                    frames=args.frames, model_size=args.model_size, device=device,
                    config_extra=config_extra)

    negatives = args.negatives if args.approach == "prompt" else "-"
    selection = args.selection if args.approach == "prompt" else args.match
    _append_experiment_log(out_dir, approach=args.approach, negatives=negatives,
                           selection=selection, resolver=args.resolver,
                           frames=args.frames, rows=rows)


# Approach 1's knob grid: negatives on/off x selection (3 ways) x resolver (2 ways) = 12 combos.
SWEEP_NEGATIVES = ("on", "off")
SWEEP_SELECTIONS = ("pred_iou", "generous", "metric")
SWEEP_RESOLVERS = ("argmax", "watershed")


def _combo_name(negatives: str, selection: str, resolver: str) -> str:
    return f"neg_{negatives}-sel_{selection}-res_{resolver}"


def _run_sweep(args: argparse.Namespace) -> None:
    """Loop the Approach-1 knob grid over `args.frames`, one `_run_one` call per combo,
    each writing to its own auto-named subdirectory of `args.out` and appending its own
    row to the experiments log. Thin by design: all the segmentation logic stays in
    segment_frame_prompt / _run_one, this just drives the grid. Approach-1 (prompt) only;
    Approach 2 (amg) has no knob grid here yet (that is Plan 2 Task 2's tuner).
    """
    if args.approach != "prompt":
        raise ValueError("--sweep only supports --approach prompt (the Approach-1 knob "
                         "grid); Approach 2 has its own tuner (--tune, Plan 2 Task 2)")

    from eval import merge_metric

    cfg = PerframeCfg(scale=args.scale, radius=args.radius, tau=args.tau,
                      k_max_neg=args.k_max_neg, box_margin=args.box_margin)
    base_out = Path(args.out)

    image_predictor, device = setup.build_predictor(size=args.model_size, kind="image")
    annotate_df = merge_metric.load_node_table()

    combos = list(itertools.product(SWEEP_NEGATIVES, SWEEP_SELECTIONS, SWEEP_RESOLVERS))
    for i, (negatives, selection, resolver) in enumerate(combos, start=1):
        name = _combo_name(negatives, selection, resolver)
        out_dir = base_out / name
        print(f"[run_perframe] sweep {i}/{len(combos)}: {name}")

        def segment_fn(frame_sam, node_index, mem, _neg=negatives, _sel=selection, _res=resolver):
            return segment_frame_prompt(
                image_predictor, frame_sam, node_index, mem,
                negatives=_neg == "on", selection=_sel, resolver=_res, cfg=cfg)

        config_extra = {
            "negatives": negatives, "selection": selection, "resolver": resolver,
            "k_max_neg": cfg.k_max_neg, "box_margin": cfg.box_margin,
        }
        rows = _run_one(segment_fn, annotate_df, cfg, out_dir, approach=args.approach,
                        frames=args.frames, model_size=args.model_size, device=device,
                        config_extra=config_extra)

        _append_experiment_log(out_dir, approach=args.approach, negatives=negatives,
                               selection=selection, resolver=resolver,
                               frames=args.frames, rows=rows)


# Approach 2's default tune grid: pred_iou_thresh x stability_score_thresh x
# points_per_side = 12 combos. Overridable via --tune-grid.
DEFAULT_TUNE_GRID = {
    "pred_iou_thresh": (0.7, 0.8, 0.88),
    "stability_score_thresh": (0.9, 0.95),
    "points_per_side": (32, 64),
}

_TUNE_GAMEABLE_NOTE = ("objective can be gamed (e.g. by shrinking masks to cut overlap "
                      "and foreign hits at the cost of coverage); montages are the real check")


def _tune_grid_combos(grid: dict) -> list[dict]:
    """Cartesian product of a {param: [values, ...]} grid into a list of one dict per grid
    point, e.g. {"a": [1, 2], "b": [3]} -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]. Pure and
    torch-free, so the grid-building logic is unit-testable without SAM2."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run_tune(args: argparse.Namespace) -> None:
    """Grid-search AMG params against eval.perframe_score.objective, Approach 2's tuner.

    Loads each of --frames once (frame_sam, node_index, membrane map), then for every grid
    point builds an AMG (build_amg, on top of one shared SAM2 model) and runs
    segment_frame_amg over the cached frames, scoring each with score_frame + objective.
    Every trial's params, mean objective, and per-frame scores are written to
    `<out>/trials.csv`. The best trial is then re-run through `_run_one` to also produce its
    montages/scores.csv/config.json under `--out`, and a summary row (with a NOTE that the
    objective can be gamed, since a degenerate small-mask trial can score well without
    looking right) is appended to the experiments log.

    Grid default: DEFAULT_TUNE_GRID; override with --tune-grid, a JSON object of
    {param: [values, ...]}.
    """
    if args.approach != "amg":
        raise ValueError("--tune only supports --approach amg (Approach 2's own tuner); "
                         "Approach 1 has its own knob grid (--sweep)")

    from eval import merge_metric

    cfg = PerframeCfg(scale=args.scale, radius=args.radius, tau=args.tau,
                      k_max_neg=args.k_max_neg, box_margin=args.box_margin)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = dict(DEFAULT_TUNE_GRID)
    if args.tune_grid:
        grid.update(json.loads(args.tune_grid))
    combos = _tune_grid_combos(grid)

    annotate_df = merge_metric.load_node_table()
    frame_cache = []
    for z in args.frames:
        frame_sam, _full_hw = pipeline.load_frame_sam(int(z), scale=cfg.scale)
        gray = frame_sam.mean(axis=2) if frame_sam.ndim == 3 else frame_sam
        mem = mb.membrane_map(gray)
        node_index = pf.nodes_in_frame(annotate_df, int(z), cfg.scale)
        if not node_index:
            print(f"[run_perframe] tune z={z}: no nodes in frame, skipping")
            continue
        frame_cache.append((int(z), frame_sam, node_index, mem))

    sam2_model, device = _build_sam2_model(args.model_size)

    trial_rows: list[dict] = []
    best: Optional[tuple[float, dict, dict]] = None
    for i, params in enumerate(combos, start=1):
        amg_params = dict(DEFAULT_AMG_PARAMS)
        if args.amg_params:
            amg_params.update(json.loads(args.amg_params))
        amg_params.update(params)
        amg = build_amg(sam2_model, **amg_params)

        per_frame: list[dict] = []
        for z, frame_sam, node_index, mem in frame_cache:
            _cell_masks, _label_map, score = segment_frame_amg(
                amg, frame_sam, node_index, mem, match=args.match,
                resolver=args.resolver, cfg=cfg)
            obj = objective(score)
            per_frame.append({"z": z, "objective": obj,
                              **{k: v for k, v in score.items() if k != "per_cell"}})

        mean_obj = float(np.mean([r["objective"] for r in per_frame])) if per_frame else float("nan")
        print(f"[run_perframe] tune {i}/{len(combos)} {params} mean_objective={mean_obj:.4f}")
        trial_rows.append({**params, "mean_objective": mean_obj,
                           "per_frame_scores": json.dumps(per_frame)})

        if per_frame and (best is None or mean_obj > best[0]):
            best = (mean_obj, params, amg_params)

    pd.DataFrame(trial_rows).to_csv(out_dir / "trials.csv", index=False)
    print(f"[run_perframe] wrote {out_dir / 'trials.csv'} ({len(trial_rows)} trial(s))")

    if best is None:
        print("[run_perframe] tune: no trial produced a score (empty --frames?), stopping "
             "before montages/log")
        return

    best_mean_obj, best_params, best_amg_params = best
    print(f"[run_perframe] tune: best {best_params} mean_objective={best_mean_obj:.4f}")

    best_amg = build_amg(sam2_model, **best_amg_params)

    def segment_fn(frame_sam, node_index, mem):
        return segment_frame_amg(best_amg, frame_sam, node_index, mem,
                                 match=args.match, resolver=args.resolver, cfg=cfg)

    config_extra = {
        "match": args.match, "amg_params": best_amg_params, "tune_grid": grid,
        "tune_best_params": best_params, "tune_best_mean_objective": best_mean_obj,
    }
    rows = _run_one(segment_fn, annotate_df, cfg, out_dir, approach="amg",
                    frames=args.frames, model_size=args.model_size, device=device,
                    config_extra=config_extra)

    notes = (f"tuned: best {best_params}, mean_objective={best_mean_obj:.4f}. "
            f"NOTE: {_TUNE_GAMEABLE_NOTE}.")
    _append_experiment_log(out_dir, approach="amg", negatives="-", selection=args.match,
                           resolver=args.resolver, frames=args.frames, rows=rows, notes=notes)


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-frame neuron segmentation. Approach 1 (prompt-based): image-mode "
                    "SAM2 once per node. Approach 2 (amg): SAM2AutomaticMaskGenerator + "
                    "match to nodes, competitors kept. Both share membrane-aware overlap "
                    "resolution and F2 scoring.")
    ap.add_argument("--approach", choices=["prompt", "amg"], default="prompt",
                    help="segmentation approach: 'prompt' (Approach 1, image-mode SAM2 per "
                         "node) or 'amg' (Approach 2, SAM2AutomaticMaskGenerator + match "
                         "to nodes, keeping the rest as competitors)")
    ap.add_argument("--frames", nargs="+", type=int, required=True,
                    help="CATMAID z's to segment")
    ap.add_argument("--negatives", choices=["on", "off"], default="on",
                    help="prompt only: pass other cells' nodes as negative points "
                         "(ignored with --sweep or --approach amg)")
    ap.add_argument("--selection", choices=["pred_iou", "generous", "metric"], default="metric",
                    help="prompt only: how to pick among SAM2's 3 multimask candidates "
                         "(ignored with --sweep or --approach amg)")
    ap.add_argument("--match", choices=["area", "metric"], default="metric",
                    help="amg only: how to match a node to one of the AMG masks; 'area' "
                         "picks the smallest containing mask, 'metric' uses the F2 "
                         "composite (pf.match_amg_to_nodes). 'area' has no cross-node "
                         "foreign exclusion, so a fused blob can match two different "
                         "cells and inflate own_coverage; prefer 'metric' when the match "
                         "itself needs to be trusted")
    ap.add_argument("--amg-params", default=None,
                    help="amg only: JSON object overriding DEFAULT_AMG_PARAMS key-by-key, "
                         "e.g. '{\"points_per_side\": 32}'")
    ap.add_argument("--resolver", choices=["argmax", "watershed"], default="argmax",
                    help="overlap-resolution method, F3 (ignored with --sweep)")
    ap.add_argument("--sweep", action="store_true",
                    help="loop the Approach-1 knob grid (negatives x selection x resolver, "
                         "12 combos) over --frames instead of running one combo; each combo "
                         "gets its own auto-named subdirectory of --out and its own row in "
                         "docs/explanation/perframe-experiments.md; prompt approach only")
    ap.add_argument("--tune", action="store_true",
                    help="grid-search AMG params (pred_iou_thresh x stability_score_thresh "
                         "x points_per_side, or --tune-grid override) over --frames, "
                         "maximising eval.perframe_score.objective; writes "
                         "<out>/trials.csv (every trial), the winning params' montages, and "
                         "a summary row (with a gameable-objective NOTE) to "
                         "docs/explanation/perframe-experiments.md; amg approach only, "
                         "mutually exclusive with --sweep")
    ap.add_argument("--tune-grid", default=None,
                    help="JSON object overriding DEFAULT_TUNE_GRID key-by-key, e.g. "
                         "'{\"points_per_side\": [32]}' (values are lists)")
    ap.add_argument("--scale", type=int, default=8, help="downscale factor (_sam grid)")
    ap.add_argument("--model-size", default="tiny", help="SAM2 checkpoint size")
    ap.add_argument("--out", required=True,
                    help="results dir, e.g. results/perframe/<run> (with --sweep, the parent "
                         "dir under which each combo gets its own auto-named subdirectory)")
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
    _args = _parse()
    if _args.sweep and _args.tune:
        raise ValueError("--sweep and --tune are mutually exclusive")
    if _args.tune:
        _run_tune(_args)
    elif _args.sweep:
        _run_sweep(_args)
    else:
        _run(_args)
