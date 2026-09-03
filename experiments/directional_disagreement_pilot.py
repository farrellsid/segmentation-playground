"""Pilot the RoboEM-style forward/backward disagreement QC signal (roadmap.md §4.2,
"training-free, do first") on one real chain: run an INDEPENDENT forward-only tracing
seeded at the chain's own start frame, and an INDEPENDENT backward-only tracing seeded
at the chain's own end frame, then score per-z agreement with
eval.merge_metric.directional_disagreement.

Real cost note (verified 2026-09-03, corrects roadmap.md §4.2's "near-zero cost"
framing): this is a genuine SECOND full propagation sweep, not free. The existing
single-mid-anchor propagate()/run_bidirectional() gives each frame exactly one mask;
getting two independent tracings to compare needs two full directional sweeps, seeded
at the chain's own start and end frames using the same centreline-derived per-frame
seed points segment_per_slice already builds for per-slice mode. ~2x compute vs a
normal propagate() run, this script's own timing output reports the real number.

Uses the chain's EXISTING output tree only to read state.json (frames space, chain
metadata); it does NOT read or depend on that tree's masks, this is a fresh pair of
propagation runs from scratch, same "prepare frames fresh" pattern
propagate_from_corrected_seed.py uses.

    py -3 experiments/directional_disagreement_pilot.py \\
        --working "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIYL_AIYR" \\
        --neuron AIYL --chain 0 \\
        --out-csv "docs/figures/directional_disagreement/AIYL_chain00.csv"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from eval.merge_metric import directional_disagreement, summarize_directional_disagreement
from pipeline.config import PipelineConfig
from pipeline.crop import prepare_chain_crop_frames, prepare_video_frames
from pipeline.predict import build_prompts, centreline_by_z
from pipeline.propagate import _node_id_at, propagate_directional
from pipeline.state import Prompts, load_state
from sam2_utils import alignment, config, setup

SCALE = 8


def _seed_prompts_at(frame_z: int, centreline: dict[int, tuple[float, float]],
                     annotate_df: pd.DataFrame, *, cfg, cw) -> Prompts:
    """Point (+ same-z negatives) prompt at frame_z, built from the CATMAID centreline,
    the same construction pipeline.propagate.segment_per_slice uses per-frame. Remapped
    into the crop window's own space when cw is set (tier-2), same as segment_per_slice."""
    x_tif, y_tif = centreline[frame_z]
    pos_sam = alignment.tif_to_sam([x_tif, y_tif], cfg.scale)
    points_sam = [[float(pos_sam[0]), float(pos_sam[1])]]
    labels = [1]
    if cfg.k_max_neg > 0:
        node_id = _node_id_at(annotate_df, frame_z, x_tif, y_tif)
        if node_id is not None:
            neg = build_prompts(node_id, frame_z, annotate_df, scale=cfg.scale,
                                k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)
            neg_labels = np.asarray(neg.labels)
            for pt in np.asarray(neg.points_sam, dtype=float)[neg_labels == 0]:
                points_sam.append([float(pt[0]), float(pt[1])])
                labels.append(0)
    points_sam_arr = np.asarray(points_sam, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    if cw is not None:
        pts_crop = cw.sam_to_crop(points_sam_arr)
        keep = np.ones(len(labels_arr), dtype=bool)   # seed frames are inside the window by construction
        return Prompts(points_sam=pts_crop[keep], labels=labels_arr[keep])
    return Prompts(points_sam=points_sam_arr, labels=labels_arr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--working", required=True, help="tree to read state.json from")
    ap.add_argument("--neuron", required=True)
    ap.add_argument("--chain", type=int, required=True, help="chain_idx")
    ap.add_argument("--out-csv", default=None, help="optional per-z record CSV")
    ap.add_argument("--model-size", default="large")
    ap.add_argument("--low-iou-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    working = Path(args.working)
    neuron, chain_idx = args.neuron, args.chain
    src_chain_dir = working / neuron / f"chain_{chain_idx:02d}"
    src_state = load_state(src_chain_dir / "state.json")
    cw = alignment.CropWindow.from_dict(src_state.crop_window) if src_state.crop_window else None
    print(f"[dirdis] {neuron} chain_{chain_idx:02d}, "
         f"space={'_pcrop tier-2' if cw else '_sam legacy'}")

    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)
    cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
    chain = cell_chains[chain_idx]

    cfg = PipelineConfig()
    print("[dirdis] preparing fresh frames ...")
    if cw is not None:
        frames_dir, frame_to_z, _anchor_frame_idx, n_frames = prepare_chain_crop_frames(
            chain, annotate_df, cw, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=src_state.anchor_catmaid_z, neuron=neuron, chain_idx=chain_idx)
    else:
        frames_dir, frame_to_z, _anchor_frame_idx, n_frames = prepare_video_frames(
            chain, annotate_df, scale=SCALE, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=src_state.anchor_catmaid_z, neuron=neuron, chain_idx=chain_idx)
    print(f"[dirdis] {n_frames} frames prepared")

    centreline = centreline_by_z(chain, annotate_df)
    start_idx, end_idx = 0, n_frames - 1
    start_z, end_z = frame_to_z[start_idx], frame_to_z[end_idx]
    start_prompts = _seed_prompts_at(start_z, centreline, annotate_df, cfg=cfg, cw=cw)
    end_prompts = _seed_prompts_at(end_z, centreline, annotate_df, cfg=cfg, cw=cw)

    print(f"[dirdis] building {args.model_size} video predictor ...")
    video_predictor, _ = setup.build_predictor(size=args.model_size, kind="video")

    t0 = perf_counter()
    print(f"[dirdis] forward-only sweep seeded at frame {start_idx} (z={start_z}) ...")
    fwd_segs, _fc, _fi = propagate_directional(
        video_predictor, frames_dir, start_prompts, start_idx, obj_id=1, reverse=False)
    t1 = perf_counter()
    print(f"[dirdis] backward-only sweep seeded at frame {end_idx} (z={end_z}) ...")
    back_segs, _bc, _bi = propagate_directional(
        video_predictor, frames_dir, end_prompts, end_idx, obj_id=1, reverse=True)
    t2 = perf_counter()
    print(f"[dirdis] timing: forward={t1 - t0:.1f}s backward={t2 - t1:.1f}s total={t2 - t0:.1f}s")

    fwd_by_z = {frame_to_z[fi]: (seg[1], 0, 0) for fi, seg in fwd_segs.items() if 1 in seg}
    back_by_z = {frame_to_z[fi]: (seg[1], 0, 0) for fi, seg in back_segs.items() if 1 in seg}

    records = directional_disagreement(fwd_by_z, back_by_z)
    summary = summarize_directional_disagreement(records, low_iou_threshold=args.low_iou_threshold)
    print(f"[dirdis] n_z_compared={summary['n_z']} n_dropout_z={summary['n_dropout_z']} "
         f"mean_iou={summary['mean_disagreement_iou']} "
         f"mean_drift_px={summary['mean_centroid_drift_px']} "
         f"frac_low_agreement={summary['frac_low_agreement']}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"[dirdis] wrote {len(records)} per-z records to {out_path}")


if __name__ == "__main__":
    main()
