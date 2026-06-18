"""
ab_crop_from_mask.py — A/B harness: tier-2 crop sized from the SKELETON bbox
(chain_crop_from_mask=False, current default) vs sized from the _sam MASK bbox
(chain_crop_from_mask=True), on the same chains.

Why: the skeleton-bbox window is sized to the centerline NODES, so a cell whose
membrane bulges past the nodes + pad gets CLIPPED at the window edge (measured:
AIAL/chain_00 clips 24/113 frames). chain_crop_from_mask grows the window to the
union of the skeleton bbox and the bbox of the chain's already-saved _sam masks.

This reproduces the real auto-second-pass flow (batch.tier2_on_flagged): for the
mask-sized arm we FIRST run the plain _sam pass into that arm's chain dir (writing
the _sam masks + qc.csv the bbox is read from), THEN re-run tier-2 in place with
chain_crop_from_mask=True — exactly what `_run_one_chain` does.

Headline metric is the CLIP: how many frames have mask foreground touching the crop
border. We want that to drop to ~0 without regressing the QC queue. The ruler is
relative deltas at fixed thresholds, not absolute truth.
(Once the ERL eval exists, score this on confirmed GT instead.)

    py -3 experiments/ab_crop_from_mask.py
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_utils import setup, alignment, diagnostics, config
import pipeline
from pipeline import PipelineConfig, ChainState, run_chain, save_state

# chains to A/B (neuron, chain_idx) — the measured clippers first.
CHAINS = [("AIAL", 0), ("AIAL", 5)]

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_crop_from_mask"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"


def _cfg(*, chain_crop: bool, from_mask: bool, out_root: Path) -> PipelineConfig:
    return PipelineConfig(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=True,                 # tier-1 anchor crop is the baseline default
        chain_crop=chain_crop,
        chain_crop_from_mask=from_mask,
        chain_crop_pad_tif=64, chain_crop_scale=2, chain_crop_max_px=1536,
        output_root=out_root, frames_root=FRAMES_ROOT,
    )


def _mask_dims(chain_dir: Path) -> str:
    masks = sorted((chain_dir / "masks").glob("mask_*.png"))
    if not masks:
        return "no masks"
    import cv2
    m = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    return f"{m.shape[1]}x{m.shape[0]}"


def _border_clip(chain_dir: Path) -> tuple[int, int, int]:
    """(#frames whose mask touches the crop border, #frames total, max edge px)."""
    import cv2
    masks = sorted((chain_dir / "masks").glob("mask_*.png"))
    touch = 0
    max_edge = 0
    for p in masks:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        if not m.any():
            continue
        edge = (int(m[0, :].sum()) + int(m[-1, :].sum())
                + int(m[:, 0].sum()) + int(m[:, -1].sum()))
        if edge > 0:
            touch += 1
            max_edge = max(max_edge, edge)
    return touch, len(masks), max_edge


def _summarize(state: ChainState, chain_dir: Path) -> dict:
    s = state.qc_summary or {}
    cw = state.crop_window
    touch, ntot, max_edge = _border_clip(chain_dir)
    return {
        "status": state.status,
        "n_frames": s.get("n_frames"),
        "n_queue": s.get("n_queue"),
        "n_noskel": s.get("n_missing_skel"),
        "queue_rate": s.get("queue_rate"),
        "clip_frames": f"{touch}/{ntot}",
        "clip_max_edge": max_edge,
        "mask_dims": _mask_dims(chain_dir),
        "crop_scale": (cw or {}).get("crop_scale"),
        "fell_back": getattr(state, "fell_back_to_sam", None),
        "image_score": None if state.image_score is None else round(float(state.image_score), 3),
        "t_total": round(sum((state.phase_seconds or {}).values()), 1),
    }


def _run(neuron, idx, chain, cfg, image_predictor, video_predictor, annotate_df):
    chain_dir = Path(cfg.output_root) / neuron / f"chain_{idx:02d}"
    state = ChainState(neuron=neuron, chain_idx=idx, config=cfg)
    state = run_chain(state, image_predictor=image_predictor,
                      video_predictor=video_predictor,
                      annotate_df=annotate_df, chain=chain,
                      on_video_phase=diagnostics.cleanup_vram)
    save_state(state, chain_dir / "state.json")
    return state, chain_dir


def main() -> None:
    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)

    print("building predictors (large; image + video)...")
    image_predictor, _ = setup.build_predictor(size=MODEL, kind="image")
    video_predictor, _ = setup.build_predictor(size=MODEL, kind="video")
    diagnostics.snapshot("after model load")

    results: list[tuple] = []
    for neuron, idx in CHAINS:
        cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
        chain = cell_chains[idx]

        # arm A — tier-2 sized from the skeleton bbox (current default)
        print(f"\n========== {neuron} c{idx:02d} [tier2_skel] ==========")
        try:
            cfg = _cfg(chain_crop=True, from_mask=False, out_root=AB_ROOT / "tier2_skel")
            state, cdir = _run(neuron, idx, chain, cfg, image_predictor,
                               video_predictor, annotate_df)
            results.append((neuron, idx, "tier2_skel", _summarize(state, cdir)))
        except Exception as e:
            print(f"!! tier2_skel FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((neuron, idx, "tier2_skel", {"status": f"ERROR {type(e).__name__}"}))
        finally:
            image_predictor.reset_predictor()
            diagnostics.cleanup_vram()

        # arm B — reproduce the real second pass: _sam pass first (writes the masks +
        # qc.csv the bbox is read from), THEN tier-2 in place with from_mask=True.
        print(f"\n========== {neuron} c{idx:02d} [tier2_mask: _sam pre-pass] ==========")
        try:
            mask_root = AB_ROOT / "tier2_mask"
            cfg_sam = _cfg(chain_crop=False, from_mask=False, out_root=mask_root)
            _run(neuron, idx, chain, cfg_sam, image_predictor, video_predictor, annotate_df)
            image_predictor.reset_predictor()
            diagnostics.cleanup_vram()

            print(f"========== {neuron} c{idx:02d} [tier2_mask: tier-2 from mask] ==========")
            cfg_mask = _cfg(chain_crop=True, from_mask=True, out_root=mask_root)
            state, cdir = _run(neuron, idx, chain, cfg_mask, image_predictor,
                               video_predictor, annotate_df)
            results.append((neuron, idx, "tier2_mask", _summarize(state, cdir)))
        except Exception as e:
            print(f"!! tier2_mask FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((neuron, idx, "tier2_mask", {"status": f"ERROR {type(e).__name__}"}))
        finally:
            image_predictor.reset_predictor()
            diagnostics.cleanup_vram()

    print("\n\n================ A/B SUMMARY ================")
    print("clip_frames = #frames whose mask touches the crop border (lower is better)")
    cols = ["status", "n_frames", "n_queue", "n_noskel", "queue_rate",
            "clip_frames", "clip_max_edge", "mask_dims", "crop_scale", "fell_back",
            "image_score", "t_total"]
    for neuron, idx, mode, s in results:
        line = f"{neuron} c{idx:02d} {mode:11s} | " + "  ".join(
            f"{c}={s.get(c)}" for c in cols if c in s)
        print(line)


if __name__ == "__main__":
    main()
