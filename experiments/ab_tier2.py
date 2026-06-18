"""
ab_tier2.py — A/B harness: tier-2 per-chain crop (chain_crop=True) vs the _sam
full-frame baseline (chain_crop=False), on the same chains.

Throwaway measurement script (not part of the library). Runs each listed chain
through pipeline.run_chain twice (once per config) into separate output roots,
then prints a side-by-side of the QC verdict, flag/queue/noskel rates, mask
resolution, and timing. The ruler is relative deltas at fixed
thresholds (does tier-2 move the noskel queue / sharpen masks), not absolute truth.

    py -3 ab_tier2.py
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

# chains to A/B (neuron, chain_idx). Start with the bug-report chain; expand after.
CHAINS = [("AIYL", 12), ("AIYL", 29), ("AIYL", 2)]

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_tier2"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"


def _cfg(chain_crop: bool, out_root: Path) -> PipelineConfig:
    return PipelineConfig(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=True,                 # tier-1 anchor crop is the baseline default
        chain_crop=chain_crop,
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


def _summarize(state: ChainState, chain_dir: Path) -> dict:
    s = state.qc_summary or {}
    cw = state.crop_window
    return {
        "status": state.status,
        "n_frames": s.get("n_frames"),
        "n_flagged": s.get("n_flagged"),
        "n_queue": s.get("n_queue"),
        "n_intervene": s.get("n_intervene"),
        "n_noskel": s.get("n_missing_skel"),
        "flag_rate": s.get("flag_rate"),
        "queue_rate": s.get("queue_rate"),
        "mask_dims": _mask_dims(chain_dir),
        "crop_scale": (cw or {}).get("crop_scale"),
        "image_score": None if state.image_score is None else round(float(state.image_score), 3),
        "t_total": round(sum((state.phase_seconds or {}).values()), 1),
    }


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
        for mode, chain_crop in (("baseline", False), ("tier2", True)):
            out_root = AB_ROOT / mode
            chain_dir = out_root / neuron / f"chain_{idx:02d}"
            print(f"\n========== {neuron} chain {idx:02d} [{mode}] ==========")
            try:
                cfg = _cfg(chain_crop, out_root)
                state = ChainState(neuron=neuron, chain_idx=idx, config=cfg)
                state = run_chain(state, image_predictor=image_predictor,
                                  video_predictor=video_predictor,
                                  annotate_df=annotate_df, chain=chain,
                                  on_video_phase=diagnostics.cleanup_vram)
                save_state(state, chain_dir / "state.json")
                results.append((neuron, idx, mode, _summarize(state, chain_dir)))
            except Exception as e:
                print(f"!! {mode} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results.append((neuron, idx, mode, {"status": f"ERROR {type(e).__name__}"}))
            finally:
                image_predictor.reset_predictor()
                diagnostics.cleanup_vram()

    print("\n\n================ A/B SUMMARY ================")
    cols = ["status", "n_frames", "n_flagged", "n_queue", "n_intervene", "n_noskel",
            "flag_rate", "queue_rate", "mask_dims", "crop_scale", "image_score", "t_total"]
    for neuron, idx, mode, s in results:
        line = f"{neuron} c{idx:02d} {mode:9s} | " + "  ".join(
            f"{c}={s.get(c)}" for c in cols if c in s)
        print(line)


if __name__ == "__main__":
    main()
