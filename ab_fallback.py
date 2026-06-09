"""
ab_fallback.py — A/B harness for the tier-2 SAFETY fallback (M4.5 item b):
image_score/anchor-gated fall-back from the per-chain crop (_pcrop) to the plain
_sam path when the crop anchor is poor.

Throwaway measurement script (not part of the library). Two questions, per the
PIPELINE_CONTEXT §6 ruler (relative deltas at fixed thresholds, not absolute truth):

  1. Does the fallback CATCH the over-zoom collapse?  We recreate the c02 failure
     mode by DROPPING the chain_crop_min_tif floor (so a low-motion chain gets a
     tiny over-zoomed window) and compare fallback OFF vs ON. Expect: OFF -> a
     collapsed/flagged _pcrop mask; ON -> fell_back_to_sam=True and a result that
     tracks the _sam baseline.
  2. Is it INERT on a good crop?  Run a chain whose tier-2 crop was already good
     (normal min_tif) with fallback ON. Expect: fell_back_to_sam=False and the same
     verdict as plain tier-2 -> no regression from adding the gate.

    py -3 ab_fallback.py

Writes ab_fallback/<mode>/<neuron>/chain_XX/. Scratch; safe to delete.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from sam2_utils import setup, alignment, diagnostics, config
from pipeline import PipelineConfig, ChainState, run_chain, save_state

# (neuron, chain_idx, purpose). c02 = the low-motion over-zoom case; c12 = a chain
# whose tier-2 crop was good (regression / inertness check).
ZOOM_CHAIN = ("AIYL", 2)
GOOD_CHAIN = ("AIYL", 12)

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_fallback"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"
FORCE_ZOOM_MIN_TIF = 1          # disables the floor -> recreate the over-zoom window

# (mode label, chain, cfg overrides). Each runs once.
# Round 2: the gate-only criterion missed the over-zoom (its anchor mask passes the
# geometry gate; the collapse is a PROPAGATION effect). The image_score floor is the
# pre-propagation tell: over-zoom=0.516 vs healthy 0.848-0.922. Re-verify with floor=0.7.
SCORE_FLOOR = 0.7
PLAN = [
    # Q1: over-zoom (floor removed) + image_score floor -> expect fell_back=True, ~baseline.
    ("baseline",        ZOOM_CHAIN, dict(chain_crop=False)),
    ("zoom_fb_score",   ZOOM_CHAIN, dict(chain_crop=True, chain_crop_fallback=True,
                                         chain_crop_min_tif=FORCE_ZOOM_MIN_TIF,
                                         chain_crop_min_image_score=SCORE_FLOOR)),
    # Q2: good tier-2 crop (score 0.879) at the same floor -> must NOT fire (0.879 > 0.7).
    ("good_tier2_score", GOOD_CHAIN, dict(chain_crop=True, chain_crop_fallback=True,
                                          chain_crop_min_image_score=SCORE_FLOOR)),
]


def _cfg(out_root: Path, **overrides) -> PipelineConfig:
    base = dict(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=True,
        chain_crop=False, chain_crop_pad_tif=64, chain_crop_scale=2,
        chain_crop_max_px=1536, chain_crop_min_tif=1024,
        output_root=out_root, frames_root=FRAMES_ROOT,
    )
    base.update(overrides)
    return PipelineConfig(**base)


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
        "fell_back": state.fell_back_to_sam,
        "n_frames": s.get("n_frames"),
        "n_flagged": s.get("n_flagged"),
        "n_queue": s.get("n_queue"),
        "n_noskel": s.get("n_missing_skel"),
        "flag_rate": s.get("flag_rate"),
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
    for mode, (neuron, idx), overrides in PLAN:
        cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
        chain = cell_chains[idx]
        out_root = AB_ROOT / mode
        chain_dir = out_root / neuron / f"chain_{idx:02d}"
        print(f"\n========== {neuron} chain {idx:02d} [{mode}] ==========")
        try:
            cfg = _cfg(out_root, **overrides)
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

    print("\n\n================ FALLBACK A/B SUMMARY ================")
    cols = ["status", "fell_back", "n_frames", "n_flagged", "n_queue", "n_noskel",
            "flag_rate", "mask_dims", "crop_scale", "image_score", "t_total"]
    for neuron, idx, mode, s in results:
        line = f"{neuron} c{idx:02d} {mode:14s} | " + "  ".join(
            f"{c}={s.get(c)}" for c in cols if c in s)
        print(line)

    print("\n---- verdict ----")
    by_mode = {m: s for (_, _, m, s) in results}
    # Q1: with the score floor, the over-zoom should fall back and recover ~baseline.
    if "zoom_fb_score" in by_mode and "baseline" in by_mode:
        fb, base = by_mode["zoom_fb_score"], by_mode["baseline"]
        print(f"Q1 over-zoom+floor: fell_back={fb.get('fell_back')} status={fb.get('status')} "
              f"dims={fb.get('mask_dims')} flag_rate={fb.get('flag_rate')}  "
              f"(baseline status={base.get('status')} flag_rate={base.get('flag_rate')} "
              f"dims={base.get('mask_dims')})")
        recovered = (fb.get("fell_back") and fb.get("mask_dims") == base.get("mask_dims")
                     and fb.get("status") == base.get("status"))
        print("   PASS Q1" if recovered
              else "   CHECK Q1 (expected fell_back=True and result == baseline)")
    # Q2: the good crop (score > floor) should NOT fall back -> keeps tier-2 (_pcrop).
    if "good_tier2_score" in by_mode:
        g = by_mode["good_tier2_score"]
        print(f"Q2 good crop+floor: fell_back={g.get('fell_back')} status={g.get('status')} "
              f"crop_scale={g.get('crop_scale')} dims={g.get('mask_dims')}")
        print("   PASS Q2" if g.get("fell_back") is False and g.get("crop_scale") is not None
              else "   CHECK Q2 (expected fell_back=False, crop_scale set)")


if __name__ == "__main__":
    main()
