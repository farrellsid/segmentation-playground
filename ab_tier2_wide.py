"""
ab_tier2_wide.py — wider tier-2 A/B for the "run flagged chains as tier-2 by default?"
decision (item d). Baseline (_sam) vs tier-2 (chain_crop=True, with the item-b fallback
ON) across several chains in 3 diverse neurons, with an aggregate verdict:
  - how many chains tier-2 IMPROVED (queue down) / REGRESSED (queue up) / unchanged
  - how often the _sam fallback fired (fell_back_to_sam)
  - net queue delta

Per the §6 ruler: relative queue deltas at fixed thresholds. Throwaway harness.

    py -3 ab_tier2_wide.py

SEED_OVERRIDE lets this adopt the seed-ablation winner (ab_seed.py); defaults to the
current pipeline seed (fixed-margin box + positive point).
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from sam2_utils import setup, alignment, diagnostics, config
from pipeline import PipelineConfig, ChainState, run_chain, save_state

# 3 diverse neurons; a spread of chain indices each. Tune indices toward chains that
# flag under baseline once a baseline scan identifies them (more signal to move).
NEURON_CHAINS = {
    "AIYL": [12, 29, 2, 7, 18],     # interneuron (known A/B set + extras)
    "RMDR": [3, 10, 20, 30, 40],    # motor neuron
    "AVBR": [4, 12, 22, 32, 42],    # interneuron (large arbor)
}

# adopt the seed-ablation winner here once known; {} = current pipeline default.
SEED_OVERRIDE: dict = {}

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_tier2_wide"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"


def _cfg(chain_crop: bool, out_root: Path) -> PipelineConfig:
    base = dict(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=True,
        chain_crop=chain_crop,
        chain_crop_pad_tif=64, chain_crop_scale=2, chain_crop_max_px=1536,
        chain_crop_min_tif=1024,
        chain_crop_fallback=True,            # item b: safety net on
        output_root=out_root, frames_root=FRAMES_ROOT,
    )
    base.update(SEED_OVERRIDE)
    return PipelineConfig(**base)


def _summarize(state: ChainState) -> dict:
    s = state.qc_summary or {}
    return {
        "status": state.status,
        "fell_back": state.fell_back_to_sam,
        "n_frames": s.get("n_frames"),
        "n_queue": s.get("n_queue"),
        "n_flagged": s.get("n_flagged"),
        "n_noskel": s.get("n_missing_skel"),
        "flag_rate": s.get("flag_rate"),
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

    rows: list[tuple] = []   # (neuron, idx, base_summary, tier2_summary)
    for neuron, idxs in NEURON_CHAINS.items():
        cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
        for idx in idxs:
            if idx >= len(cell_chains):
                print(f"skip {neuron} c{idx} (only {len(cell_chains)} chains)")
                continue
            chain = cell_chains[idx]
            summ = {}
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
                    summ[mode] = _summarize(state)
                except Exception as e:
                    print(f"!! {mode} FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    summ[mode] = {"status": f"ERROR {type(e).__name__}"}
                finally:
                    image_predictor.reset_predictor()
                    diagnostics.cleanup_vram()
            rows.append((neuron, idx, summ.get("baseline", {}), summ.get("tier2", {})))

    # ---- per-chain table ----
    print("\n\n================ WIDER TIER-2 A/B (per chain) ================")
    print(f"{'chain':14s} | base_q  tier2_q  dq   base_flag tier2_flag  fellback  status(b/t2)")
    improved = regressed = unchanged = fellback = 0
    net_dq = 0
    for neuron, idx, b, t in rows:
        bq, tq = b.get("n_queue"), t.get("n_queue")
        if isinstance(bq, int) and isinstance(tq, int):
            dq = tq - bq
            net_dq += dq
            if dq < 0: improved += 1
            elif dq > 0: regressed += 1
            else: unchanged += 1
        else:
            dq = "?"
        if t.get("fell_back"): fellback += 1
        print(f"{neuron+' c'+str(idx):14s} | {str(bq):5s}  {str(tq):6s}  {str(dq):4s} "
              f"{str(b.get('flag_rate')):8s}  {str(t.get('flag_rate')):9s}  {str(t.get('fell_back')):8s}  "
              f"{b.get('status')}/{t.get('status')}")

    print("\n================ AGGREGATE ================")
    print(f"chains: {len(rows)}  |  tier-2 improved: {improved}  regressed: {regressed}  "
          f"unchanged: {unchanged}  |  fallback fired: {fellback}  |  net queue delta: {net_dq:+d}")
    print("DECISION INPUT: tier-2-by-default looks safe if regressed ~0 and net_dq <= 0 "
          "(queue reduced or held, no new regressions).")


if __name__ == "__main__":
    main()
