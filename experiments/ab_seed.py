"""
ab_seed.py — comprehensive video-SEED ablation. Which anchor-frame conditioning
gives the best propagation? "More prompts != better", so we sweep the seed space and
rank by the review queue it produces.

SAM2 treats MASK and POINTS/BOX as mutually-exclusive per frame (add_new_mask pops
point_inputs and vice-versa; sam2_video_predictor.py), so the valid seed space is:
mask-only, OR any combination of {box(fixed|frac), positive point, negative points}.

Design choice: anchor phase held CONSTANT at crop_anchor=False (legacy scale-8 _sam),
so (a) every seed is compared on the SAME _sam anchor mask + _sam propagation and (b)
the mask seed is valid (the anchor mask is in the propagation space). Caveat: the anchor
is scale-8 here, not the production tier-1 crop; this isolates the SEED variable. Re-run
under chain_crop=True later to confirm the winner holds at tier-2 resolution.

    py -3 ab_seed.py

Writes ab_seed/<config>/<neuron>/chain_XX/. Throwaway; safe to delete.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from sam2_utils import setup, alignment, diagnostics, config
from pipeline import PipelineConfig, ChainState, run_chain, save_state

# chains to ablate over. Mix of flagged + clean (from the tier-2 A/B set) so there is
# queue signal to move. Expand freely.
CHAINS = [("AIYL", 12), ("AIYL", 29), ("AIYL", 2)]

# the seed configs. label -> cfg overrides. box_margin_frac=0.3 = pad the box by 30% of
# the bbox's longest side (first-pass value for the underfill fix).
FRAC = 0.3
SEED_CONFIGS = [
    ("box_pos",         dict(seed_box="fixed", seed_points=True,  seed_negatives=False, seed_mask=False)),  # current default
    ("box_pos_neg",     dict(seed_box="fixed", seed_points=True,  seed_negatives=True,  seed_mask=False)),
    ("box_only",        dict(seed_box="fixed", seed_points=False, seed_negatives=False, seed_mask=False)),
    ("pos_only",        dict(seed_box="none",  seed_points=True,  seed_negatives=False, seed_mask=False)),
    ("pos_neg",         dict(seed_box="none",  seed_points=True,  seed_negatives=True,  seed_mask=False)),
    ("boxfrac_pos",     dict(seed_box="frac",  seed_points=True,  seed_negatives=False, seed_mask=False, box_margin_frac=FRAC)),
    ("boxfrac_only",    dict(seed_box="frac",  seed_points=False, seed_negatives=False, seed_mask=False, box_margin_frac=FRAC)),
    ("boxfrac_pos_neg", dict(seed_box="frac",  seed_points=True,  seed_negatives=True,  seed_mask=False, box_margin_frac=FRAC)),
    ("mask_only",       dict(seed_box="none",  seed_points=False, seed_negatives=False, seed_mask=True)),
]

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_seed"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"


def _cfg(out_root: Path, **overrides) -> PipelineConfig:
    base = dict(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=False,      # CONSTANT: scale-8 _sam anchor -> mask is in propagation space
        chain_crop=False,
        output_root=out_root, frames_root=FRAMES_ROOT,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _summarize(state: ChainState) -> dict:
    s = state.qc_summary or {}
    return {
        "status": state.status,
        "n_frames": s.get("n_frames"),
        "n_flagged": s.get("n_flagged"),
        "n_queue": s.get("n_queue"),
        "n_intervene": s.get("n_intervene"),
        "n_noskel": s.get("n_missing_skel"),
        "flag_rate": s.get("flag_rate"),
        "queue_rate": s.get("queue_rate"),
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

    results: list[tuple] = []   # (neuron, idx, label, summary)
    for neuron, idx in CHAINS:
        cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
        chain = cell_chains[idx]
        for label, overrides in SEED_CONFIGS:
            out_root = AB_ROOT / label
            chain_dir = out_root / neuron / f"chain_{idx:02d}"
            print(f"\n========== {neuron} chain {idx:02d} [{label}] ==========")
            try:
                cfg = _cfg(out_root, **overrides)
                state = ChainState(neuron=neuron, chain_idx=idx, config=cfg)
                state = run_chain(state, image_predictor=image_predictor,
                                  video_predictor=video_predictor,
                                  annotate_df=annotate_df, chain=chain,
                                  on_video_phase=diagnostics.cleanup_vram)
                save_state(state, chain_dir / "state.json")
                results.append((neuron, idx, label, _summarize(state)))
            except Exception as e:
                print(f"!! {label} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results.append((neuron, idx, label, {"status": f"ERROR {type(e).__name__}"}))
            finally:
                image_predictor.reset_predictor()
                diagnostics.cleanup_vram()

    # ---- per-chain table ----
    print("\n\n================ SEED ABLATION (per chain) ================")
    cols = ["status", "n_frames", "n_flagged", "n_queue", "n_intervene", "n_noskel",
            "flag_rate", "image_score", "t_total"]
    for neuron, idx, label, s in results:
        print(f"{neuron} c{idx:02d} {label:16s} | " +
              "  ".join(f"{c}={s.get(c)}" for c in cols if c in s))

    # ---- aggregate ranking (lower queue then flag is better) ----
    print("\n================ AGGREGATE (ranked by total queue, then flags) ================")
    agg: dict[str, dict] = {}
    for _, _, label, s in results:
        a = agg.setdefault(label, {"n_queue": 0, "n_flagged": 0, "n_noskel": 0,
                                   "n_frames": 0, "t_total": 0.0, "errors": 0})
        if str(s.get("status", "")).startswith("ERROR"):
            a["errors"] += 1
            continue
        for k in ("n_queue", "n_flagged", "n_noskel", "n_frames"):
            a[k] += (s.get(k) or 0)
        a["t_total"] += (s.get("t_total") or 0.0)
    ranked = sorted(agg.items(), key=lambda kv: (kv[1]["n_queue"], kv[1]["n_flagged"]))
    print(f"{'config':16s} | tot_queue  tot_flag  tot_noskel  /frames   t_total  errors")
    for label, a in ranked:
        print(f"{label:16s} | {a['n_queue']:8d}  {a['n_flagged']:8d}  {a['n_noskel']:9d}  "
              f"{a['n_frames']:6d}   {a['t_total']:6.1f}   {a['errors']}")
    if ranked:
        print(f"\nbest (fewest queued): {ranked[0][0]}   |   "
              f"current default 'box_pos': queue={agg.get('box_pos',{}).get('n_queue')}")


if __name__ == "__main__":
    main()
