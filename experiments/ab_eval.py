"""ab_eval.py, GT-scored A/B campaign over the pipeline levers (cross-worm SEM-Dauer 1).

Now that the eval ruler exists, measure the levers that were built on intuition (tier-2 crop,
the fallback floor, postprocess, multimask selection, negatives, seed shape, ...) against the GT
instead of by flag-rate. One factor at a time from a pinned baseline.

Design (see the plan): build the SAM2 predictors ONCE and loop config variants in-process via
batch.run_batch (so the exact production run path, incl. the tier-2 second pass, is exercised).
Score each variant cheaply at QUARTER scale (region IoU/precision/recall, GT decode cached and
reused across variants); confirm the winners at FULL res with ERL/VOI (no extra GPU, re-scores the
same masks). Results are written AFTER EVERY variant (per-variant screen.json) and the aggregate
CSV / Markdown / chart are regenerated from those, so a kill at any point loses nothing and a
re-run resumes (finished variants are skipped).

Run (free the GPU first):
    py -3 experiments/ab_eval.py                 # the screen (all variants, quarter-scale)
    py -3 experiments/ab_eval.py --confirm        # full-res + ERL on baseline + top winners
    py -3 experiments/ab_eval.py --variants baseline mm_excl_neg   # a subset
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from dataclasses import replace
from time import perf_counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

import batch
from pipeline import PipelineConfig
from sam2_utils import config, diagnostics
from eval.groundtruth import GroundTruth
from eval.score import score_region
from eval.score_batch import BatchPredictionSource, _gt_grid_hw

# --- where things go ---------------------------------------------------------
CAMPAIGN = pathlib.Path(__file__).resolve().parent / "ab_eval"   # reports here
MASK_ROOT = config.GT_PRED_DIR / "ab"                            # per-variant masks here (big)
QUARTER_MASK_DIR = config.GT_ROOT / "Segmentations" / "one_fourth_scale"
SCREEN_CACHE_CAP = 700        # quarter-res masks are small; cache the whole scored set across variants

# A small, fast sweep set (mix of multi-chain + single-chain), omitting the heavy AVAL.
SWEEP_NEURONS = ["GLRDR", "IL1L", "AS3"]

# Pinned baseline = the as-built defaults with the candidate FEATURES OFF, so each variant that
# turns one ON is a clean single-factor delta. tier2_on_flagged stays True (the production reality;
# several levers are tier-2 sub-features), and the "no_tier2" variant measures the whole feature.
BASELINE = PipelineConfig(
    model_size="large", scale=8, save_downscale=8,
    k_max_neg=3, neg_radius=150, box_margin=10,
    crop_anchor=True, chain_crop=False, chain_crop_from_mask=False,
    chain_crop_min_image_score=0.70,
    multimask_anchor=False, multimask_exclude_neg=False,
    postprocess_masks=False,
    seed_box="fixed", seed_points=True, seed_negatives=False, seed_mask=False,
    # Shared across all variants: the frame cache (frames_cache_s{scale}/z*.jpg) is keyed by
    # z+scale, config-independent, so the expensive full-res decode happens once (first variant)
    # and is reused; per-chain views are rebuilt fresh each run, so no stale crop frames leak
    # between variants. output_root is set per-variant in run_and_screen.
    frames_root=config.GT_PRED_DIR / "frames",
)

# (label, human description, config overrides, tier2_on_flagged). Ordered by value: baseline first,
# then Tier-1 (the burning questions), then Tier-2 sweeps, then the small-model arm (needs a rebuild).
VARIANTS: list[tuple[str, str, dict, bool]] = [
    ("baseline",      "as-built, features off",                 {},                                                  True),
    # Tier-1
    ("mm_anchor",     "multimask anchor select",                {"multimask_anchor": True},                          True),
    ("mm_excl_neg",   "multimask + exclude negatives",          {"multimask_anchor": True, "multimask_exclude_neg": True}, True),
    ("postproc",      "postprocess masks (open/close/cc/fill)", {"postprocess_masks": True},                         True),
    ("cropmin060",    "tier-2 fallback floor 0.70->0.60",       {"chain_crop_min_image_score": 0.60},                True),
    ("ccfm",          "chain crop sized from mask",             {"chain_crop_from_mask": True},                      True),
    ("no_tier2",      "tier-2 OFF (pure _sam)",                 {},                                                  False),
    # Tier-2
    ("kneg0",         "no negative prompts (k=0)",              {"k_max_neg": 0},                                    True),
    ("kneg7",         "more negatives (k=7)",                   {"k_max_neg": 7},                                    True),
    ("seed_neg",      "seed negatives into video",              {"seed_negatives": True},                            True),
    ("seed_frac",     "fractional box pad (underfill)",         {"seed_box": "frac", "box_margin_frac": 0.1},        True),
    ("seed_mask",     "mask seed (needs _sam anchor)",          {"seed_mask": True, "seed_box": "none", "seed_points": False, "crop_anchor": False}, True),
    ("crop_anchor_off", "legacy scale-8 anchor (tier-1 off)",   {"crop_anchor": False},                              True),
    ("model_small",   "small model",                            {"model_size": "small"},                             True),
]

_SESSIONS: dict[str, object] = {}     # model_size -> built Session (predictors reused across variants)


def _session_for(model_size: str):
    if model_size not in _SESSIONS:
        print(f"[ab] building session (model_size={model_size}) ...", flush=True)
        _SESSIONS[model_size] = batch._build_gt_session(
            replace(BASELINE, model_size=model_size), neurons=SWEEP_NEURONS)
    return _SESSIONS[model_size]


def _quarter_gt() -> GroundTruth:
    """Quarter-scale GT for the cheap screen, with a big cache so decode amortizes across variants."""
    gt = GroundTruth.load(config.GT_METADATA, QUARTER_MASK_DIR, downscale=4)
    gt._cache_cap = SCREEN_CACHE_CAP
    return gt


# --- one variant: run + screen-score, persisting its own result ----------------------------------

def run_and_screen(label: str, overrides: dict, tier2_on_flagged: bool, gt_q: GroundTruth) -> dict:
    cfg = replace(BASELINE, output_root=(MASK_ROOT / label), **overrides)
    masks_dir = MASK_ROOT / label
    masks_dir.mkdir(parents=True, exist_ok=True)

    session = _session_for(cfg.model_size)
    t0 = perf_counter()
    batch.run_batch(session, cfg, masks_dir, neurons=SWEEP_NEURONS,
                    gif_mode="off", tier2_on_flagged=tier2_on_flagged)
    run_secs = perf_counter() - t0
    diagnostics.cleanup_vram()

    # Score at quarter scale. BatchPredictionSource's save_downscale is the GT-grid/mask-grid ratio:
    # masks are saved at full/save_downscale; the quarter GT grid is full/4, so the ratio is
    # save_downscale//4 (= 2 for the default 8). Getting this wrong places the mask in a frame the
    # wrong size. The full-res confirm uses save_downscale//1 = 8.
    gt_hw = _gt_grid_hw(gt_q)
    ratio = max(1, cfg.save_downscale // gt_q.downscale)
    src = BatchPredictionSource(masks_dir, gt_hw, save_downscale=ratio)
    _frames, neurons_df = score_region(gt_q, src, neurons=SWEEP_NEURONS, progress=False)

    rec = {
        "label": label,
        "overrides": overrides,
        "tier2_on_flagged": tier2_on_flagged,
        "run_secs": round(run_secs, 1),
        "per_neuron": neurons_df.to_dict(orient="records"),
        "overall": _micro_overall(neurons_df),
    }
    (masks_dir / "screen.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec


def _micro_overall(neurons_df: pd.DataFrame) -> dict:
    """Pixel-summed (micro) IoU/precision/recall/dice over the variant's scored neurons."""
    tp = float(neurons_df["tp"].sum()); fp = float(neurons_df["fp"].sum()); fn = float(neurons_df["fn"].sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"iou_micro": iou, "precision_micro": prec, "recall_micro": rec, "dice_micro": dice,
            "tp": tp, "fp": fp, "fn": fn}


# --- aggregate reporting (regenerated from per-variant screen.json after every run) ---------------

def _load_screens() -> list[dict]:
    out = []
    for label, *_ in VARIANTS:
        sj = MASK_ROOT / label / "screen.json"
        if sj.exists():
            try:
                out.append(json.loads(sj.read_text()))
            except Exception:
                pass
    return out


def regenerate_reports() -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    screens = _load_screens()
    if not screens:
        return
    by_label = {s["label"]: s for s in screens}
    base = by_label.get("baseline", {}).get("overall")

    rows = []
    desc = {lbl: d for lbl, d, *_ in VARIANTS}
    order = {lbl: i for i, (lbl, *_) in enumerate(VARIANTS)}
    for s in sorted(screens, key=lambda s: order.get(s["label"], 99)):
        o = s["overall"]
        row = {"variant": s["label"], "lever": desc.get(s["label"], ""),
               "iou_micro": o["iou_micro"], "precision_micro": o["precision_micro"],
               "recall_micro": o["recall_micro"], "dice_micro": o["dice_micro"],
               "run_secs": s.get("run_secs")}
        if base:
            row["d_iou"] = o["iou_micro"] - base["iou_micro"]
            row["d_precision"] = o["precision_micro"] - base["precision_micro"]
            row["d_recall"] = o["recall_micro"] - base["recall_micro"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(CAMPAIGN / "results_screen.csv", index=False)

    # Markdown table (sorted by IoU delta, baseline pinned at top).
    md = ["# A/B screen (quarter-scale region metrics, SEM-Dauer 1 GT)", "",
          f"Neurons: {', '.join(SWEEP_NEURONS)}. Baseline = as-built, features off.", "",
          "| variant | lever | IoU | dIoU | precision | dprec | recall | drecall |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append("| {variant} | {lever} | {iou:.3f} | {diou} | {prec:.3f} | {dprec} | {rec:.3f} | {drec} |".format(
            variant=r["variant"], lever=r["lever"], iou=r["iou_micro"],
            diou=("" if "d_iou" not in r or pd.isna(r["d_iou"]) else f"{r['d_iou']:+.3f}"),
            prec=r["precision_micro"],
            dprec=("" if "d_precision" not in r or pd.isna(r["d_precision"]) else f"{r['d_precision']:+.3f}"),
            rec=r["recall_micro"],
            drec=("" if "d_recall" not in r or pd.isna(r["d_recall"]) else f"{r['d_recall']:+.3f}")))
    (CAMPAIGN / "results_screen.md").write_text("\n".join(md) + "\n")

    _chart(df)


def _chart(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[ab] chart skipped ({e})")
        return
    metrics = ["iou_micro", "precision_micro", "recall_micro"]
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(df)), 5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, df[m].values, w, label=m.replace("_micro", ""))
    base = df[df["variant"] == "baseline"]
    if not base.empty:
        ax.axhline(float(base["iou_micro"].iloc[0]), ls="--", lw=0.8, color="grey", label="baseline IoU")
    ax.set_xticks(x); ax.set_xticklabels(df["variant"], rotation=45, ha="right")
    ax.set_ylabel("micro metric"); ax.set_title("A/B screen (quarter-scale, SEM-Dauer 1 GT)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(CAMPAIGN / "results_screen.png", dpi=130)
    plt.close(fig)


# --- full-res confirm (baseline + winners): adds ERL/VOI, re-scores existing masks, no GPU --------

def confirm(winner_labels: list[str]) -> None:
    from eval.score_labelmap import score_labelmap
    from eval.gt_dataset import gt_paths
    gp = gt_paths()
    gt_full = GroundTruth.from_config()       # full_scale, GT_DOWNSCALE=1
    rows = []
    for label in winner_labels:
        masks_dir = MASK_ROOT / label
        if not (masks_dir / "screen.json").exists():
            print(f"[ab] confirm: no screen for {label}, skipping"); continue
        print(f"[ab] confirm (full-res + ERL): {label}", flush=True)
        try:
            rep = score_labelmap(masks_dir, gt_full, SWEEP_NEURONS,
                                 skeleton_csv=gp["skeleton_csv"], registration_json=gp["registration_json"],
                                 save_downscale=BASELINE.save_downscale, progress=False)
            voi = rep.get("voi", {}) or {}
            row = {"variant": label, "erl_um": rep.get("erl_um"), "pct_of_ceiling": rep.get("pct_of_ceiling"),
                   "voi_split": voi.get("voi_split"), "voi_merge": voi.get("voi_merge"),
                   "arand_are": (rep.get("arand") or {}).get("are")}
            (masks_dir / "confirm.json").write_text(json.dumps(rep, indent=2, default=float))
            rows.append(row)
        except Exception as e:
            print(f"[ab] confirm FAILED {label}: {e}"); traceback.print_exc()
        # persist after each so a kill keeps partial confirm results
        if rows:
            pd.DataFrame(rows).to_csv(CAMPAIGN / "results_confirm.csv", index=False)
    print(f"[ab] confirm done -> {CAMPAIGN / 'results_confirm.csv'}")


def _pick_winners(k: int = 4) -> list[str]:
    """Baseline + the top-k variants by screen IoU (the confirm set)."""
    screens = _load_screens()
    ranked = sorted((s for s in screens if s["label"] != "baseline"),
                    key=lambda s: s["overall"]["iou_micro"], reverse=True)
    return ["baseline"] + [s["label"] for s in ranked[:k]]


def main() -> None:
    ap = argparse.ArgumentParser(description="GT-scored A/B campaign over the pipeline levers")
    ap.add_argument("--confirm", action="store_true",
                    help="full-res + ERL on baseline + top winners (after the screen)")
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant labels to run")
    ap.add_argument("--confirm-variants", nargs="*", default=None,
                    help="explicit confirm set (default: baseline + top-4 by screen IoU)")
    args = ap.parse_args()

    if args.confirm:
        confirm(args.confirm_variants or _pick_winners())
        return

    todo = [(l, d, o, t) for (l, d, o, t) in VARIANTS
            if args.variants is None or l in args.variants]
    gt_q = _quarter_gt()
    for label, _desc, overrides, tier2 in todo:
        if (MASK_ROOT / label / "screen.json").exists():
            print(f"[ab] skip {label} (screen.json exists)"); continue
        print(f"\n[ab] ===== variant: {label} =====", flush=True)
        try:
            run_and_screen(label, overrides, tier2, gt_q)
            regenerate_reports()                       # refresh CSV/md/png after EVERY variant
            print(f"[ab] {label} done -> {CAMPAIGN / 'results_screen.csv'}")
        except Exception as e:
            print(f"[ab] VARIANT FAILED {label}: {e}"); traceback.print_exc()
            diagnostics.cleanup_vram()
    print(f"\n[ab] screen complete. Reports in {CAMPAIGN}. Next: --confirm")


if __name__ == "__main__":
    main()
