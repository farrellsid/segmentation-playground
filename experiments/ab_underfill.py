"""
ab_underfill.py, find an under-filling anchor and test whether the %-of-bbox box
margin (box_margin_frac) fixes it.

"Underfill" = the anchor mask covers only PART of the cell, so box_from_mask draws a
box too tight to enclose the whole process and propagation can't recover the missing
extent. There's no ground-truth label, so the proxy is: anchor present (contained=True,
non-empty) BUT a high per-frame noskel rate (skeleton node falls outside the mask) ->
the mask is under-covering the cell as it moves.

Two stages, one GPU session (one model load):
  A. SCAN POOL under the current default seed (box_pos), record anchor area_frac +
     contained + noskel_rate + queue.
  B. Take the top-K underfill suspects and A/B three seeds: box_pos (fixed margin,
     already have), boxfrac_pos (margin_frac), mask_only. If frac/mask cut noskel/queue,
     the underfill hypothesis + the fix are confirmed; if not, the noskel is tracking
     drift, not underfill (honest negative).

    py -3 ab_underfill.py

Anchor held at scale-8 _sam (crop_anchor=False) so the box/frac/mask seeds are
comparable and the mask seed is valid. Throwaway harness.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from sam2_utils import setup, alignment, diagnostics, config
from pipeline import PipelineConfig, ChainState, run_chain, save_state

# scan pool: a spread across neuron types likely to differ in cross-section. Muscle
# (BWM) cells are large; interneurons thin. (neuron, [chain idxs])
POOL = {
    "AIYL":      [12, 29, 5, 15, 25],
    "RMDR":      [3, 8, 15, 25, 35],
    "BWM-DL02":  [2, 8, 16, 24, 32],
    "AVBR":      [4, 12, 20, 30],
    "RIML":      [5, 15, 25, 35],
}
TOP_K = 3                 # how many suspects to A/B in stage B
FRAC = 0.5                # bigger first-pass frac for the underfill test

AB_ROOT = Path(config.OUTPUT_ROOT).parent / "ab_underfill"
FRAMES_ROOT = config.FRAMES_ROOT
MODEL = "large"


def _cfg(out_root: Path, **overrides) -> PipelineConfig:
    base = dict(
        model_size=MODEL, scale=8, save_downscale=8,
        k_max_neg=3, box_margin=10,
        crop_anchor=False, chain_crop=False,
        seed_box="fixed", seed_points=True, seed_negatives=False, seed_mask=False,
        output_root=out_root, frames_root=FRAMES_ROOT,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _run(neuron, idx, chain, overrides, out_root, ip, vp, annotate_df):
    chain_dir = out_root / neuron / f"chain_{idx:02d}"
    cfg = _cfg(out_root, **overrides)
    state = ChainState(neuron=neuron, chain_idx=idx, config=cfg)
    state = run_chain(state, image_predictor=ip, video_predictor=vp,
                      annotate_df=annotate_df, chain=chain,
                      on_video_phase=diagnostics.cleanup_vram)
    save_state(state, chain_dir / "state.json")
    s = state.qc_summary or {}
    asc = state.anchor_score or {}
    n = s.get("n_frames") or 0
    return {
        "status": state.status,
        "anchor_area_frac": round(float(asc.get("area_frac", 0.0)), 5),
        "anchor_contained": asc.get("contained"),
        "n_frames": n,
        "n_noskel": s.get("n_missing_skel"),
        "noskel_rate": round((s.get("n_missing_skel") or 0) / n, 3) if n else None,
        "n_queue": s.get("n_queue"),
        "flag_rate": s.get("flag_rate"),
        "image_score": None if state.image_score is None else round(float(state.image_score), 3),
    }


def main() -> None:
    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)

    def get_chain(neuron, idx):
        cc = [c for c in all_chains if c["cell_name"] == neuron]
        return cc[idx] if idx < len(cc) else None

    print("building predictors (large; image + video)...")
    ip, _ = setup.build_predictor(size=MODEL, kind="image")
    vp, _ = setup.build_predictor(size=MODEL, kind="video")
    diagnostics.snapshot("after model load")

    # ---- stage A: scan ----
    scan: list[tuple] = []
    for neuron, idxs in POOL.items():
        for idx in idxs:
            chain = get_chain(neuron, idx)
            if chain is None:
                print(f"skip {neuron} c{idx} (out of range)"); continue
            print(f"\n===== SCAN {neuron} c{idx:02d} (box_pos) =====")
            try:
                r = _run(neuron, idx, chain, dict(), AB_ROOT / "scan", ip, vp, annotate_df)
                scan.append((neuron, idx, r))
            except Exception as e:
                print(f"!! scan {neuron} c{idx} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
            finally:
                ip.reset_predictor(); diagnostics.cleanup_vram()

    print("\n\n================ SCAN (box_pos) ================")
    for neuron, idx, r in scan:
        print(f"{neuron} c{idx:02d} | area_frac={r['anchor_area_frac']} contained={r['anchor_contained']} "
              f"noskel={r['n_noskel']}/{r['n_frames']} ({r['noskel_rate']}) queue={r['n_queue']} "
              f"img={r['image_score']} {r['status']}")

    # underfill suspects: anchor present (contained, non-empty) + highest noskel_rate.
    suspects = [(neu, idx, r) for (neu, idx, r) in scan
                if r["anchor_contained"] and (r["noskel_rate"] or 0) > 0]
    suspects.sort(key=lambda t: t[2]["noskel_rate"], reverse=True)
    suspects = suspects[:TOP_K]
    print(f"\nunderfill suspects (top {TOP_K} by noskel_rate, anchor contained): "
          + ", ".join(f"{n} c{i}({r['noskel_rate']})" for n, i, r in suspects))

    # ---- stage B: A/B the fix on suspects ----
    variants = [
        ("box_pos_fixed", dict()),                                              # = scan baseline
        ("boxfrac_pos",   dict(seed_box="frac", box_margin_frac=FRAC)),
        ("mask_only",     dict(seed_box="none", seed_points=False, seed_mask=True)),
    ]
    abres: dict = {}
    for neuron, idx, base_r in suspects:
        chain = get_chain(neuron, idx)
        abres[(neuron, idx)] = {"box_pos_fixed": base_r}     # reuse scan result
        for label, ov in variants[1:]:
            print(f"\n===== FIX {neuron} c{idx:02d} [{label}] =====")
            try:
                abres[(neuron, idx)][label] = _run(neuron, idx, chain, ov,
                                                   AB_ROOT / label, ip, vp, annotate_df)
            except Exception as e:
                print(f"!! {label} {neuron} c{idx} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                abres[(neuron, idx)][label] = {"status": f"ERROR {type(e).__name__}"}
            finally:
                ip.reset_predictor(); diagnostics.cleanup_vram()

    print("\n\n================ UNDERFILL FIX A/B ================")
    print(f"(margin_frac={FRAC}; lower noskel/queue = better)")
    for (neuron, idx), d in abres.items():
        print(f"\n{neuron} c{idx:02d}:")
        for label, _ in variants:
            r = d.get(label, {})
            print(f"  {label:14s} | noskel={r.get('n_noskel')}/{r.get('n_frames')} "
                  f"({r.get('noskel_rate')}) queue={r.get('n_queue')} "
                  f"area_frac={r.get('anchor_area_frac')} {r.get('status')}")
        fixed = d.get("box_pos_fixed", {}); frac = d.get("boxfrac_pos", {})
        if isinstance(fixed.get("n_noskel"), int) and isinstance(frac.get("n_noskel"), int):
            dn = frac["n_noskel"] - fixed["n_noskel"]
            print(f"  -> frac vs fixed noskel delta: {dn:+d} "
                  f"({'frac HELPS underfill' if dn < 0 else 'no underfill benefit' if dn >= 0 else ''})")


if __name__ == "__main__":
    main()
