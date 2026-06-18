"""score_batch.py, score a SEM-Dauer 1 batch.py run against GT.

`batch.py --preset eval` writes per-CHAIN masks in `_sam` space:
    <output_root>/<neuron>/chain_NN/masks/mask_<slice:04d>.png   (0/255, ~1216x1152)
The GT (eval.score) needs per-NEURON masks on the full-res GT grid (9728x9216). This
module bridges the two:

  * `BatchPredictionSource` indexes every chain's masks, **unions a neuron's chains per
    slice**, and **upscales `_sam` -> GT grid** (nearest, x save_downscale) on demand, so it
    plugs straight into `eval.score.score_region` as a PredictionSource.

Region metrics (IoU/Dice/precision/recall) are the immediate read; per-neuron ERL +
split/merge (the gate) is layered on via labelmap compositing -> `eval.run_erl`.

    py -3 -m eval.score_batch --root data/groundtruth/pred_p280/batch_masks --out eval/out_gt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from sam2_utils import config
from .groundtruth import GroundTruth
from .score import score_region

Image.MAX_IMAGE_PIXELS = None


class BatchPredictionSource:
    """PredictionSource over a batch.py per-chain mask tree, GT-grid-aligned.

    Each chain's mask is first put on the full `_sam` frame (a `_sam` chain's PNG IS
    the frame; a tier-2 `_pcrop` chain is placed via its `crop_window`, see
    `score_labelmap.chain_sam_mask`), the neuron's chains are unioned, then the `_sam`
    union is upscaled to the GT grid so `score_region`'s shape check passes. Handling
    `_pcrop` here is essential, a raw resize of a crop would stretch it across the
    whole frame. `save_downscale` sets the `_sam` grid (gt_hw // save_downscale).
    """

    def __init__(self, root: Path, gt_hw: "tuple[int, int]", save_downscale: int = 8):
        self.root = Path(root)
        self.gt_hw = gt_hw
        self.sam_hw = (gt_hw[0] // save_downscale, gt_hw[1] // save_downscale)
        # neuron -> slice -> [(mask_path, crop_window)]
        self._idx: Dict[str, Dict[int, List["tuple[Path, Optional[dict]]"]]] = \
            defaultdict(lambda: defaultdict(list))
        for nd in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for ch in sorted(nd.glob("chain_*")):
                cw = None
                st = ch / "state.json"
                if st.exists():
                    try:
                        cw = json.loads(st.read_text()).get("crop_window")
                    except Exception:
                        cw = None
                for m in (ch / "masks").glob("mask_*.png"):
                    digits = "".join(c for c in m.stem if c.isdigit())
                    if digits:
                        self._idx[nd.name][int(digits)].append((m, cw))

    def available_neurons(self) -> List[str]:
        return sorted(self._idx)

    def slices_for(self, neuron: str) -> List[int]:
        return sorted(self._idx.get(neuron, {}))

    def mask(self, neuron: str, slice_idx: int) -> Optional[np.ndarray]:
        import cv2
        from .score_labelmap import chain_sam_mask
        items = self._idx.get(neuron, {}).get(int(slice_idx), [])
        if not items:
            return None
        union = np.zeros(self.sam_hw, dtype=bool)        # full _sam frame
        for p, cw in items:
            union |= chain_sam_mask(p, cw, self.sam_hw)
        Hg, Wg = self.gt_hw
        return cv2.resize(union.astype(np.uint8), (Wg, Hg),
                          interpolation=cv2.INTER_NEAREST) > 0


def _gt_grid_hw(gt: GroundTruth) -> "tuple[int, int]":
    """GT (H, W) from one slice decode (cached by the source thereafter)."""
    s = next(iter(gt.slice_indices))
    return tuple(gt.label_slice(s).shape)  # type: ignore[return-value]


def _pred_provenance(root: Path) -> dict:
    """Best-effort: read one chain's state.json to record how the prediction was made
    (model size, scale, crop), so the measurement log says *what* it scored."""
    out: dict = {}
    for st in root.glob("*/chain_*/state.json"):
        try:
            d = json.loads(st.read_text())
        except Exception:
            continue
        cfg = d.get("config", {}) or {}
        out = {"model_size": cfg.get("model_size"), "scale": cfg.get("scale"),
               "save_downscale": cfg.get("save_downscale"),
               "chain_crop": cfg.get("chain_crop"),
               "example_chain": str(st.parent.relative_to(root))}
        break
    return out


def main() -> None:
    from sam2_utils import presets
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(presets.PRESETS), default=None,
                    help="use a presets.py config: defaults --root to its batch output_root "
                         "and --out to its score_out (any flag below overrides)")
    ap.add_argument("--root", type=Path, default=None,
                    help="batch run output root (default: from --preset, else pred_p280/batch_masks)")
    ap.add_argument("--out", type=Path, default=None,
                    help="metrics output dir (default: from --preset, else eval/out_gt)")
    ap.add_argument("--neurons", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress per-slice progress")
    ap.add_argument("--no-labelmap", action="store_true",
                    help="skip the labelmap metrics (VOI/ARAND/ERL); region only")
    ap.add_argument("--save-downscale", type=int, default=None,
                    help="_sam downscale the masks were saved at (default: from --preset, else 8)")
    ap.add_argument("--merge-tol-frac", type=float, default=0.1,
                    help="ERL merge tolerance (a neuron counts toward a label's merge only if "
                         "it holds >= this fraction of the label's nodes)")
    args = ap.parse_args()

    # resolve root / out / save-downscale from the preset, with CLI overrides
    if args.preset:
        p = presets.get_preset(args.preset)
        if p.get("score_out") is None:
            ap.error(f"preset {args.preset!r} has no GT scoring (score_out=None), "
                     "scoring is for cross-worm GT presets like 'eval'.")
        if args.root is None:
            args.root = Path(p["output_root"])
        if args.out is None:
            args.out = Path(p["score_out"])
        if args.save_downscale is None:
            args.save_downscale = p["pipeline"].get("save_downscale", 8)
    if args.root is None:
        args.root = Path("data/groundtruth/pred_p280/batch_masks")
    if args.out is None:
        args.out = Path("eval/out_gt")
    if args.save_downscale is None:
        args.save_downscale = 8

    args.out.mkdir(parents=True, exist_ok=True)
    gt = GroundTruth.from_config()
    gt_hw = _gt_grid_hw(gt)
    src = BatchPredictionSource(args.root, gt_hw, save_downscale=args.save_downscale)
    neurons = args.neurons or src.available_neurons()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    frames, per_neuron = score_region(gt, src, neurons=neurons, out_dir=args.out,
                                       progress=not args.quiet)
    region_secs = time.perf_counter() - t0

    # --- labelmap metrics (VOI / ARAND / per-neuron ERL) ---
    labelmap_report = None
    if not args.no_labelmap:
        from .score_labelmap import score_labelmap
        from .gt_dataset import gt_paths
        gp = gt_paths()
        try:
            labelmap_report = score_labelmap(
                args.root, gt, neurons,
                skeleton_csv=gp["skeleton_csv"], registration_json=gp["registration_json"],
                save_downscale=args.save_downscale, merge_tol_frac=args.merge_tol_frac,
                progress=not args.quiet)
        except Exception as e:                       # never let labelmap scoring sink region results
            import traceback
            print(f"[score_batch] labelmap metrics FAILED (region results still valid): {e}")
            traceback.print_exc()
    total_secs = time.perf_counter() - t0

    # --- per-neuron timing CSV (the eval analogue of batch's _timing.csv) ---
    if not per_neuron.empty:
        per_neuron[["neuron", "n_frames", "seconds", "slices_per_s",
                    "iou_micro", "dice_micro", "precision_micro", "recall_micro"]] \
            .to_csv(args.out / "eval_timing.csv", index=False)

    # --- persist labelmap metrics to CSV (not just measurement_log.jsonl) ---
    # (a) merge per-neuron ERL into eval_neurons.csv (score_region wrote it region-only);
    # (b) an eval_labelmap.csv summary row with overall VOI / ARAND / ERL.
    if labelmap_report is not None:
        pn_erl = labelmap_report.get("per_neuron_erl_um", {}) or {}
        if not per_neuron.empty:
            per_neuron["erl_um"] = per_neuron["neuron"].map(pn_erl)
            per_neuron.to_csv(args.out / "eval_neurons.csv", index=False)   # rewrite w/ ERL
        voi = labelmap_report.get("voi") or {}
        ar = labelmap_report.get("arand") or {}
        pd.DataFrame([{
            "voi": voi.get("voi"), "voi_split": voi.get("voi_split"),
            "voi_merge": voi.get("voi_merge"),
            "arand_are": ar.get("are"), "arand_precision": ar.get("precision"),
            "arand_recall": ar.get("recall"),
            "erl_um": labelmap_report.get("erl_um"),
            "ceiling_um": labelmap_report.get("ceiling_um"),
            "pct_of_ceiling": labelmap_report.get("pct_of_ceiling"),
            "n_merge_labels": labelmap_report.get("n_merge_labels"),
            "n_nodes": labelmap_report.get("n_nodes"),
            "metric_backend": labelmap_report.get("metric_backend"),
            "node_sample_radius": labelmap_report.get("node_sample_radius"),
            "merge_tol_frac": labelmap_report.get("merge_tol_frac"),
        }]).to_csv(args.out / "eval_labelmap.csv", index=False)

    # --- measurement provenance log: what / against what / when / metrics / time ---
    tp = int(per_neuron["tp"].sum()) if not per_neuron.empty else 0
    fp = int(per_neuron["fp"].sum()) if not per_neuron.empty else 0
    fn = int(per_neuron["fn"].sum()) if not per_neuron.empty else 0
    micro_iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    record = {
        "when_utc": started.isoformat(timespec="seconds"),
        "measured": "SEM2 pipeline prediction masks (per-chain, _sam, upscaled to GT grid)",
        "against": {
            "gt": "SEM-Dauer 1 manual VAST segmentation (cross-worm; every segment lab-confirmed)",
            "gt_mask_dir": str(config.GT_MASK_DIR), "gt_downscale": gt.downscale,
            "gt_grid_hw": list(gt_hw)},
        "prediction_source": str(args.root),
        "prediction_provenance": _pred_provenance(args.root),
        "metrics_computed": (
            ["region: iou, dice, precision, recall (binary per-neuron, micro=pixel-summed "
             "+ macro=per-frame mean)"]
            + ([] if labelmap_report is None else [
                f"labelmap: VOI_split/VOI_merge + ARAND via {labelmap_report.get('metric_backend')} "
                "(CAD/FGNet methodology; pred vs GT over GT-foreground == ignore_labels=(0,); _sam grid)",
                "ERL + split/merge (per-neuron, skeleton-node sampling via registration)"])),
        "labelmap_metrics": labelmap_report,
        "neurons": list(neurons),
        "n_neurons_scored": int(len(per_neuron)),
        "n_frame_pairs": int(len(frames)),
        "results_overall": {"micro_iou": micro_iou, "tp": tp, "fp": fp, "fn": fn},
        "results_per_neuron": (per_neuron[["neuron", "n_frames", "iou_micro", "dice_micro",
                                           "precision_micro", "recall_micro"]]
                               .to_dict("records") if not per_neuron.empty else []),
        "timing": {"total_seconds": round(total_secs, 1),
                   "per_neuron_seconds": (dict(zip(per_neuron["neuron"], per_neuron["seconds"]))
                                          if not per_neuron.empty else {})},
        "notes": "region metrics are the right measure for a sparse neuron subset; VOI/ARAND "
                 "suit full-coverage runs; ERL works per-neuron once pred labelmaps are composited.",
    }
    with open(args.out / "measurement_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    if per_neuron.empty:
        print("[score_batch] no scored frames (no GT overlap / no masks).")
        return
    cols = ["neuron", "n_frames", "iou_micro", "dice_micro",
            "precision_micro", "recall_micro", "iou_mean", "seconds"]
    print("\n=== per-neuron region metrics (vs SEM-Dauer 1 GT) ===")
    print(per_neuron[cols].to_string(index=False))
    print(f"\n[score_batch] overall micro-IoU: {micro_iou:.3f}  "
          f"(tp={tp} fp={fp} fn={fn})")
    if labelmap_report is not None:
        lr = labelmap_report
        v, a = lr.get("voi"), lr.get("arand")
        print("=== labelmap metrics (VOI / ARAND over GT-foreground; per-neuron ERL) ===")
        if v: print(f"  VOI {v['voi']:.3f}  (split {v['voi_split']:.3f} + merge {v['voi_merge']:.3f})")
        if a: print(f"  ARAND (ARE) {a['are']:.3f}  (merge_err {a['merge_error']:.3f}, "
                    f"split_err {a['split_error']:.3f})")
        print(f"  ERL {lr['erl_um']:.2f} µm / ceiling {lr['ceiling_um']:.2f} µm "
              f"({lr['pct_of_ceiling']:.0f}%)  merges={lr['n_merge_labels']}  "
              f"overlap_px={lr['overlap_collisions_px']}")
    print(f"[score_batch] total scoring time: {total_secs:.1f}s")
    print(f"[score_batch] artifacts -> {args.out}/  "
          f"(eval_frames.csv, eval_neurons.csv [+erl_um], eval_labelmap.csv, "
          f"eval_timing.csv, measurement_log.jsonl)")


if __name__ == "__main__":
    main()
