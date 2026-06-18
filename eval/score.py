"""
score.py — region-metric scoring harness.

Joins predicted masks to the cross-worm GT (:mod:`eval.groundtruth`) and rolls the
:mod:`eval.metrics` overlap numbers up per (neuron, slice) and per neuron. This is
the **wire-in point** for whatever produces predictions in the GT frame.

The unsolved dependency (deliberately external)
-----------------------------------------------
The current pipeline's predicted masks are in the *target* worm's frame (``_sam``
space, indexed by ``catmaid_z``); the GT is a *different* worm in its own VAST
pixel grid. There is no established xy/z registration between them yet (see
eval/README.md). So this module does NOT try to register or resample: it requires a
``PredictionSource`` that already yields a boolean mask **on the GT mask grid**, for
a given ``(neuron, slice_idx)``. Producing that — running the pipeline on the GT
worm's EM, or resampling existing output through a fitted transform — is the next
task; this harness scores it the moment it exists.

``DirPredictionSource`` is the concrete reference implementation: a directory tree
``<root>/<neuron>/<slice_idx:03d>.png`` of binary PNGs. Swap in any object with the
same three methods to score a different prediction store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .groundtruth import GroundTruth
from . import metrics as M


# =============================================================================
# Prediction source contract
# =============================================================================

class PredictionSource(Protocol):
    """What score_region needs from any prediction store.

    Masks MUST be returned on the GT mask grid (same H×W as
    ``GroundTruth.label_slice``); registration is the caller's job upstream.
    """

    def available_neurons(self) -> Sequence[str]: ...
    def slices_for(self, neuron: str) -> Sequence[int]: ...
    def mask(self, neuron: str, slice_idx: int) -> Optional[np.ndarray]: ...


@dataclass
class DirPredictionSource:
    """Predictions as ``<root>/<neuron>/<slice_idx:03d>.png`` binary masks.

    Any non-zero pixel is foreground. ``slice_fmt`` controls the filename; adjust
    if your store names slices differently.
    """

    root: Union[str, Path]
    slice_fmt: str = "{idx:03d}.png"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def available_neurons(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def slices_for(self, neuron: str) -> List[int]:
        d = self.root / neuron
        out: List[int] = []
        for p in d.glob("*.png"):
            stem = p.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if digits:
                out.append(int(digits))
        return sorted(out)

    def mask(self, neuron: str, slice_idx: int) -> Optional[np.ndarray]:
        p = self.root / neuron / self.slice_fmt.format(idx=int(slice_idx))
        if not p.exists():
            return None
        return np.asarray(Image.open(p)) > 0


# =============================================================================
# Scoring
# =============================================================================

# Per-frame columns the harness emits.
_FRAME_COLS = ["neuron", "slice", "iou", "dice", "precision", "recall",
               "tp", "fp", "fn", "pred_area", "gt_area"]


def score_region(
    gt: GroundTruth,
    source: PredictionSource,
    *,
    neurons: Optional[Sequence[str]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    progress: bool = False,
    progress_every: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score predictions against GT with region-overlap metrics.

    Parameters
    ----------
    gt : the GroundTruth read layer.
    source : a PredictionSource yielding GT-grid boolean masks.
    neurons : which neuron labels to score; default = every neuron the source
        offers that also exists in the GT metadata.
    out_dir : if given, write ``eval_frames.csv`` and ``eval_neurons.csv`` there.

    Returns
    -------
    (frames_df, neurons_df).
      frames_df  — one row per scored (neuron, slice).
      neurons_df — one row per neuron: micro (volume, count-summed) IoU/Dice/
                   precision/recall, plus the per-frame means and frame count.
    """
    import sys
    import time as _time

    want = list(neurons) if neurons is not None else list(source.available_neurons())

    frame_rows: List[dict] = []
    skipped: Dict[str, str] = {}
    timing: Dict[str, Tuple[int, float]] = {}    # neuron -> (n_scored_slices, seconds)
    t_all = _time.perf_counter()
    if progress:
        print(f"[eval] scoring {len(want)} neuron(s) against GT "
              f"(per-slice mask overlap; full-res GT decode is the cost)", flush=True)
    for ni, neuron in enumerate(want, 1):
        if not gt.nr_for_label(neuron):
            skipped[neuron] = "not in GT metadata"
            continue
        slices = list(source.slices_for(neuron))
        if progress:
            print(f"[eval] ({ni}/{len(want)}) {neuron}: {len(slices)} slices", flush=True)
        t_n = _time.perf_counter()
        scored = 0
        for j, s in enumerate(slices, 1):
            if not gt.has_slice(s):
                continue
            pred = source.mask(neuron, s)
            if pred is None:
                continue
            gtm = gt.neuron_mask(s, neuron)
            if pred.shape != gtm.shape:
                raise ValueError(
                    f"{neuron} slice {s}: pred {pred.shape} not on GT grid {gtm.shape}. "
                    "Predictions must be registered/resampled to the GT mask grid "
                    "before scoring (see module docstring).")
            m = M.binary_metrics(pred, gtm)
            frame_rows.append({"neuron": neuron, "slice": int(s), **{
                k: m[k] for k in ("iou", "dice", "precision", "recall",
                                  "tp", "fp", "fn", "pred_area", "gt_area")}})
            scored += 1
            if progress and j % max(1, progress_every) == 0:
                el = _time.perf_counter() - t_n
                print(f"      {neuron} {j}/{len(slices)} slices  "
                      f"{j / el:.2f} slice/s  elapsed {el:.0f}s", flush=True)
        secs = _time.perf_counter() - t_n
        timing[neuron] = (scored, secs)
        if progress:
            rate = scored / secs if secs else float("nan")
            print(f"      {neuron} done: {scored} slices in {secs:.1f}s ({rate:.2f} slice/s)",
                  flush=True)

    total_secs = _time.perf_counter() - t_all
    frames = pd.DataFrame(frame_rows, columns=_FRAME_COLS)
    neurons_df = _rollup_neurons(frames, timing=timing)
    if progress:
        print(f"[eval] scoring done: {len(frames)} (neuron,slice) pairs in "
              f"{total_secs:.1f}s", flush=True)

    if skipped:
        print(f"[eval] skipped {len(skipped)} label(s) not in GT: "
              f"{sorted(skipped)[:10]}{' ...' if len(skipped) > 10 else ''}")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames.to_csv(out_dir / "eval_frames.csv", index=False)
        neurons_df.to_csv(out_dir / "eval_neurons.csv", index=False)
        print(f"[eval] wrote {len(frames)} frame rows + {len(neurons_df)} neuron rows "
              f"-> {out_dir}")

    return frames, neurons_df


def _rollup_neurons(frames: pd.DataFrame,
                    timing: Optional[Dict[str, Tuple[int, float]]] = None) -> pd.DataFrame:
    """Per-neuron rollup: micro (count-summed volume) + macro (per-frame mean).

    `timing` (neuron -> (n_slices, seconds)) adds `seconds` / `slices_per_s` columns
    so the caller can log how long each neuron took to score."""
    cols = ["neuron", "n_frames", "iou_micro", "dice_micro",
            "precision_micro", "recall_micro", "iou_mean", "dice_mean",
            "tp", "fp", "fn", "seconds", "slices_per_s"]
    if frames.empty:
        return pd.DataFrame(columns=cols)

    rows: List[dict] = []
    for neuron, g in frames.groupby("neuron"):
        tp, fp, fn = int(g["tp"].sum()), int(g["fp"].sum()), int(g["fn"].sum())
        micro = M.metrics_from_counts(tp, fp, fn)
        secs = float(timing[neuron][1]) if timing and neuron in timing else float("nan")
        n = int(len(g))
        rows.append({
            "neuron": neuron,
            "n_frames": n,
            "iou_micro": micro["iou"], "dice_micro": micro["dice"],
            "precision_micro": micro["precision"], "recall_micro": micro["recall"],
            "iou_mean": float(g["iou"].mean()), "dice_mean": float(g["dice"].mean()),
            "tp": tp, "fp": fp, "fn": fn,
            "seconds": round(secs, 2) if secs == secs else secs,
            "slices_per_s": round(n / secs, 3) if (secs == secs and secs > 0) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("neuron").reset_index(drop=True)
