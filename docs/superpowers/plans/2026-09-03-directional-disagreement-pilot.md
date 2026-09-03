# Directional (forward-vs-backward) Disagreement Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a real pilot of roadmap.md §4.2's top-ranked, not-yet-built QC idea —
score per-frame disagreement between an INDEPENDENT forward-only tracing (seeded at the chain's
own start frame) and an INDEPENDENT backward-only tracing (seeded at the chain's own end frame),
the RoboEM-style training-free error detector — and report real numbers from at least one actual
chain, not just land the plumbing.

**Architecture:** Three additive pieces, no existing behavior touched: (1) a pure scoring function
in `eval/merge_metric.py` mirroring the existing `z_transitions`/`summarize_z_consistency` pattern
exactly (canvas-paste IoU on a shared `_sam`-grid coordinate frame), but comparing two masks at the
SAME z instead of adjacent z's; (2) a thin single-direction propagation wrapper in
`pipeline/propagate.py` reusing the existing `PropagationSession` primitive (seed once, sweep one
direction only); (3) a CLI pilot script in `experiments/`, mirroring
`experiments/propagate_from_corrected_seed.py`'s real-chain-loading pattern, that builds start/end
seed prompts from the chain's own CATMAID centreline (the same construction
`pipeline/propagate.py::segment_per_slice` already uses per-frame), runs both directional sweeps,
scores disagreement, and prints/saves a report.

**Tech Stack:** Python 3.13, PyTorch 2.12 (CUDA available on this machine, verified), SAM2 video
predictor, pandas/numpy, pytest.

## Global Constraints

- `pipeline/` must never import `eval` (enforced by `tests/test_import_direction.py`) — the new
  scoring function lives in `eval/merge_metric.py`; `pipeline/propagate.py`'s new function must
  stay eval-free, exactly like `select_second_pass_frames`'s existing docstring warns.
- No existing default behavior changes: `batch.py`, presets, and `propagate()` are untouched. This
  is a new, opt-in diagnostic path only (script + library functions), same pattern as
  `experiments/propagate_from_corrected_seed.py` and `experiments/microsam_spotcheck.py`.
- TDD: write the failing test before the implementation for every code task.
- Verified fact this plan corrects: roadmap.md §4.2 currently claims scoring "+z vs -z" disagreement
  is "near-zero cost" because "we already run both passes." Checked against the installed
  `sam2/sam2_video_predictor.py::propagate_in_video` source directly (2026-09-03): `reverse=False`
  covers `[start_frame_idx, num_frames-1]` and `reverse=True` covers `[0, start_frame_idx]` —
  strictly DISJOINT except at the shared anchor frame. The current single-mid-anchor
  `PropagationSession.run_bidirectional()` therefore gives each frame exactly ONE propagated mask,
  never two. Getting genuine per-frame double coverage requires two independent full sweeps seeded
  at the chain's OWN start and end frames — a real ~2x compute cost, not free. Task 5 fixes the doc.

---

### Task 1: `directional_disagreement` + `summarize_directional_disagreement` in `eval/merge_metric.py`

**Files:**
- Modify: `eval/merge_metric.py` (add two functions after `summarize_z_consistency`, i.e. after
  line 223)
- Test: `tests/test_directional_disagreement.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks (pure function, numpy only).
- Produces: `directional_disagreement(fwd: dict[int, tuple[np.ndarray, int, int]], back: dict[int, tuple[np.ndarray, int, int]]) -> list[dict]`
  and `summarize_directional_disagreement(records: list[dict], *, low_iou_threshold: float = 0.5) -> dict`.
  Both dicts are keyed by CATMAID `z` (int) -> `(mask: bool HxW ndarray, x0: int, y0: int)`, the
  SAME shape `z_transitions` already takes (from `pipeline.chain_masks_in_sam`). Task 3 constructs
  these dicts from the two directional propagation runs (frame_idx -> z via the chain's own
  `frame_to_z`, then re-keyed to z since `fwd`/`back` may cover different frame_idx ranges but must
  be compared by the z they share).

- [x] **Step 1: Write the failing tests**

Create `tests/test_directional_disagreement.py`:

```python
import numpy as np
from eval import merge_metric as mm


def _rect(h=10, w=10, y0=3, y1=7, x0=3, x1=7):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks_perfect_agreement():
    m = _rect()
    fwd = {5: (m, 0, 0)}
    back = {5: (m.copy(), 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["z"] == 5
    assert r["iou"] == 1.0
    assert r["centroid_drift_px"] == 0.0


def test_shifted_mask_hand_computed_iou_and_drift():
    m_a = _rect()                      # x[3:7), y[3:7)
    m_b = _rect(x0=5, x1=9)             # x[5:9), y[3:7), shifted +2 in x
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_b, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    # intersection x[5:7),y[3:7) = 2*4=8; each area 16; union = 16+16-8=24
    assert abs(r["iou"] - 8 / 24) < 1e-9
    assert abs(r["centroid_drift_px"] - 2.0) < 1e-9


def test_offset_and_shape_difference_same_region_gives_perfect_agreement():
    mask_a = np.ones((4, 4), dtype=bool)               # 4x4 window at (10, 10)
    mask_b = np.zeros((6, 6), dtype=bool)               # 6x6 window at (8, 8)
    mask_b[2:6, 2:6] = True                             # same absolute region
    fwd = {5: (mask_a, 10, 10)}
    back = {5: (mask_b, 8, 8)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] == 1.0
    assert r["centroid_drift_px"] == 0.0


def test_disjoint_masks_zero_iou():
    m_a = np.zeros((10, 10), dtype=bool); m_a[0:2, 0:2] = True
    m_b = np.zeros((10, 10), dtype=bool); m_b[8:10, 8:10] = True
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_b, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] == 0.0


def test_one_side_empty_is_dropout_not_low_agreement():
    m_a = _rect()
    m_empty = np.zeros((10, 10), dtype=bool)
    fwd = {5: (m_a, 0, 0)}
    back = {5: (m_empty, 0, 0)}
    r = mm.directional_disagreement(fwd, back)[0]
    assert r["iou"] is None
    assert r["centroid_drift_px"] is None


def test_only_shared_z_is_scored():
    m = _rect()
    fwd = {5: (m, 0, 0), 6: (m.copy(), 0, 0)}      # 6 is fwd-only (near the back seed end)
    back = {5: (m.copy(), 0, 0), 4: (m.copy(), 0, 0)}  # 4 is back-only (near the fwd seed end)
    r = mm.directional_disagreement(fwd, back)
    assert [rec["z"] for rec in r] == [5]


def test_summarize_empty_input():
    s = mm.summarize_directional_disagreement([])
    assert s["n_z"] == 0
    assert s["mean_disagreement_iou"] is None
    assert s["mean_centroid_drift_px"] is None
    assert s["frac_low_agreement"] is None


def test_summarize_aggregates_and_excludes_dropout():
    records = [
        {"z": 1, "iou": 0.9, "centroid_drift_px": 1.0},
        {"z": 2, "iou": 0.3, "centroid_drift_px": 5.0},
        {"z": 3, "iou": None, "centroid_drift_px": None},   # one-sided dropout
    ]
    s = mm.summarize_directional_disagreement(records)
    assert s["n_z"] == 3
    assert s["n_dropout_z"] == 1
    assert abs(s["mean_disagreement_iou"] - (0.9 + 0.3) / 2) < 1e-9
    # frac_low_agreement (threshold 0.5) over scored z's only: 1 of 2 below 0.5
    assert abs(s["frac_low_agreement"] - 0.5) < 1e-9
```

- [x] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_directional_disagreement.py -v`
Expected: FAIL with `AttributeError: module 'eval.merge_metric' has no attribute 'directional_disagreement'`

- [x] **Step 3: Implement `directional_disagreement` and `summarize_directional_disagreement`**

In `eval/merge_metric.py`, insert immediately after `summarize_z_consistency` (after line 223,
before the blank line preceding `def summarize(per: pd.DataFrame) -> dict:`):

```python
def directional_disagreement(
    fwd: dict[int, tuple[np.ndarray, int, int]],
    back: dict[int, tuple[np.ndarray, int, int]],
) -> list[dict]:
    """Per-z agreement between two INDEPENDENT directional tracings of the same chain:
    a forward-only sweep seeded at the chain's own start frame, and a backward-only
    sweep seeded at the chain's own end frame (see pipeline.propagate.propagate_directional).
    Unlike z_transitions (adjacent z within ONE tracing), this compares two DIFFERENT
    tracings at the SAME z, the RoboEM-style training-free error signal (roadmap.md §4.2):
    keep only where forward and backward tracings agree.

    fwd / back: {z: (mask, x0, y0)} in the shared _sam-grid coordinate frame, same shape
    z_transitions takes (from pipeline.chain_masks_in_sam). Only z's present in BOTH dicts
    are scored: a z near the fwd seed's own start or the back seed's own start is, by
    construction, only reachable from one direction and has nothing to compare against.

    A z where either mask is empty gets iou=None, centroid_drift_px=None: dropout is a
    separate existing signal (score_chain's own empty/dropout_rate), not an agreement
    reading, same convention z_transitions uses."""
    shared = sorted(set(fwd) & set(back))
    out: list[dict] = []
    for z in shared:
        mask_a, x0_a, y0_a = fwd[z]
        mask_b, x0_b, y0_b = back[z]
        rec = {"z": int(z), "iou": None, "centroid_drift_px": None}
        if not mask_a.any() or not mask_b.any():
            out.append(rec)
            continue
        h_a, w_a = mask_a.shape[:2]
        h_b, w_b = mask_b.shape[:2]
        x_min = min(x0_a, x0_b)
        y_min = min(y0_a, y0_b)
        x_max = max(x0_a + w_a, x0_b + w_b)
        y_max = max(y0_a + h_a, y0_b + h_b)
        canvas_a = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_b = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
        canvas_a[y0_a - y_min:y0_a - y_min + h_a, x0_a - x_min:x0_a - x_min + w_a] = mask_a
        canvas_b[y0_b - y_min:y0_b - y_min + h_b, x0_b - x_min:x0_b - x_min + w_b] = mask_b
        intersection = int((canvas_a & canvas_b).sum())
        union = int((canvas_a | canvas_b).sum())
        rec["iou"] = intersection / union if union > 0 else 0.0

        ys_a, xs_a = np.where(mask_a)
        ys_b, xs_b = np.where(mask_b)
        cx_a, cy_a = float(xs_a.mean()) + x0_a, float(ys_a.mean()) + y0_a
        cx_b, cy_b = float(xs_b.mean()) + x0_b, float(ys_b.mean()) + y0_b
        rec["centroid_drift_px"] = float(np.hypot(cx_b - cx_a, cy_b - cy_a))
        out.append(rec)
    return out


def summarize_directional_disagreement(records: list[dict], *,
                                       low_iou_threshold: float = 0.5) -> dict:
    """Aggregate directional_disagreement records (one chain's, or a run's concatenated).

    n_dropout_z counts z's where either tracing was empty separately from the IoU/drift
    means, mirroring summarize_z_consistency's dropout handling. frac_low_agreement is
    the fraction of SCORED z's below low_iou_threshold, the direct "how often would the
    RoboEM keep-only-where-they-agree rule have flagged this chain" reading."""
    n = len(records)
    dropout = [r for r in records if r["iou"] is None]
    scored = [r for r in records if r["iou"] is not None]
    return {
        "n_z": n,
        "n_dropout_z": len(dropout),
        "mean_disagreement_iou": float(np.mean([r["iou"] for r in scored])) if scored else None,
        "mean_centroid_drift_px": (
            float(np.mean([r["centroid_drift_px"] for r in scored])) if scored else None),
        "frac_low_agreement": (
            sum(1 for r in scored if r["iou"] < low_iou_threshold) / len(scored)
        ) if scored else None,
    }
```

- [x] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_directional_disagreement.py -v`
Expected: PASS, 8 passed

- [x] **Step 5: Commit**

```bash
git add eval/merge_metric.py tests/test_directional_disagreement.py
git commit -m "feat(eval): add directional (fwd/back) disagreement scoring"
```

---

### Task 2: `propagate_directional` in `pipeline/propagate.py`

**Files:**
- Modify: `pipeline/propagate.py` (add function after `propagate_from_verified_masks`, i.e. after
  line 359)
- Test: `tests/test_propagate_directional.py` (new)

**Interfaces:**
- Consumes: `PropagationSession` (existing, `pipeline/propagate.py`), `Prompts` (existing,
  `pipeline/state.py`).
- Produces: `propagate_directional(video_predictor, frames_dir: str, prompts: Prompts, seed_frame_idx: int, *, obj_id: int, reverse: bool, seed_negatives: bool = False, seed_box: bool = True, seed_points: bool = True, subtimings: Optional[dict] = None) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]`,
  same return shape as `propagate()`. Task 3 calls this twice: once with `seed_frame_idx=0,
  reverse=False` (forward-only from the chain's start) and once with
  `seed_frame_idx=n_frames-1, reverse=True` (backward-only from the chain's end).

- [x] **Step 1: Write the failing tests**

Create `tests/test_propagate_directional.py`:

```python
import importlib

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

prop = importlib.import_module("pipeline.propagate")
from pipeline.state import Prompts


class _FakeVideoPredictor:
    """Minimal fake covering both seeding paths propagate_directional needs:
    add_new_points_or_box (point/box seed) and propagate_in_video (single-direction
    sweep). Mirrors the real predictor's documented range behaviour: reverse=False
    covers [start, num_frames-1], reverse=True covers [0, start]."""

    def __init__(self, num_frames: int, hw=(4, 4)):
        self.num_frames = num_frames
        self.hw = hw
        self.seeded_at: list[int] = []
        self.visits: list[int] = []

    def init_state(self, video_path, offload_video_to_cpu=True):
        return {"cond_frame": None}

    def reset_state(self, inference_state):
        inference_state["cond_frame"] = None

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None,
                              points=None, labels=None, clear_old_points=False):
        inference_state["cond_frame"] = int(frame_idx)
        self.seeded_at.append(int(frame_idx))

    def propagate_in_video(self, inference_state, start_frame_idx=None,
                           max_frame_num_to_track=None, reverse=False):
        start = inference_state["cond_frame"] if start_frame_idx is None else start_frame_idx
        n = self.num_frames
        if max_frame_num_to_track is None:
            max_frame_num_to_track = n
        if reverse:
            end = max(start - max_frame_num_to_track, 0)
            order = range(start, end - 1, -1) if start > 0 else []
        else:
            end = min(start + max_frame_num_to_track, n - 1)
            order = range(start, end + 1)
        h, w = self.hw
        for f in order:
            self.visits.append(f)
            mask_logits = torch.full((1, h, w), 10.0, dtype=torch.float32)
            yield f, [1], mask_logits


def _prompts():
    return Prompts(points_sam=np.asarray([[2.0, 2.0]]), labels=np.asarray([1]))


def test_forward_from_start_covers_start_to_end():
    vp = _FakeVideoPredictor(num_frames=10)
    video_segments, _conf, _iou = prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=0, obj_id=1, reverse=False)
    assert vp.seeded_at == [0]
    assert sorted(video_segments.keys()) == list(range(10))
    assert vp.visits == list(range(0, 10))


def test_backward_from_end_covers_end_to_start():
    vp = _FakeVideoPredictor(num_frames=10)
    video_segments, _conf, _iou = prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=9, obj_id=1, reverse=True)
    assert vp.seeded_at == [9]
    assert sorted(video_segments.keys()) == list(range(10))
    assert vp.visits == list(range(9, -1, -1))


def test_only_seeds_once_not_bidirectional():
    vp = _FakeVideoPredictor(num_frames=10)
    prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=0, obj_id=1, reverse=False)
    assert len(vp.seeded_at) == 1
```

- [x] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_propagate_directional.py -v`
Expected: FAIL with `AttributeError: module 'pipeline.propagate' has no attribute 'propagate_directional'`

- [x] **Step 3: Implement `propagate_directional`**

In `pipeline/propagate.py`, insert immediately after `propagate_from_verified_masks` (after line
359, before `def _node_id_at`):

```python
def propagate_directional(video_predictor, frames_dir: str, prompts: Prompts,
                          seed_frame_idx: int, *, obj_id: int, reverse: bool,
                          seed_negatives: bool = False, seed_box: bool = True,
                          seed_points: bool = True,
                          subtimings: Optional[dict] = None
                          ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]:
    """Seed ONE frame, propagate in ONE direction only, no return sweep. The building
    block for an independent directional-disagreement pilot (see
    experiments/directional_disagreement_pilot.py and eval.merge_metric.
    directional_disagreement): call this once with seed_frame_idx=0, reverse=False (a
    forward-only tracing seeded at the chain's own start) and once with
    seed_frame_idx=n_frames-1, reverse=True (a backward-only tracing seeded at the
    chain's own end), so the two calls' outputs are two INDEPENDENT tracings of the
    whole chain, unlike propagate()'s single-mid-anchor run_bidirectional(), which
    seeds once and sweeps both directions from the SAME frame, covering each frame
    exactly once (verified against the installed sam2_video_predictor.py: reverse=False
    covers [start, num_frames-1], reverse=True covers [0, start], strictly disjoint
    except at start itself), never producing two masks to compare.

    Returns the same shape as propagate(): (video_segments, frame_conf, pred_iou).
    """
    _t = perf_counter()
    session = PropagationSession(video_predictor, frames_dir, obj_id=obj_id)
    if subtimings is not None:
        subtimings["jpeg_load"] = perf_counter() - _t
    try:
        session.seed(prompts, seed_frame_idx, seed_box=seed_box,
                     seed_points=seed_points, seed_negatives=seed_negatives)
        _t = perf_counter()
        for _ in session.propagate(reverse=reverse):
            pass
        if subtimings is not None:
            subtimings["propagate_only"] = perf_counter() - _t
        return session.video_segments, session.frame_conf, session.pred_iou
    finally:
        session.close()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_propagate_directional.py -v`
Expected: PASS, 3 passed

- [x] **Step 5: Commit**

```bash
git add pipeline/propagate.py tests/test_propagate_directional.py
git commit -m "feat(pipeline): add propagate_directional, a single-direction sweep primitive"
```

---

### Task 3: `experiments/directional_disagreement_pilot.py`

**Files:**
- Create: `experiments/directional_disagreement_pilot.py`

**Interfaces:**
- Consumes: `eval.merge_metric.directional_disagreement`,
  `eval.merge_metric.summarize_directional_disagreement` (Task 1);
  `pipeline.propagate.propagate_directional` (Task 2); `pipeline.predict.centreline_by_z`,
  `pipeline.propagate.build_prompts` / `_node_id_at` (existing, same construction
  `segment_per_slice` uses); `pipeline.crop.prepare_chain_crop_frames` /
  `prepare_video_frames`, `pipeline.state.load_state`, `sam2_utils.alignment`, `sam2_utils.config`,
  `sam2_utils.setup` (existing, same imports `experiments/propagate_from_corrected_seed.py` uses).
- Produces: a `main(argv=None)` CLI entry point; no other task depends on this script's internals.

- [x] **Step 1: Write the script**

Create `experiments/directional_disagreement_pilot.py`:

```python
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

    cfg = config.PipelineConfig()
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
```

- [x] **Step 2: Smoke-test the script imports and argument parsing**

Run: `py -3 -c "import experiments.directional_disagreement_pilot as m; m.main(['--help'])"`
Expected: argparse help text printed, exit code 0 (SystemExit(0) from `--help` is expected and fine)

- [x] **Step 3: Commit**

```bash
git add experiments/directional_disagreement_pilot.py
git commit -m "feat(experiments): add directional disagreement pilot script"
```

---

### Task 4: Run the pilot on a real chain and record results

**Files:**
- Create: `docs/figures/directional_disagreement/<neuron>_chain<NN>.csv` (per-z output, one file
  per chain run)

**Interfaces:**
- Consumes: Task 3's script, real data at `F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR`
  (confirmed present on this machine 2026-09-03) and CUDA (confirmed available: torch 2.12+cu130,
  `torch.cuda.is_available()` True).
- Produces: real `mean_disagreement_iou` / `frac_low_agreement` / timing numbers for the final
  report; no other task depends on this task's output programmatically.

- [x] **Step 1: Run on AIYL chain_00**

Run:
```
py -3 experiments/directional_disagreement_pilot.py --working "F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR" --neuron AIYL --chain 0 --out-csv "docs/figures/directional_disagreement/AIYL_chain00.csv"
```
Expected: prints frame prep, forward/backward timing, and a `[dirdis] n_z_compared=... mean_iou=...`
summary line; writes the CSV. Record the printed numbers verbatim for the report — do not
paraphrase or round further than the script already does.

- [x] **Step 2: Run on one more chain for a second data point (AIYR chain_00, or the shortest
  available chain in the same tree if AIYR chain_00 errors)**

Run:
```
py -3 experiments/directional_disagreement_pilot.py --working "F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR" --neuron AIYR --chain 0 --out-csv "docs/figures/directional_disagreement/AIYR_chain00.csv"
```
Expected: same shape of output as Step 1.

- [x] **Step 3: If either run errors, capture the real error text** (do not silently skip or
  fabricate a result) and note it plainly in the final report instead of a number for that chain.

---

### Task 5: Correct roadmap.md §4.2's cost claim and record the real pilot numbers

**Files:**
- Modify: `docs/explanation/roadmap.md` (the bullet at line 197-202, §4.2's first bullet)

**Interfaces:**
- Consumes: Task 4's real numbers.
- Produces: nothing further downstream; this is the documentation close-out.

- [x] **Step 1: Replace the inaccurate "near-zero cost" framing**

In `docs/explanation/roadmap.md`, replace the existing bullet (currently reading "**Forward/backward
propagation consistency (training-free, do first).** ... We already run both passes, so scoring
per-frame +z vs −z mask disagreement is near-zero cost and hits our dominant merge and abrupt-jump
modes directly.") with a corrected version that: (a) keeps the RoboEM citation and rationale, (b)
states plainly that the CURRENT single-mid-anchor bidirectional sweep gives each frame only one
mask, so real per-frame disagreement scoring needs two independent full sweeps seeded at the
chain's own start/end frames (~2x compute, not free), (c) reports the real pilot numbers from Task
4 (`mean_disagreement_iou`, `frac_low_agreement`, and the measured wall-clock cost) and a pointer to
`experiments/directional_disagreement_pilot.py` and the CSVs under
`docs/figures/directional_disagreement/`, and (d) states the resulting recommendation: whether the
measured disagreement rate looks like a usable QC signal worth wiring into `batch.py` as a real
option, based on whether `frac_low_agreement` picks out a plausible-sized minority of frames (e.g.
a rate in the same ballpark as the existing `foreign_frame_rate`/`mild_bleed_rate` figures already
in this doc) rather than flagging almost everything or almost nothing.

- [x] **Step 2: Commit**

```bash
git add docs/explanation/roadmap.md
git commit -m "docs: correct §4.2 disagreement-QC cost claim, record real pilot numbers"
```

---

## Self-Review Notes (for the implementer to re-check before starting)

- **Spec coverage:** Task 1 = the scoring primitive (testable in isolation, no GPU). Task 2 = the
  propagation primitive (testable with a fake predictor, no GPU/real model). Task 3 = wiring them
  together against real chain data (needs GPU + the F: drive). Task 4 = actually running it and
  recording real numbers, not simulated ones. Task 5 = closing the loop on the doc's own inaccurate
  claim this plan's Global Constraints section already caught. All five map directly to what the
  student asked for: build the pilot, run it, report results.
- **No placeholders:** every step above has literal code or an exact command, not a description of
  what to write.
- **Type consistency check:** `directional_disagreement` (Task 1) takes `dict[int, tuple[np.ndarray,
  int, int]]` keyed by `z`; Task 3's script builds exactly that shape
  (`{frame_to_z[fi]: (seg[1], 0, 0) ...}`) before calling it. `propagate_directional` (Task 2)
  returns the same 3-tuple shape as the existing `propagate()`; Task 3 unpacks it the same way
  `propagate_from_corrected_seed.py` unpacks `propagate()`'s return. Confirmed consistent.
