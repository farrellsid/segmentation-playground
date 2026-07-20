# Phase 1 close-out: blow-up guard + generous-first/neg-crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-slice blow-up guard and a per-pass tier-2 seed config (for a generous-first / negatives-in-crop bundle), plus the presets and tests to measure both.

**Architecture:** The guard is a pure post-pass over `segment_per_slice`'s collected masks in `pipeline/propagate.py`, gated by a new config flag. The bundle reuses the existing `tier2_all` + `chain_crop_from_mask` two-pass machinery; the only new capability is per-pass tier-2 seed overrides applied in `batch._run_one_chain`'s rerun. Both are gated off by default so existing runs are byte-identical.

**Tech Stack:** Python, numpy, dataclasses (`PipelineConfig`), pytest. Tests are CPU-only and torch-free.

## Global Constraints

- No em dashes anywhere (code, comments, docstrings, commit messages).
- Tests CPU-only and torch-free: `py -3 -m pytest`.
- Lint with `ruff check .`; only clean files you touch.
- The library (`pipeline/`, `sam2_utils/`) must never import the drivers (`batch`, `gui`, `run_aval`) or `eval`.
- Defaults: `blowup_area_factor=25.0`, guard `min_accepted=3`. Bundle first-pass seeds generous+no-neg (`multimask_generous=True, k_max_neg=0, seed_negatives=False`); tier-2 overrides neg-on (`tier2_k_max_neg=3, tier2_seed_negatives=True, tier2_multimask_generous=False`).
- All new config flags default to the no-op value (`False`/`None`) so current behaviour is unchanged.

---

### Task 1: Per-slice blow-up guard

**Files:**
- Modify: `pipeline/config.py` (add two fields near `per_slice_reseed`, line ~211)
- Modify: `pipeline/propagate.py` (add `apply_blowup_guard`, call it at the end of `segment_per_slice`)
- Test: `tests/test_blowup_guard.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `apply_blowup_guard(video_segments, frame_conf, pred_iou, *, obj_id, area_factor, min_accepted=3) -> set[int]` (mutates the dicts in place, returns the guarded frame indices); `PipelineConfig.blowup_guard: bool`, `PipelineConfig.blowup_area_factor: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blowup_guard.py
import numpy as np
from pipeline.propagate import apply_blowup_guard


def _mask(n_true, shape=(50, 50)):
    m = np.zeros(shape, dtype=bool)
    m.flat[:n_true] = True
    return m


def _chain(areas, obj_id=1):
    vs = {i: {obj_id: _mask(a)} for i, a in enumerate(areas)}
    return vs, {i: 0.9 for i in range(len(areas))}, {i: 0.9 for i in range(len(areas))}


def test_guard_replaces_blowup_with_nearest_accepted():
    vs, fc, pi = _chain([100, 110, 90, 100, 2000, 105])  # frame 4 is 2000 vs median ~103
    guarded = apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0)
    assert guarded == {4}
    assert int(vs[4][1].sum()) == int(vs[3][1].sum())     # replaced by nearest accepted (frame 3)
    assert fc[4] == 0.0 and pi[4] == 0.0                  # flagged for review
    assert fc[0] == 0.9                                   # others untouched


def test_guard_ignores_normal_variation():
    vs, fc, pi = _chain([100, 100, 100, 100, 900])        # 9x median, below 25x
    assert apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0) == set()
    assert int(vs[4][1].sum()) == 900                     # unchanged


def test_guard_noop_when_too_few_accepted():
    vs, fc, pi = _chain([100, 5000])                      # only 2 non-empty -> no baseline
    assert apply_blowup_guard(vs, fc, pi, obj_id=1, area_factor=25.0, min_accepted=3) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_blowup_guard.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_blowup_guard'`.

- [ ] **Step 3: Write the config fields**

In `pipeline/config.py`, immediately after the `per_slice_reseed: bool = False` field (line ~211), add:

```python
    # Per-slice blow-up guard (roadmap Phase 1 close-out). When True (per_slice_reseed only),
    # a post-pass rejects any slice whose mask area exceeds blowup_area_factor times the
    # chain's median non-empty area (a per-frame explosion with no memory to catch it) and
    # replaces it with the nearest accepted slice's mask, flagging the guarded frames for
    # review. Default False keeps segment_per_slice's output unchanged.
    blowup_guard: bool = False
    blowup_area_factor: float = 25.0
```

- [ ] **Step 4: Write `apply_blowup_guard` and call it**

In `pipeline/propagate.py`, add this function just above `segment_per_slice`:

```python
def apply_blowup_guard(video_segments: dict[int, dict[int, np.ndarray]],
                       frame_conf: dict[int, float], pred_iou: dict[int, float],
                       *, obj_id: int, area_factor: float, min_accepted: int = 3) -> set[int]:
    """Replace per-slice masks that blow up (area > area_factor * median non-empty area)
    with the nearest accepted slice's mask, and flag the guarded frames (frame_conf and
    pred_iou -> 0.0 so QC queues them). Mutates the dicts in place; returns the guarded
    frame indices. No-op (returns empty) when fewer than min_accepted non-empty masks exist
    or the median is 0, so a short or mostly-empty chain sets no spurious baseline."""
    areas = {fi: int(seg[obj_id].sum()) for fi, seg in video_segments.items() if obj_id in seg}
    nonempty = {fi: a for fi, a in areas.items() if a > 0}
    if len(nonempty) < min_accepted:
        return set()
    med = float(np.median(list(nonempty.values())))
    if med <= 0:
        return set()
    cap = area_factor * med
    blown = {fi for fi, a in nonempty.items() if a > cap}
    accepted = sorted(fi for fi in nonempty if fi not in blown)
    if not accepted:
        return set()
    for fi in blown:
        nearest = min(accepted, key=lambda j: abs(j - fi))
        video_segments[fi][obj_id] = video_segments[nearest][obj_id].copy()
        frame_conf[fi] = 0.0
        pred_iou[fi] = 0.0
    return blown
```

Then, in `segment_per_slice`, replace the final `return video_segments, frame_conf, pred_iou` line with:

```python
    if getattr(cfg, "blowup_guard", False):
        guarded = apply_blowup_guard(video_segments, frame_conf, pred_iou,
                                     obj_id=obj_id, area_factor=cfg.blowup_area_factor)
        if guarded:
            print(f"    [blow-up guard] replaced {len(guarded)} slice(s) over "
                  f"{cfg.blowup_area_factor}x median area: {sorted(guarded)}")
    return video_segments, frame_conf, pred_iou
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_blowup_guard.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full suite and lint**

Run: `py -3 -m pytest -q && ruff check pipeline/config.py pipeline/propagate.py tests/test_blowup_guard.py`
Expected: green, clean.

- [ ] **Step 7: Commit**

```bash
git add pipeline/config.py pipeline/propagate.py tests/test_blowup_guard.py
git commit -m "feat(pipeline): per-slice blow-up guard (Phase 1 close-out)"
```

---

### Task 2: Per-pass tier-2 seed overrides

**Files:**
- Modify: `pipeline/config.py` (add three fields near the seed block, line ~205)
- Modify: `batch.py` (add `_tier2_overrides`, use it in `_run_one_chain`'s rerun, line ~419)
- Test: `tests/test_tier2_seed_overrides.py`

**Interfaces:**
- Consumes: `PipelineConfig` (Task 1 is independent; no dependency).
- Produces: `PipelineConfig.tier2_k_max_neg: Optional[int]`, `tier2_seed_negatives: Optional[bool]`, `tier2_multimask_generous: Optional[bool]`; `batch._tier2_overrides(cfg) -> dict` mapping the non-None tier-2 fields to their base-config names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tier2_seed_overrides.py
from dataclasses import replace
from pipeline.config import PipelineConfig
from batch import _tier2_overrides


def test_overrides_empty_when_all_none():
    assert _tier2_overrides(PipelineConfig()) == {}


def test_overrides_map_set_fields_to_base_names():
    cfg = PipelineConfig(tier2_k_max_neg=3, tier2_seed_negatives=True,
                         tier2_multimask_generous=False)
    assert _tier2_overrides(cfg) == {
        "k_max_neg": 3, "seed_negatives": True, "multimask_generous": False}


def test_all_none_rerun_equals_plain_chain_crop():
    cfg = PipelineConfig()
    assert replace(cfg, chain_crop=True, **_tier2_overrides(cfg)) == replace(cfg, chain_crop=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_tier2_seed_overrides.py -v`
Expected: FAIL with `ImportError: cannot import name '_tier2_overrides'`.

- [ ] **Step 3: Write the config fields**

In `pipeline/config.py`, immediately after `box_margin_frac` (line ~205, the end of the video-seed block), add:

```python
    # Per-pass tier-2 seed overrides (roadmap Phase 1 close-out). When set (not None), the
    # tier-2 rerun (batch, under tier2_all or tier2_on_flagged) applies these instead of the
    # base value, so the first _sam pass and the tier-2 crop pass can seed differently, e.g.
    # a generous, negative-free first pass to size the crop (chain_crop_from_mask), then
    # negatives in the crop. None = inherit the base value (current behaviour).
    tier2_k_max_neg: Optional[int] = None
    tier2_seed_negatives: Optional[bool] = None
    tier2_multimask_generous: Optional[bool] = None
```

- [ ] **Step 4: Write `_tier2_overrides` and use it in the rerun**

In `batch.py`, add near `_run_one_chain` (above it):

```python
def _tier2_overrides(cfg) -> dict:
    """The non-None tier-2 seed overrides, mapped to their base PipelineConfig names, for
    the tier-2 rerun. Empty when none are set, so replace(cfg, chain_crop=True, **overrides)
    is identical to the plain rerun."""
    out = {}
    if cfg.tier2_k_max_neg is not None:
        out["k_max_neg"] = cfg.tier2_k_max_neg
    if cfg.tier2_seed_negatives is not None:
        out["seed_negatives"] = cfg.tier2_seed_negatives
    if cfg.tier2_multimask_generous is not None:
        out["multimask_generous"] = cfg.tier2_multimask_generous
    return out
```

Then change the rerun call in `_run_one_chain` (line ~419) from:

```python
        state = _run_chain_once(session, replace(cfg, chain_crop=True),
                                neuron, chain_idx, chain)
```

to:

```python
        state = _run_chain_once(session, replace(cfg, chain_crop=True, **_tier2_overrides(cfg)),
                                neuron, chain_idx, chain)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_tier2_seed_overrides.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full suite and lint**

Run: `py -3 -m pytest -q && ruff check pipeline/config.py batch.py tests/test_tier2_seed_overrides.py`
Expected: green, clean.

- [ ] **Step 7: Commit**

```bash
git add pipeline/config.py batch.py tests/test_tier2_seed_overrides.py
git commit -m "feat(batch): per-pass tier-2 seed overrides for the two-pass rerun (Phase 1 close-out)"
```

---

### Task 3: Presets

**Files:**
- Modify: `sam2_utils/presets.py` (add three presets after `original_generous_only`, line ~233)
- Test: `tests/test_phase1_closeout_presets.py`

**Interfaces:**
- Consumes: `PipelineConfig.blowup_guard`, `tier2_k_max_neg`, `tier2_seed_negatives`, `tier2_multimask_generous` (Tasks 1 and 2).
- Produces: presets `original_perslice_only_guard`, `original_perslice_guard`, `original_genfirst_negcrop`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase1_closeout_presets.py
from sam2_utils.presets import PRESETS


def test_perslice_guard_presets_set_blowup_guard():
    for name in ("original_perslice_only_guard", "original_perslice_guard"):
        p = PRESETS[name]["pipeline"]
        assert p["per_slice_reseed"] is True
        assert p["blowup_guard"] is True


def test_genfirst_negcrop_seeds_split_across_passes():
    p = PRESETS["original_genfirst_negcrop"]["pipeline"]
    # first _sam pass: generous, no negatives
    assert p["multimask_generous"] is True
    assert p["k_max_neg"] == 0 and p["seed_negatives"] is False
    assert p["chain_crop_from_mask"] is True
    # tier-2 crop pass overrides: negatives on, not generous
    assert p["tier2_k_max_neg"] == 3 and p["tier2_seed_negatives"] is True
    assert p["tier2_multimask_generous"] is False
    assert PRESETS["original_genfirst_negcrop"]["tier2_all"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_phase1_closeout_presets.py -v`
Expected: FAIL with `KeyError: 'original_perslice_only_guard'`.

- [ ] **Step 3: Add the presets**

In `sam2_utils/presets.py`, before the closing `}` of the presets dict (after `original_generous_only`, line ~233), add:

```python
    "original_perslice_only_guard": {
        # original_perslice_only + the blow-up guard (Phase 1 close-out). A/B vs the guard-off
        # tree: does capping the gross per-slice tail cut total_foreign while the clean bulk holds.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "chain_crop_min_image_score": 0.0,
                     "seed_negatives": True,
                     "chain_crop_scale": 1, "chain_crop_max_px": 2048,
                     "per_slice_reseed": True, "blowup_guard": True},
        "output_root": config.OUTPUT_ROOT.parent / "exp_perslice_only_guard",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": True, "tier2_all": True, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
    "original_perslice_guard": {
        # original_perslice (per-slice + generous) + the blow-up guard (Phase 1 close-out).
        "dataset": "target",
        "pipeline": {**_PIPELINE, "chain_crop_min_image_score": 0.0,
                     "seed_negatives": True,
                     "chain_crop_scale": 1, "chain_crop_max_px": 2048,
                     "per_slice_reseed": True, "multimask_anchor": True,
                     "multimask_generous": True, "blowup_guard": True},
        "output_root": config.OUTPUT_ROOT.parent / "exp_perslice_guard",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": True, "tier2_all": True, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
    "original_genfirst_negcrop": {
        # Generous-first-pass, negatives-in-crop bundle (Phase 1 close-out). TIER2_ALL two-pass:
        # a generous, negative-free _sam first pass sizes the crop (chain_crop_from_mask), then
        # negatives in the tier-2 crop via the tier2_* overrides. Not generous in the crop pass:
        # the first pass is generous so the crop is not clipped, the crop pass wants precision.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "chain_crop_min_image_score": 0.0,
                     "chain_crop_scale": 1, "chain_crop_max_px": 2048,
                     "chain_crop_from_mask": True,
                     "multimask_anchor": True, "multimask_generous": True,
                     "k_max_neg": 0, "seed_negatives": False,
                     "tier2_k_max_neg": 3, "tier2_seed_negatives": True,
                     "tier2_multimask_generous": False},
        "output_root": config.OUTPUT_ROOT.parent / "exp_genfirst_negcrop",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": True, "tier2_all": True, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
```

Note: confirm `_PIPELINE` does not already fix `k_max_neg`/`multimask_anchor` in a way that conflicts; the explicit keys above override `_PIPELINE` since they come after the spread. If `_PIPELINE` sets `k_max_neg`, the explicit `k_max_neg: 0` still wins.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_phase1_closeout_presets.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite and lint**

Run: `py -3 -m pytest -q && ruff check sam2_utils/presets.py tests/test_phase1_closeout_presets.py`
Expected: green, clean.

- [ ] **Step 6: Commit**

```bash
git add sam2_utils/presets.py tests/test_phase1_closeout_presets.py
git commit -m "feat(presets): Phase 1 close-out presets (per-slice guard + genfirst/neg-crop)"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/reference/configuration.md`, `docs/explanation/roadmap.md`, `docs/CHANGELOG.md`

**Interfaces:**
- Consumes: the landed code from Tasks 1 to 3.
- Produces: documentation only.

- [ ] **Step 1: Update the reference doc**

`docs/reference/configuration.md`: document the new `PipelineConfig` flags: `blowup_guard`, `blowup_area_factor` (per-slice-only post-pass, median-factor cap + neighbour fallback), and `tier2_k_max_neg` / `tier2_seed_negatives` / `tier2_multimask_generous` (None = inherit; applied on the tier-2 rerun). Note the three new presets and what each measures. Read the existing sections first and match their style; present tense, no change-narration.

- [ ] **Step 2: Update roadmap and CHANGELOG**

`docs/explanation/roadmap.md`: in the Phase 1 section, note the close-out levers landed (blow-up guard, generous-first/neg-crop bundle) and that the Phase-1 exit decision waits on their CCDB A/B. `docs/CHANGELOG.md`: add an entry for the Phase 1 close-out (both features + presets), referencing the spec `docs/superpowers/specs/2026-07-17-phase1-blowup-guard-and-genfirst-negcrop-design.md`. Match existing entry style.

- [ ] **Step 3: Humanize and verify**

Run the `humanizer` skill on the CHANGELOG and roadmap prose. Then:
Run: `py -3 -m pytest -q && ruff check .`
Expected: green, clean. Confirm no em dashes in touched docs and that internal links resolve.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: Phase 1 close-out (blow-up guard + genfirst/neg-crop presets)"
```

---

## Post-plan: the CCDB batch

Delivered as copy-pasteable commands after the code lands (paths depend on the run). Shape:
1. `sbatch run_exp.sh` for each of the three new presets (`original_perslice_only_guard`, `original_perslice_guard`, `original_genfirst_negcrop`).
2. `sbatch --dependency=afterok:<id> run_merge_exp.sh` per preset.
3. A CPU big-memory `retro_eval --membrane --min-scale 1` over all trees (finishes the report).
4. An `afterok` `retro_eval` over the three new merged trees, so the comparison CSV is produced on-cluster.

Regenerate `cluster/exp_neuron_chunks.txt` only if EXP_NEURONS changed (it did not); the arrays reuse the existing 16-chunk layout.
