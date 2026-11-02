# Item 3: generous-capped multimask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** When selecting among SAM2's multimask candidates, prefer a LARGER candidate (so a soma mask includes the nucleus and reaches the outer membrane) while hard-rejecting whole-frame blobs, so nested-membrane objects stop being segmented as just the nucleus.

**Architecture:** A new selection mode inside the existing `_select_anchor_mask` (`pipeline/predict.py`), gated by `PipelineConfig.multimask_generous` (default False), consulted only when `multimask_anchor` is on. Pure ranking-key change; no new prediction path. Self-contained and CPU-testable with synthetic candidate masks.

**Tech Stack:** Python, numpy, pytest (CPU-only, torch-free).

## Global Constraints

- No em dashes anywhere in code, comments, or commit messages.
- Tests CPU-only, torch-free.
- `ruff check .` clean; touch only the files each task names.
- Import direction unchanged (this is all inside `pipeline`).
- Default byte-identical when `multimask_generous` is False: the current ranking key must be reproduced exactly in that case.
- "Generous" is CAPPED: never select a candidate whose area fraction exceeds the max-area cap (SAM2's largest candidate is frequently the whole frame). Leeway on the bounds scales with resolution, reusing the caller's existing `scale/crop_scale` rescale.

---

### Task 1: `multimask_generous` config flag

**Files:**
- Modify: `pipeline/config.py`
- Test: `tests/test_generous_multimask.py`

**Interfaces:**
- Produces: `PipelineConfig.multimask_generous: bool = False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generous_multimask.py
from pipeline import PipelineConfig

def test_multimask_generous_defaults_false():
    assert PipelineConfig().multimask_generous is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/test_generous_multimask.py::test_multimask_generous_defaults_false -v`
Expected: FAIL on the missing field.

- [ ] **Step 3: Implement**

Add to `PipelineConfig` near `multimask_anchor` / `multimask_exclude_neg`:

```python
    # Generous-capped multimask pick (roadmap Phase 1 item 3). When True (and
    # multimask_anchor on), among candidates that contain the node, are single-CC, and pass
    # the area gate, prefer the LARGEST area rather than the highest score, so a soma mask
    # includes the nucleus and reaches the outer membrane. The area gate's upper bound still
    # rejects whole-frame blobs. Default False reproduces the current ranking exactly.
    multimask_generous: bool = False
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3 -m pytest tests/test_generous_multimask.py::test_multimask_generous_defaults_false -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_generous_multimask.py
git commit -m "feat(config): add multimask_generous flag (default off)"
```

---

### Task 2: Generous-capped ranking in `_select_anchor_mask`

**Files:**
- Modify: `pipeline/predict.py` (`_select_anchor_mask`, and thread `generous` through `image_predict`)
- Test: `tests/test_generous_multimask.py`

**Context for the implementer:** Read `_select_anchor_mask` and `image_predict` in `pipeline/predict.py`. `_select_anchor_mask` currently ranks the 3 candidates by a tuple key like `(contains_pos, no_neg, area_ok, lcc, score)` and returns the argmax. You are adding a `generous: bool = False` parameter. When `generous` is True, among candidates that already satisfy (contains the positive node, single-CC health, area fraction within `[min, max]`), the tie-break prefers LARGER area instead of higher SAM score. Candidates failing the area cap (`area_frac > max`) must never win (they are already excluded by `area_ok`; keep it that way). When `generous` is False, the key must be byte-identical to today. Thread the flag from `image_predict(..., select_generous=...)` and have callers pass `cfg.multimask_generous` the same way they pass `cfg.multimask_exclude_neg`.

**Interfaces:**
- Consumes: existing `_select_anchor_mask` candidate scoring, Task 1 flag.
- Produces: `_select_anchor_mask(..., generous: bool = False)` and `image_predict(..., select_generous: bool = False)`; the generous key prefers larger area among gate-passing, node-containing, single-CC candidates.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generous_multimask.py (append)
import numpy as np
from pipeline import predict
from pipeline.state import Prompts

def _cand(hw, box):
    m = np.zeros(hw, bool); y0,y1,x0,x1 = box; m[y0:y1, x0:x1] = True; return m

def test_generous_prefers_larger_but_capped():
    hw = (100, 100)
    node = (50.0, 50.0)  # sam-space positive point
    small = _cand(hw, (45, 55, 45, 55))     # nucleus-only, contains node, ~1% area
    soma  = _cand(hw, (30, 70, 30, 70))     # whole soma, contains node, ~16% area
    whole = np.ones(hw, bool)               # whole-frame blob, 100% area (over cap)
    masks = np.stack([small, soma, whole])
    scores = np.array([0.95, 0.80, 0.99])   # SAM scores would pick 'small' or 'whole'
    prompts = Prompts(points_sam=np.array([node]), labels=np.array([1]))
    # strict (default): highest score among gate-passers -> 'small' (nucleus)
    idx_strict, _, _ = predict._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(0.001, 0.5))
    # generous: prefer larger area among gate-passers, but 'whole' is over the 0.5 cap -> 'soma'
    idx_gen, mask_gen, _ = predict._select_anchor_mask(
        masks, scores, prompts, hw, contain_radius_px=3, area_bounds=(0.001, 0.5), generous=True)
    assert idx_gen == 1                      # soma, not nucleus, not whole-frame
    assert idx_gen != idx_strict
    assert mask_gen.sum() == soma.sum()
```

Note to implementer: match the real `_select_anchor_mask` signature (parameter names/order may differ from the sketch above); adjust the test call to the true signature while keeping the assertion (generous picks the bounded larger `soma`, strict does not, and the over-cap `whole` never wins).

- [ ] **Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/test_generous_multimask.py::test_generous_prefers_larger_but_capped -v`
Expected: FAIL (no `generous` kwarg, or strict ranking returned).

- [ ] **Step 3: Implement**

Add `generous` to `_select_anchor_mask`. Keep the gate predicates (contains_pos, no_neg if exclude_neg, area within bounds, single-CC) exactly as today. Change only the final tie-break: when `generous`, sort gate-passing candidates by area descending (largest-first) instead of by SAM score; when not `generous`, the key is unchanged. Thread `select_generous` through `image_predict` to `_select_anchor_mask`, and pass `cfg.multimask_generous` at the call sites in `orchestrator.run_chain` / `anchor_crop_predict` alongside `cfg.multimask_exclude_neg`.

- [ ] **Step 4: Run to verify it passes**

Run: `py -3 -m pytest tests/test_generous_multimask.py -v` then `py -3 -m pytest -q`
Expected: PASS, no regressions (the default-False path must not change any existing anchor-selection test).

- [ ] **Step 5: Commit**

```bash
git add pipeline/predict.py tests/test_generous_multimask.py
git commit -m "feat(pipeline): generous-capped multimask selection (prefer larger, cap whole-frame)"
```

---

### Task 3: Enable on the per-slice preset + smoke

**Files:**
- Modify: `sam2_utils/presets.py`

- [ ] **Step 1: Enable the flag on the experiment preset**

Add `"multimask_anchor": True, "multimask_generous": True` to the `original_perslice` preset's pipeline dict (from the item 1 plan; if item 1 is not yet landed, add a standalone `original_generous` preset mirroring `original_tier2_s1forced_neg` with those two flags). Keep `multimask_exclude_neg` off.

- [ ] **Step 2: Verify the preset builds**

Run: `py -3 -c "from sam2_utils import presets; from pipeline import PipelineConfig; p=presets.get_preset('original_perslice'); c=PipelineConfig(**p['pipeline']); print(c.multimask_anchor, c.multimask_generous)"`
Expected: `True True`.

- [ ] **Step 3: Commit**

```bash
git add sam2_utils/presets.py
git commit -m "feat(presets): enable generous-capped multimask on the per-slice preset"
```

- [ ] **Step 4: HUMAN-RUN GPU smoke (notify)**

Verified for real only on a GPU run (downscaled locally, then CCDB), scored with `eval.merge_metric`: does generous-capped raise soma coverage without raising neurite foreign_frame_rate. Notify the human; do not mark done on unit tests alone.

## Notes for the implementer
- The default (`generous=False`) MUST reproduce the current selection exactly; guard it with a test that the existing `tests/test_anchor_select.py` cases still pass unchanged.
- Do not change the area gate itself; generosity is only a tie-break among candidates that already pass the gate, so the whole-frame cap is already enforced by `area_ok`.
