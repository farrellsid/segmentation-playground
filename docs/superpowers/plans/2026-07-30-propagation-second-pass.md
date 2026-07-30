# Propagation second pass implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in post-hoc pass that re-segments a finished propagation chain's flagged
frames (dropout, foreign-node bleed, lost own node) via a neighbour mask-prompt, guarded against
seeding from a nucleus-capture neighbour, without touching the video predictor's memory bank.

**Architecture:** Two new pure functions (`select_second_pass_frames`, `find_second_pass_neighbour`)
plus one orchestrator (`apply_second_pass`) in `pipeline/propagate.py`, alongside the existing
`apply_blowup_guard`. A new `mask_to_low_res_logits` helper and a `mask_input` parameter on
`image_predict` in `pipeline/predict.py` let a saved boolean mask stand in for SAM2's own low-res
logits hint. `eval.merge_metric.score_chain` (the trigger signal) is called from `batch.py`, never
from `pipeline/`, because `pipeline/propagate.py` sits inside this project's enforced library
boundary and must never import `eval`.

**Tech Stack:** numpy, opencv (`cv2`), pandas, the existing SAM2/SAM3 image predictor interface.
No new dependency.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (`CLAUDE.md`).
- Run the `humanizer` skill on any prose you are about to commit before committing.
- New tests stay torch-free and CPU-only where possible; `py -3 -m pytest`. Tests that exercise
  `image_predict` (which enters `torch.inference_mode()`) use `pytest.importorskip("torch")`,
  matching `tests/test_image_predict_box.py`'s existing pattern; they still run in this environment
  since torch is installed, but stay skippable in principle.
- Lint with `ruff check .`, touching only files you edit.
- `pipeline/*.py` and `sam2_utils/*.py` are the enforced library boundary
  (`tests/test_import_direction.py`): they must never import `batch`, `gui`, `run_aval`,
  `pull_worm`, or `eval`. `eval.merge_metric.score_chain` is called only from `batch.py`.
- This design only touches propagation-mode chains (`not cfg.per_slice_reseed`); per-slice chains
  already have `apply_blowup_guard` and are untouched by this plan.
- `PipelineConfig.second_pass` defaults to `False`; nothing changes for existing runs or presets
  until `batch.py --second-pass` is passed explicitly.
- Commit incrementally, one concern per commit.

---

### Task 1: `mask_to_low_res_logits` and `image_predict`'s `mask_input` parameter

**Files:**
- Modify: `pipeline/predict.py`
- Modify: `tests/test_image_predict_box.py`
- Create: `tests/test_mask_to_low_res_logits.py`

**Interfaces:**
- Produces: `mask_to_low_res_logits(mask: np.ndarray) -> np.ndarray` (returns `1x256x256` float32),
  `image_predict(..., mask_input: Optional[np.ndarray] = None)` (new keyword-only parameter,
  forwarded to `image_predictor.predict(mask_input=...)`, `None` unchanged from today).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mask_to_low_res_logits.py`:

```python
"""Unit tests for pipeline.predict.mask_to_low_res_logits, the helper that lets a
saved boolean mask stand in for SAM2's mask_input (a low-res logits array SAM2
normally gets from a previous prediction iteration, not a binary mask)."""
import cv2
import numpy as np

from pipeline.predict import mask_to_low_res_logits


def test_output_shape_and_dtype():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:60, 10:40] = True
    out = mask_to_low_res_logits(mask)
    assert out.shape == (1, 256, 256)
    assert out.dtype == np.float32


def test_threshold_at_zero_matches_resized_mask():
    mask = np.zeros((100, 80), dtype=bool)
    mask[20:60, 10:40] = True
    out = mask_to_low_res_logits(mask)
    recovered = out[0] > 0.0
    expected = cv2.resize(mask.astype(np.uint8), (256, 256),
                          interpolation=cv2.INTER_NEAREST) > 0
    assert np.array_equal(recovered, expected)


def test_all_false_mask_is_all_negative():
    mask = np.zeros((50, 50), dtype=bool)
    out = mask_to_low_res_logits(mask)
    assert (out < 0).all()


def test_all_true_mask_is_all_positive():
    mask = np.ones((50, 50), dtype=bool)
    out = mask_to_low_res_logits(mask)
    assert (out > 0).all()
```

Extend `tests/test_image_predict_box.py`: the fake predictor's `predict` needs a `mask_input=None`
parameter or it will raise `TypeError` once `image_predict` starts passing that keyword
unconditionally. Change `_FakePredictor.predict`'s signature from:

```python
    def predict(self, *, point_coords, point_labels, box, multimask_output):
        self.last = dict(point_coords=point_coords, point_labels=point_labels,
                         box=box, multimask_output=multimask_output)
```

to:

```python
    def predict(self, *, point_coords, point_labels, box, mask_input=None, multimask_output):
        self.last = dict(point_coords=point_coords, point_labels=point_labels,
                         box=box, mask_input=mask_input, multimask_output=multimask_output)
```

Then add two new test functions at the end of the file, before `_main`:

```python
def test_mask_input_forwards_when_given():
    fp = _FakePredictor()
    pr = Prompts(points_sam=np.array([[2.0, 2.0]]), labels=np.array([1]))
    hint = np.zeros((1, 256, 256), dtype=np.float32)
    pipeline.image_predict(fp, _IMG, pr, mask_input=hint)
    assert fp.last["mask_input"] is hint


def test_mask_input_defaults_to_none():
    fp = _FakePredictor()
    pr = Prompts(points_sam=np.array([[2.0, 2.0]]), labels=np.array([1]))
    pipeline.image_predict(fp, _IMG, pr)
    assert fp.last["mask_input"] is None
```

And register both in `_main`'s `tests` list the same way the existing three are (it already collects
every `test_*` global automatically via `globals().items()`, so no manual registration is needed,
just confirm the new functions are named `test_*` at module level, which they are above).

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_mask_to_low_res_logits.py tests/test_image_predict_box.py -v`
Expected: `test_mask_to_low_res_logits.py` FAILS with `ImportError: cannot import name
'mask_to_low_res_logits'`. `test_image_predict_box.py`'s two new tests FAIL with `AssertionError`
(mask_input key missing from `fp.last`, since `image_predict` does not pass it yet).

- [ ] **Step 3: Implement `mask_to_low_res_logits` and `image_predict`'s `mask_input` parameter**

In `pipeline/predict.py`, add this function directly above `image_predict` (currently at line 231):

```python
def mask_to_low_res_logits(mask: np.ndarray) -> np.ndarray:
    """Convert a full-resolution boolean mask into SAM2's 1x256x256 low-res logits
    format, so a previously-saved mask can stand in for a `mask_input` hint.

    SAM2's `mask_input` contract (verified against the installed SAM2ImagePredictor.predict
    docstring) is a low-resolution mask, `1xHxW` with `H=W=256`, "typically coming from a
    previous prediction iteration", i.e. real logits, not a binary mask. This resizes the
    mask to 256x256 (nearest-neighbour, preserves the hard edge) and maps True to +8.0 and
    False to -8.0, values far enough from 0 that SAM2's internal 0-threshold reliably
    recovers the same shape.
    """
    import cv2

    m = mask.astype(np.uint8)
    resized = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    logits = np.where(resized > 0, 8.0, -8.0).astype(np.float32)
    return logits[np.newaxis, :, :]
```

Change `image_predict`'s signature (currently lines 231-236) from:

```python
def image_predict(image_predictor, image_sam: np.ndarray, prompts: Prompts, *,
                  multimask: bool = False, select_contain_radius_px: int = 0,
                  select_area_bounds: tuple[float, float] = (0.0, 1.0),
                  select_exclude_neg: bool = False,
                  select_generous: bool = False,
                  ) -> tuple[np.ndarray, float, np.ndarray]:
```

to:

```python
def image_predict(image_predictor, image_sam: np.ndarray, prompts: Prompts, *,
                  multimask: bool = False, select_contain_radius_px: int = 0,
                  select_area_bounds: tuple[float, float] = (0.0, 1.0),
                  select_exclude_neg: bool = False,
                  select_generous: bool = False,
                  mask_input: Optional[np.ndarray] = None,
                  ) -> tuple[np.ndarray, float, np.ndarray]:
```

Add one sentence to the docstring (after the existing paragraph about `prompts.box_sam`): `` `mask_input`,
when given, is a `1x256x256` low-res logits array forwarded straight to SAM2's own `mask_input`
parameter (see `mask_to_low_res_logits` to build one from a saved boolean mask); `None` (default) omits
it, unchanged from today. ``

Change the `image_predictor.predict(...)` call (currently lines 268-273) from:

```python
        masks, scores, logits = image_predictor.predict(
            point_coords=pts if has_pts else None,
            point_labels=labs if has_pts else None,
            box=box,
            multimask_output=multimask,
        )
```

to:

```python
        masks, scores, logits = image_predictor.predict(
            point_coords=pts if has_pts else None,
            point_labels=labs if has_pts else None,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_mask_to_low_res_logits.py tests/test_image_predict_box.py -v`
Expected: all tests PASS (4 new + 3 existing in `test_image_predict_box.py`, 4 new in
`test_mask_to_low_res_logits.py`).

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check pipeline/predict.py tests/test_image_predict_box.py tests/test_mask_to_low_res_logits.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add pipeline/predict.py tests/test_image_predict_box.py tests/test_mask_to_low_res_logits.py
git commit -m "feat(predict): add mask_input support and mask_to_low_res_logits"
```

---

### Task 2: `select_second_pass_frames` and `find_second_pass_neighbour`

**Files:**
- Modify: `pipeline/propagate.py`
- Create: `tests/test_second_pass_selection.py`

**Interfaces:**
- Produces: `select_second_pass_frames(records: list[dict]) -> set[int]`,
  `find_second_pass_neighbour(z: int, flagged: set[int], areas_by_z: dict[int, float], *,
  min_area_ratio: float) -> Optional[int]`.
- Consumes: nothing from Task 1 (independent, pure logic, no I/O).

`records` are shaped like `eval.merge_metric.score_chain`'s output: each a dict with at least
`z` (int), `own_contained` (bool), `n_foreign` (int), `empty` (bool). These two functions never
import `eval`, they only assume that dict shape (duck-typed), matching this project's import-direction
constraint (see Global Constraints).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_second_pass_selection.py`:

```python
import importlib

prop = importlib.import_module("pipeline.propagate")


def test_select_flags_empty_foreign_and_not_own_contained():
    records = [
        {"z": 10, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 11, "own_contained": True, "n_foreign": 0, "empty": True},   # dropout
        {"z": 12, "own_contained": True, "n_foreign": 2, "empty": False},  # foreign
        {"z": 13, "own_contained": False, "n_foreign": 0, "empty": False}, # lost own node
    ]
    assert prop.select_second_pass_frames(records) == {11, 12, 13}


def test_select_returns_empty_set_when_nothing_flagged():
    records = [{"z": 10, "own_contained": True, "n_foreign": 0, "empty": False}]
    assert prop.select_second_pass_frames(records) == set()


def test_find_neighbour_picks_nearest_unflagged():
    areas = {10: 100.0, 11: 100.0, 12: 5.0, 13: 5.0, 14: 100.0}
    flagged = {12, 13}
    assert prop.find_second_pass_neighbour(12, flagged, areas, min_area_ratio=0.5) == 11


def test_find_neighbour_skips_nucleus_capture_sized_candidate():
    # z=9 is the naive-nearest unflagged neighbour to z=10, but its area (3.0) is far
    # below the chain's unflagged median (100.0), simulating a nucleus-capture mask that
    # own_contained/n_foreign/empty all missed. The guard must skip it for z=8 instead.
    areas = {8: 100.0, 9: 3.0, 10: 5.0, 13: 100.0}
    flagged = {10}
    assert prop.find_second_pass_neighbour(10, flagged, areas, min_area_ratio=0.5) == 8


def test_find_neighbour_returns_none_when_nothing_unflagged():
    areas = {5: 5.0, 6: 5.0}
    flagged = {5, 6}
    assert prop.find_second_pass_neighbour(5, flagged, areas, min_area_ratio=0.5) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_second_pass_selection.py -v`
Expected: FAIL with `AttributeError: module 'pipeline.propagate' has no attribute
'select_second_pass_frames'`.

- [ ] **Step 3: Implement both functions**

In `pipeline/propagate.py`, add after `apply_blowup_guard` (currently ends at line 352):

```python
def select_second_pass_frames(records: list[dict]) -> set[int]:
    """z's needing re-segmentation: score_chain-shaped records with empty=True (dropout),
    n_foreign > 0 (bleed), or own_contained=False (lost its own node). `records` are plain
    dicts (eval.merge_metric.score_chain's output, computed by the caller), never an eval
    import here: pipeline/ must never import eval (tests/test_import_direction.py)."""
    return {
        int(r["z"]) for r in records
        if r.get("empty") or r.get("n_foreign", 0) > 0 or not r.get("own_contained", True)
    }


def find_second_pass_neighbour(z: int, flagged: set[int], areas_by_z: dict[int, float],
                               *, min_area_ratio: float) -> Optional[int]:
    """Nearest z' (by |z - z'|) that is not itself flagged and whose area is not
    anomalously small relative to the chain's unflagged median area.

    The area floor guards against a nucleus-capture mask (covers only the nested
    nucleus, not the cell) being picked as a neighbour: it has own_contained=True,
    n_foreign=0, and is non-empty, so none of score_chain's three trigger signals catch
    it, but it is typically much smaller than a correctly-segmented full-cell mask.
    Returns None if no eligible z' exists (either every z is flagged, or every unflagged
    z fails the area floor)."""
    candidates = [zz for zz in areas_by_z if zz not in flagged]
    if not candidates:
        return None
    median = float(np.median([areas_by_z[zz] for zz in candidates]))
    floor = min_area_ratio * median
    eligible = [zz for zz in candidates if areas_by_z[zz] >= floor]
    if not eligible:
        return None
    return min(eligible, key=lambda zz: abs(zz - z))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_second_pass_selection.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check pipeline/propagate.py tests/test_second_pass_selection.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add pipeline/propagate.py tests/test_second_pass_selection.py
git commit -m "feat(propagate): add second-pass trigger selection and neighbour search"
```

---

### Task 3: `apply_second_pass` orchestrator and `PipelineConfig` fields

**Files:**
- Modify: `pipeline/propagate.py`
- Modify: `pipeline/config.py`
- Create: `tests/test_second_pass.py`

**Interfaces:**
- Consumes: `mask_to_low_res_logits`, `image_predict(..., mask_input=...)` (Task 1);
  `select_second_pass_frames`, `find_second_pass_neighbour` (Task 2); `predict.centreline_by_z`
  (already exists, `pipeline/predict.py:33`); `predict._point_in_mask` (already exists,
  `pipeline/predict.py:110`); `sam2_utils.qc._iter_mask_paths`, `sam2_utils.qc._load_binary`
  (already exist, `sam2_utils/qc.py:85` and `:96`).
- Produces: `apply_second_pass(image_predictor, frames_dir: str, frame_to_z: dict[int, int],
  cw: Optional["alignment.CropWindow"], chain: dict, annotate_df: pd.DataFrame, chain_dir: Path,
  records: list[dict], *, cfg, min_neighbour_area_ratio: float = 0.5) -> dict[int, str]` (returns
  `{z: "corrected" | "guard_fallback"}` for every frame touched; rewrites those frames' saved mask
  PNGs in place). `PipelineConfig.second_pass: bool = False`,
  `PipelineConfig.second_pass_min_neighbour_area_ratio: float = 0.5`.

- [ ] **Step 1: Confirm the current exact state matches this plan's assumptions**

Run `grep -n "^def centreline_by_z\|^def _point_in_mask\|^def image_predict" pipeline/predict.py`
and confirm `centreline_by_z` and `_point_in_mask` both exist (they should, unmodified by Tasks 1-2;
`image_predict` should now have the `mask_input` parameter from Task 1). Run
`grep -n "^def _iter_mask_paths\|^def _load_binary" sam2_utils/qc.py` and confirm both exist. If
either is missing or has a different signature than described above, STOP and report NEEDS_CONTEXT
describing what's different, rather than guessing.

- [ ] **Step 2: Write the failing test**

Create `tests/test_second_pass.py`:

```python
"""Unit tests for pipeline.propagate.apply_second_pass: given a finished chain's saved
masks and eval.merge_metric.score_chain-shaped records, it re-segments flagged frames via
a neighbour mask-prompt and falls back to a neighbour copy when that is not possible."""
import importlib

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

prop = importlib.import_module("pipeline.propagate")
from pipeline import config as cfgmod
from sam2_utils import qc as qc_mod


class _StubPredictor:
    """set_image records shape; predict returns a mask covering the seeded point,
    sized generously if a mask_input hint was given (simulating the hint improving
    the prediction), narrowly otherwise, so tests can tell the two paths apart."""
    def set_image(self, img):
        self._hw = img.shape[:2]

    def predict(self, point_coords=None, point_labels=None, box=None,
               mask_input=None, multimask_output=False):
        h, w = self._hw
        m = np.zeros((h, w), dtype=bool)
        if point_coords is not None and len(point_coords):
            x, y = int(point_coords[0][0]), int(point_coords[0][1])
            r = 8 if mask_input is not None else 2
            m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        masks = m[None]
        scores = np.array([0.9])
        logits = np.zeros((1, 256, 256), dtype=np.float32)
        return masks, scores, logits


def _write_mask(masks_dir, z, mask):
    cv2.imwrite(str(masks_dir / f"mask_{z:04d}.png"), (mask.astype("uint8") * 255))


def _make_chain(tmp_path, *, hw=(40, 40)):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    masks_dir = tmp_path / "chain" / "masks"
    masks_dir.mkdir(parents=True)
    for i in range(5):
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), np.full((*hw, 3), 127, np.uint8))
    frame_to_z = {i: 1400 + i for i in range(5)}
    return frames_dir, masks_dir, frame_to_z


def test_apply_second_pass_reseeds_a_dropout_frame(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path)
    chain_dir = masks_dir.parent

    good = np.zeros((40, 40), dtype=bool)
    good[15:25, 15:25] = True                 # 100 px, centred on (20, 20)
    for z in (1400, 1401, 1403, 1404):
        _write_mask(masks_dir, z, good)
    _write_mask(masks_dir, 1402, np.zeros((40, 40), dtype=bool))   # dropout frame

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({
        "node_id": ["n0"], "cell_name": ["AVAL"], "z": [1402],
        "x_tif": [160.0], "y_tif": [160.0],     # scale 8 -> _sam (20, 20), inside `good`'s footprint
    })
    records = [
        {"z": 1400, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1401, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1402, "own_contained": False, "n_foreign": 0, "empty": True},
        {"z": 1403, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 1404, "own_contained": True, "n_foreign": 0, "empty": False},
    ]
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)

    assert outcomes == {1402: "corrected"}
    fixed = qc_mod._load_binary(masks_dir / "mask_1402.png")
    assert fixed.any()


def test_apply_second_pass_falls_back_when_whole_chain_flagged(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path, hw=(40, 40))
    chain_dir = masks_dir.parent

    small = np.zeros((40, 40), dtype=bool)
    small[19:21, 19:21] = True
    for z in range(1400, 1405):
        _write_mask(masks_dir, z, small)

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({
        "node_id": ["n0"], "cell_name": ["AVAL"], "z": [1402],
        "x_tif": [160.0], "y_tif": [160.0],
    })
    records = [{"z": z, "own_contained": True, "n_foreign": 1, "empty": False}
              for z in range(1400, 1405)]   # every frame flagged (foreign)
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)

    assert outcomes == {}   # no unflagged neighbour anywhere -> nothing to fall back to either


def test_apply_second_pass_noop_when_nothing_flagged(tmp_path):
    frames_dir, masks_dir, frame_to_z = _make_chain(tmp_path)
    chain_dir = masks_dir.parent
    good = np.zeros((40, 40), dtype=bool)
    good[15:25, 15:25] = True
    for z in range(1400, 1405):
        _write_mask(masks_dir, z, good)

    chain = {"cell_name": "AVAL", "nodes": ["n0"]}
    annotate_df = pd.DataFrame({"node_id": [], "cell_name": [], "z": [], "x_tif": [], "y_tif": []})
    records = [{"z": z, "own_contained": True, "n_foreign": 0, "empty": False}
              for z in range(1400, 1405)]
    cfg = cfgmod.PipelineConfig(scale=8)

    outcomes = prop.apply_second_pass(
        _StubPredictor(), str(frames_dir), frame_to_z, None, chain, annotate_df,
        chain_dir, records, cfg=cfg, min_neighbour_area_ratio=0.5)
    assert outcomes == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_second_pass.py -v`
Expected: FAIL with `AttributeError: module 'pipeline.propagate' has no attribute
'apply_second_pass'`.

- [ ] **Step 4: Add the `PipelineConfig` fields**

In `pipeline/config.py`, add directly after the existing `blowup_area_factor: float = 25.0` line
(currently line 230):

```python

    # Post-hoc second pass for propagation chains (roadmap: propagation second-pass design,
    # 2026-07-30). When True (propagation-mode chains only, cfg.per_slice_reseed=False), a
    # post-pass re-segments frames eval.merge_metric.score_chain flags (dropout, foreign-node
    # bleed, lost own node) via a neighbour mask-prompt, falling back to a neighbour-copy when
    # re-segmentation is not possible or does not pass a sanity check. Default False keeps
    # every existing propagation run byte-identical.
    second_pass: bool = False
    second_pass_min_neighbour_area_ratio: float = 0.5
```

- [ ] **Step 5: Implement `apply_second_pass`**

In `pipeline/propagate.py`, change the import line (currently line 16) from:

```python
from .predict import build_prompts, image_predict
```

to:

```python
from . import predict as predict_mod
from .predict import build_prompts, centreline_by_z, image_predict, mask_to_low_res_logits
```

(`_point_in_mask` is a private name; reach it via `predict_mod._point_in_mask(...)` inside
`apply_second_pass` below, matching this codebase's existing convention of reaching a private
helper through its module reference rather than importing it by name directly, e.g.
`chain_masks_in_sam`'s own `from sam2_utils import qc` then `qc._load_binary(...)`.)

Add `apply_second_pass` after `find_second_pass_neighbour` (the function Task 2 added):

```python
def apply_second_pass(
    image_predictor,
    frames_dir: str,
    frame_to_z: dict[int, int],
    cw: Optional["alignment.CropWindow"],
    chain: dict,
    annotate_df: pd.DataFrame,
    chain_dir: Path,
    records: list[dict],
    *,
    cfg,
    min_neighbour_area_ratio: float = 0.5,
) -> dict[int, str]:
    """Re-segment a finished propagation chain's flagged frames via a neighbour
    mask-prompt, without touching the video predictor's memory bank.

    `records` are eval.merge_metric.score_chain's output for this chain, computed by
    the caller (batch.py): pipeline/ must never import eval directly (see
    tests/test_import_direction.py). Rewrites the touched frames' saved mask PNGs in
    place and returns {z: "corrected" | "guard_fallback"} for every frame touched; a
    frame absent from the returned dict was left untouched (either it was never
    flagged, or no fallback was possible either, e.g. every frame in the chain is
    flagged).
    """
    import cv2

    from sam2_utils import qc as qc_mod   # lazy: keeps pipeline import free of qc's heavy deps

    flagged = select_second_pass_frames(records)
    if not flagged:
        return {}

    chain_dir = Path(chain_dir)
    masks_dir = chain_dir / "masks"
    mask_paths = dict(qc_mod._iter_mask_paths(masks_dir))
    masks_by_z = {z: qc_mod._load_binary(p) for z, p in mask_paths.items()}
    areas_by_z = {z: float(m.sum()) for z, m in masks_by_z.items()}

    z_to_frame_idx = {z: fi for fi, z in frame_to_z.items()}
    centreline = centreline_by_z(chain, annotate_df)

    space_ratio = (float(cfg.scale) / float(cw.crop_scale)) if cw is not None else 1.0
    contain_r = int(round(cfg.qc_skeleton_dilation_px * space_ratio))

    outcomes: dict[int, str] = {}
    for z in sorted(flagged):
        corrected = False
        neighbour_z = find_second_pass_neighbour(
            z, flagged, areas_by_z, min_area_ratio=min_neighbour_area_ratio)

        if neighbour_z is not None and z in z_to_frame_idx and z in centreline:
            frame_idx = z_to_frame_idx[z]
            img_path = Path(frames_dir) / f"{frame_idx:05d}.jpg"
            raw = cv2.imread(str(img_path))
            if raw is not None:
                image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                x_tif, y_tif = centreline[z]
                pos_sam = alignment.tif_to_sam([x_tif, y_tif], cfg.scale)
                point_sam = np.asarray([[float(pos_sam[0]), float(pos_sam[1])]])
                point_pred = cw.sam_to_crop(point_sam) if cw is not None else point_sam
                prompts = Prompts(points_sam=point_pred, labels=np.asarray([1]))
                mask_hint = mask_to_low_res_logits(masks_by_z[neighbour_z])

                new_mask, _score, _logits = image_predict(
                    image_predictor, image, prompts, mask_input=mask_hint)

                px, py = float(point_pred[0, 0]), float(point_pred[0, 1])
                if new_mask.any() and predict_mod._point_in_mask(new_mask, px, py, contain_r):
                    cv2.imwrite(str(mask_paths[z]), (new_mask.astype("uint8") * 255))
                    masks_by_z[z] = new_mask
                    areas_by_z[z] = float(new_mask.sum())
                    outcomes[z] = "corrected"
                    corrected = True

        if not corrected:
            accepted = [zz for zz in areas_by_z if zz not in flagged]
            if accepted:
                nearest = min(accepted, key=lambda zz: abs(zz - z))
                fallback_mask = masks_by_z[nearest]
                cv2.imwrite(str(mask_paths[z]), (fallback_mask.astype("uint8") * 255))
                masks_by_z[z] = fallback_mask
                areas_by_z[z] = float(fallback_mask.sum())
                outcomes[z] = "guard_fallback"

    return outcomes
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_second_pass.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 7: Lint and run the full suite**

Run: `ruff check pipeline/propagate.py pipeline/config.py tests/test_second_pass.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 8: Commit**

```bash
git add pipeline/propagate.py pipeline/config.py tests/test_second_pass.py
git commit -m "feat(propagate): add apply_second_pass orchestrator and its config fields"
```

---

### Task 4: wire into `batch.py`

**Files:**
- Modify: `batch.py`

**Interfaces:**
- Consumes: `pipeline.propagate.apply_second_pass` (Task 3), `PipelineConfig.second_pass` /
  `second_pass_min_neighbour_area_ratio` (Task 3), `eval.merge_metric.score_chain`,
  `eval.merge_metric.nodes_by_z`, `eval.merge_metric.DEFAULT_RADIUS` (all already exist,
  `eval/merge_metric.py:108`, `:71`, `:21`).
- Produces: `batch.py --second-pass` / `--no-second-pass` and
  `--second-pass-min-neighbour-area-ratio` CLI flags. A new `second_pass` column in every touched
  chain's `qc.csv`.

- [ ] **Step 1: Confirm the current exact state matches this plan's assumptions**

Run `grep -n "^def _run_one_chain\|save_state(state, chain_dir" batch.py` and confirm `_run_one_chain`
is still structured as: build `state` via `_run_chain_once`/tier-2 rerun, then
`save_state(state, chain_dir / "state.json")`, then `return state` (currently lines 397-443). Run
`grep -n '"--postprocess"' batch.py` and confirm the `--postprocess`/`--no-postprocess` flag pair and
the `if args.postprocess is not None: pipe["postprocess_masks"] = args.postprocess` pattern still
exist (currently around lines 810-821). If either has changed structurally, STOP and report
NEEDS_CONTEXT describing what's different.

- [ ] **Step 2: Add the CLI flags**

In `batch.py`, add directly after the existing `--no-postprocess` argument (currently line 813):

```python
    ap.add_argument("--second-pass", dest="second_pass", action="store_true", default=None,
                    help="force the propagation second pass ON (overrides the preset)")
    ap.add_argument("--no-second-pass", dest="second_pass", action="store_false",
                    help="force the propagation second pass OFF (overrides the preset)")
    ap.add_argument("--second-pass-min-neighbour-area-ratio", type=float, default=None,
                    help="override the second pass's nucleus-capture neighbour area floor "
                         "(default 0.5, see PipelineConfig.second_pass_min_neighbour_area_ratio)")
```

Add directly after the existing `if args.postprocess is not None: pipe["postprocess_masks"] =
args.postprocess` block (currently lines 820-821):

```python
    if args.second_pass is not None:
        pipe["second_pass"] = args.second_pass
    if args.second_pass_min_neighbour_area_ratio is not None:
        pipe["second_pass_min_neighbour_area_ratio"] = args.second_pass_min_neighbour_area_ratio
```

- [ ] **Step 3: Add the qc.csv update helper**

Add this function in `batch.py`, directly above `_run_one_chain` (currently line 397):

```python
def _apply_second_pass_and_update_qc(session, cfg, neuron: str, chain: dict,
                                     chain_dir: Path, state) -> None:
    """Run the propagation second pass on a finished chain and fold its outcomes into
    qc.csv, so _triage.csv (rebuilt from qc.csv every run) reflects the corrected state
    instead of the stale pre-pass flags. No-op when second_pass is off, the chain used
    per-slice re-seeding (already has its own guard), or the chain has nothing flagged.
    """
    import pandas as pd

    from eval import merge_metric
    from pipeline.propagate import apply_second_pass
    from sam2_utils import alignment

    if not cfg.second_pass or cfg.per_slice_reseed:
        return

    qc_csv_path = chain_dir / "qc.csv"
    if not qc_csv_path.exists():
        return
    qc_df = pd.read_csv(qc_csv_path)
    if "queue" in qc_df.columns:
        has_flags = bool(qc_df["queue"].any())
    elif "intervene" in qc_df.columns:
        has_flags = bool(qc_df["intervene"].any())
    elif "flag" in qc_df.columns:
        has_flags = bool(qc_df["flag"].any())
    else:
        has_flags = False
    if not has_flags:
        return

    nodes_by_z = merge_metric.nodes_by_z(session.annotate_df, cfg.scale)
    records = merge_metric.score_chain(chain_dir, neuron, nodes_by_z, merge_metric.DEFAULT_RADIUS)

    cw = alignment.CropWindow.from_dict(state.crop_window) if state.crop_window else None
    outcomes = apply_second_pass(
        session.image_predictor, state.frames_dir, state.frame_to_z, cw, chain,
        session.annotate_df, chain_dir, records, cfg=cfg,
        min_neighbour_area_ratio=cfg.second_pass_min_neighbour_area_ratio)
    if not outcomes:
        return

    if "second_pass" not in qc_df.columns:
        qc_df["second_pass"] = ""
    for z, tag in outcomes.items():
        row = qc_df["z"] == z
        qc_df.loc[row, "second_pass"] = tag
        if tag == "guard_fallback":
            # mirrors apply_blowup_guard's own zero-confidence intent: no live
            # frame_conf/pred_iou dict exists post-hoc, so queue the frame for a
            # human directly via the same columns build_triage_queue reads.
            for col in ("flag", "intervene", "queue"):
                if col in qc_df.columns:
                    qc_df.loc[row, col] = True
    qc_df.to_csv(qc_csv_path, index=False)
    print(f"[batch] second pass {neuron}/{chain_dir.name}: "
          f"{len(outcomes)} frame(s), "
          f"{sum(1 for t in outcomes.values() if t == 'corrected')} corrected, "
          f"{sum(1 for t in outcomes.values() if t == 'guard_fallback')} guard_fallback")
```

- [ ] **Step 4: Call it from `_run_one_chain`**

In `batch.py`, change `_run_one_chain`'s body (currently ending at lines 442-443) from:

```python
    save_state(state, chain_dir / "state.json")
    return state
```

to:

```python
    _apply_second_pass_and_update_qc(session, cfg, neuron, chain, chain_dir, state)
    save_state(state, chain_dir / "state.json")
    return state
```

- [ ] **Step 5: Smoke-test the flag runs without error on a real chain**

This requires the F: drive (`ls "F:/ZhenLab/Data/output_masks/resolution_experiments/"` should list
directories; if not mounted, STOP and report BLOCKED). Pick one real chain directory from
`target_tier2_s1forced_neg_sam3_merged` that `_triage.csv` shows as flagged. Confirm
`_apply_second_pass_and_update_qc` runs against it without exception via a one-off `py -3 -c "..."`
that imports `batch` and calls the function directly with a hand-built minimal `state`/`chain`/`cfg`
(constructed the same way this task's own code builds them), or via a real (short, `--neuron-limit 1`)
`batch.py --second-pass` run against a small scope. This step only confirms the wiring itself runs
end-to-end without error; Task 5 is where the real before/after numbers get evaluated and reported.

- [ ] **Step 6: Run the full test suite**

Run: `py -3 -m pytest -q`
Expected: all tests pass. `ruff check batch.py` clean.

- [ ] **Step 7: Commit**

```bash
git add batch.py
git commit -m "feat(batch): wire the propagation second pass into _run_one_chain"
```

---

### Task 5: real verification against the existing propagation tree

**Files:** none (verification only, no code changes).

**Interfaces:**
- Consumes: `batch.py --second-pass` (Task 4), the real `target_tier2_s1forced_neg_sam3_merged`
  tree on F:.

- [ ] **Step 1: Confirm F: is mounted**

Run `ls "F:/ZhenLab/Data/output_masks/resolution_experiments/target_tier2_s1forced_neg_sam3_merged"`.
If it does not list neuron directories, STOP and report BLOCKED (F: is known to disconnect
physically; this is not fixable from code).

- [ ] **Step 2: Pick one real flagged chain and record its before-state**

From `target_tier2_s1forced_neg_sam3_merged/_triage.csv`, pick one chain with several flagged frames
(prefer one with a mix of dropout and foreign flags if available, for a meaningful test of both
paths). Record that chain's current `qc.csv` flagged rows and run
`py -3 -m eval.merge_metric --root <that chain's neuron/chain dir's PARENT tree root> --no-membrane
--neurons <that neuron>` (or read the existing `_merge_metric.csv` row for this chain if already
current) to get its `own_contained`/`n_foreign`/`empty` values per flagged z before any change.

- [ ] **Step 3: Copy the chain to a scratch location and run the second pass on the copy**

Do not mutate the real tree in place for this verification (F: is shared, other work reads it). Copy
the one chain directory (e.g. `<neuron>/chain_<NN>/`) to a scratch directory
(`C:\Users\User\AppData\Local\Temp\claude\...\scratchpad\second_pass_verify\`), then run a short
Python script (one-off, not committed) that loads the copied chain's `state.json` via
`pipeline.state.load_state`, builds `nodes_by_z`/`score_chain` records the same way
`_apply_second_pass_and_update_qc` does, and calls `apply_second_pass` directly against the copy,
using the real session's `image_predictor` (this needs GPU/model weights loaded, matching how
`experiments/nucleonet_spotcheck.py` and similar one-off scripts in this repo load a real predictor).

- [ ] **Step 4: Report the real before/after numbers honestly**

Re-score the copied, now-corrected chain with `eval.merge_metric.score_chain` and compare
`own_contained`/`n_foreign`/`empty` per touched frame against Step 2's before-state. Report plainly:
how many frames got `"corrected"` vs `"guard_fallback"`, and whether the corrected frames' new
records actually look better (own_contained flips to True, n_foreign drops to 0, empty flips to
False) or not. This is a single real chain, first signal not verdict, matching this project's
established practice (see the SAM3 bake-off's own single-chain-then-long-chain confirmation
pattern); do not generalize a one-chain result into a strong claim in Task 6's docs beyond what it
actually shows.

- [ ] **Step 5: Record the result for Task 6**

No commit needed for this task (verification only). Write the real chain id, the before/after
records, and the corrected/guard_fallback counts into your report, since Task 6 (docs) depends on
this exact evidence, not a paraphrase.

---

### Task 6: docs

**Files:**
- Modify: `docs/explanation/roadmap.md`
- Modify: `docs/CHANGELOG.md`

**Interfaces:**
- Consumes: Task 5's real verification result.

- [ ] **Step 1: Read what actually happened**

Read Task 5's report for the real chain id, before/after records, and corrected/guard_fallback
counts. Do not write generic placeholder prose.

- [ ] **Step 2: Update the roadmap**

In `docs/explanation/roadmap.md`, add a new dated entry under section 5b (the immediate queue,
following the numbered-item convention items 9-12 already use) covering: what was built (the three
functions, the `mask_input`/`mask_to_low_res_logits` addition, the two new config fields, the
`batch.py --second-pass` flag), the import-direction constraint that shaped where `score_chain` gets
called from, the nucleus-capture neighbour guard and its stated limits (from the design spec, carry
these forward accurately, do not soften them), and Task 5's real one-chain verification result
stated as a first signal, not a verdict. Cite `docs/superpowers/specs/2026-07-30-propagation-second-pass-design.md`
for the full design rationale.

- [ ] **Step 3: Add a CHANGELOG entry**

Follow the existing entries' format (anchor id, `##` heading, prose, `---` separator,
most-recent-first in both the file body and the Contents list). Cover the same ground as the roadmap
entry, more concisely, matching the existing entries' length and tone.

- [ ] **Step 4: Humanize and check for em dashes**

Run the `humanizer` skill on everything written in Steps 2-3 before committing (required by this
project's CLAUDE.md, not optional). Then:

Run: `grep -n $'\xe2\x80\x94\|\xe2\x80\x93' docs/explanation/roadmap.md docs/CHANGELOG.md`
Expected: no output.

- [ ] **Step 5: Run the full suite one last time**

Run: `py -3 -m pytest -q`
Expected: all tests pass (this task shouldn't touch code, but confirm nothing else drifted).

- [ ] **Step 6: Commit**

```bash
git add docs/explanation/roadmap.md docs/CHANGELOG.md
git commit -m "docs(roadmap): record the propagation second-pass landing"
```
