# Temporal membrane projection implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress organelles in the scale-8 membrane map by projecting a small window of
adjacent z-slices into one intensity image before running the existing Sato ridge filter, and
measure whether that moves the ~40% bleed-per-fill floor `dense_membrane_fill.py` found.

**Architecture:** Two new pure array functions in `sam2_utils/membrane.py`
(`register_crops`, `project_crops`), unit-tested in isolation. `experiments/dense_membrane_fill.py`
gains the orchestration: load a window of whole frames once per run, slice+register+project each
neuron's own crop from that window, then hand the result to the existing, unmodified
`membrane_map()`. A `--sweep-temporal` mode grids window/combine settings and reads the gate.

**Tech Stack:** numpy, scipy.ndimage (already a dependency), skimage.registration (skimage is
already a dependency via `membrane_map`'s `skimage.filters.sato`). No torch. CPU only.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (`CLAUDE.md`).
- Run the `humanizer` skill on any prose (docstrings read as prose, commit messages) before
  committing.
- New pure-logic tests stay torch-free and CPU-only; run via `py -3 -m pytest`.
- Lint with `ruff check .`, touching only the files this plan edits.
- The library (`sam2_utils/`) must never import the drivers or `eval/`; `register_crops` and
  `project_crops` take already-loaded arrays only, no frame-store or pipeline imports.
- Commit incrementally, one concern per commit.
- Integer-pixel registration only, no subpixel interpolation (spec's deliberate exclusion: avoids
  blurring membrane sharpness).

---

### Task 1: `register_crops` and `project_crops` in `sam2_utils/membrane.py`

**Files:**
- Modify: `sam2_utils/membrane.py`
- Test: `tests/test_membrane_temporal.py` (new)

**Interfaces:**
- Produces: `register_crops(crops: list[np.ndarray], *, max_shift: int = 5) -> list[np.ndarray]`
- Produces: `project_crops(crops: list[np.ndarray], *, combine: str = "median") -> np.ndarray`
- Consumes: nothing from other tasks (this task has no dependencies).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_membrane_temporal.py`:

```python
import numpy as np
import pytest
from sam2_utils import membrane as mb


def test_register_crops_single_element_passthrough():
    crop = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = mb.register_crops([crop])
    assert len(out) == 1
    assert np.array_equal(out[0], crop)


def test_register_crops_aligns_to_center():
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    shifted = np.zeros((24, 24), dtype=np.float32)
    shifted[6:10, 13:17] = 200.0  # same block, moved relative to ref
    aligned_list = mb.register_crops([shifted, ref])  # center = index 1 (ref)
    aligned = aligned_list[0]
    ref_peak = np.unravel_index(np.argmax(ref), ref.shape)
    aligned_peak = np.unravel_index(np.argmax(aligned), aligned.shape)
    assert abs(ref_peak[0] - aligned_peak[0]) <= 1
    assert abs(ref_peak[1] - aligned_peak[1]) <= 1


def test_register_crops_clamps_large_shift():
    ref = np.zeros((24, 24), dtype=np.float32)
    ref[10:14, 10:14] = 200.0
    shifted = np.zeros((24, 24), dtype=np.float32)
    shifted[0:4, 0:4] = 200.0  # far shift, larger than max_shift
    aligned_list = mb.register_crops([shifted, ref], max_shift=2)
    aligned = aligned_list[0]
    ref_peak = np.array(np.unravel_index(np.argmax(ref), ref.shape))
    aligned_peak = np.array(np.unravel_index(np.argmax(aligned), aligned.shape))
    assert np.any(np.abs(ref_peak - aligned_peak) > 2)  # clamp prevented full correction


def test_project_crops_median_suppresses_transient_dark():
    bright = np.full((6, 6), 200.0, dtype=np.float32)
    dark = bright.copy()
    dark[3, 3] = 10.0  # transient dark pixel, present in 1 of 3 crops
    crops = [bright.copy(), dark, bright.copy()]
    proj = mb.project_crops(crops, combine="median")
    assert proj[3, 3] > 150.0  # pulled back toward bright, not stuck dark


def test_project_crops_persistent_dark_stays_dark():
    dark_val = 10.0
    crops = [np.full((6, 6), dark_val, dtype=np.float32) for _ in range(3)]
    proj = mb.project_crops(crops, combine="median")
    assert np.allclose(proj, dark_val)


def test_project_crops_mean_partially_suppresses_less_than_median():
    bright = np.full((6, 6), 200.0, dtype=np.float32)
    dark = bright.copy()
    dark[3, 3] = 10.0
    crops = [bright.copy(), dark, bright.copy()]
    med = mb.project_crops(crops, combine="median")
    mean = mb.project_crops(crops, combine="mean")
    assert mean[3, 3] < med[3, 3]  # mean pulls less far back toward bright than median


def test_project_crops_single_element_identity():
    crop = np.arange(16, dtype=np.float32).reshape(4, 4)
    for combine in ("median", "mean", "max", "min"):
        assert np.array_equal(mb.project_crops([crop], combine=combine), crop)


def test_project_crops_unknown_combine_raises():
    with pytest.raises(ValueError):
        mb.project_crops([np.zeros((4, 4), np.float32)], combine="bogus")


def test_temporal_projection_then_membrane_map_suppresses_blob_response():
    def make_slice(with_blob):
        p = np.full((30, 30), 200.0, dtype=np.float32)
        p[:, 14:16] = 20.0  # persistent vertical ridge (a membrane)
        if with_blob:
            p[20:24, 20:24] = 15.0  # transient dark blob (an organelle), center slice only
        return p

    crops = [make_slice(False), make_slice(True), make_slice(False)]
    registered = mb.register_crops(crops)
    projected = mb.project_crops(registered, combine="median")

    mem_single = mb.membrane_map(crops[1])  # single center slice, blob included
    mem_temporal = mb.membrane_map(projected)  # temporal-projected

    blob_region = np.s_[20:24, 20:24]
    ridge_region = np.s_[:, 14:16]
    assert mem_temporal[blob_region].mean() < mem_single[blob_region].mean()
    assert mem_temporal[ridge_region].mean() > 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_membrane_temporal.py -v`
Expected: FAIL, `AttributeError: module 'sam2_utils.membrane' has no attribute 'register_crops'`
(and similarly for `project_crops`).

- [ ] **Step 3: Implement `register_crops` and `project_crops`**

In `sam2_utils/membrane.py`, add after `membrane_map` (imports `ndi` and `np` are already at the
top of the file):

```python
def register_crops(crops: list[np.ndarray], *, max_shift: int = 5) -> list[np.ndarray]:
    """Align every crop in `crops` to the center crop by an integer-pixel translation.

    Uses phase correlation at pixel precision only, no subpixel interpolation, so no
    blur is introduced by the alignment itself. Each estimated shift is clamped to
    +/- max_shift px per axis before being applied, a safety valve against a
    low-texture crop returning a wild or ambiguous shift. Returns a new list, same
    length and shape as the input; the center crop (index len(crops) // 2) is
    returned unchanged, everything else is float32."""
    from skimage.registration import phase_cross_correlation

    n = len(crops)
    if n <= 1:
        return list(crops)
    center_i = n // 2
    ref = crops[center_i].astype(np.float32)
    out = list(crops)
    for i, crop in enumerate(crops):
        if i == center_i:
            continue
        moving = crop.astype(np.float32)
        shift, _error, _diffphase = phase_cross_correlation(ref, moving, upsample_factor=1)
        shift = np.clip(np.round(shift), -max_shift, max_shift)
        out[i] = ndi.shift(moving, shift, order=0, mode="nearest")
    return out


_COMBINERS = {
    "median": lambda stack: np.median(stack, axis=0),
    "mean": lambda stack: np.mean(stack, axis=0),
    "max": lambda stack: np.max(stack, axis=0),
    "min": lambda stack: np.min(stack, axis=0),
}


def project_crops(crops: list[np.ndarray], *, combine: str = "median") -> np.ndarray:
    """Reduce an already-registered window of crops to one projected image.

    `combine` selects the per-pixel reducer across the window: median (default) pulls
    a pixel dark in a minority of slices, a transient organelle, toward the brighter
    majority value, while a pixel dark in most or all slices, a persistent membrane,
    stays dark; mean, max, and min are also available for the same call so a sweep can
    compare them. A single-element `crops` returns that element unchanged for every
    combiner, the natural window=0 fallback."""
    if combine not in _COMBINERS:
        raise ValueError(f"unknown combine {combine!r}, choose one of {sorted(_COMBINERS)}")
    stack = np.stack([np.asarray(c, dtype=np.float32) for c in crops], axis=0)
    return _COMBINERS[combine](stack).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_membrane_temporal.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check sam2_utils/membrane.py tests/test_membrane_temporal.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass (no regressions in the existing 279).

- [ ] **Step 6: Commit**

```bash
git add sam2_utils/membrane.py tests/test_membrane_temporal.py
git commit -m "feat(membrane): add register_crops and project_crops for temporal projection"
```

---

### Task 2: wire the temporal window into `dense_membrane_fill.py`

**Files:**
- Modify: `experiments/dense_membrane_fill.py`

**Interfaces:**
- Consumes: `mb.register_crops(crops, *, max_shift=5)`, `mb.project_crops(crops, *, combine="median")`
  from Task 1.
- Produces: `grow_all(nmasks, frames, center_z, window, combine, pad, nodes, radius)` (replaces the
  current `grow_all(nmasks, em_gray, pad, nodes, radius)`), and CLI flags `--mm-window`,
  `--mm-combine`. Later tasks in this plan call this new `grow_all` signature.

- [ ] **Step 1: Capture today's baseline output before editing**

Run: `py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 > /tmp/dmf_baseline.txt 2>&1`
(or the Windows-shell equivalent redirect). Keep this file; Step 4 diffs against it. This is the
regression check for `--mm-window 0`, which must reproduce today's numbers exactly.

- [ ] **Step 2: Change `grow_all`'s signature and body**

In `experiments/dense_membrane_fill.py`, replace the current `grow_all` function:

```python
def grow_all(nmasks: dict, frames: dict, center_z: int, window: int, combine: str,
            pad: int, nodes, radius: int):
    """Grow each neuron to its ridge walls once, UNCAPPED, inside a local bbox+pad window.

    The membrane map for each neuron's crop is built from `frames`, a {z: em_gray} dict
    covering at least [center_z - window, center_z + window] (missing z, e.g. near the
    stack's edge, are simply absent and drop out of the window). window=0 uses only
    frames[center_z], reproducing the original single-slice path exactly: register_crops
    and project_crops are no-ops on a length-1 list.

    Returns {neuron: rec} with the raw and grown full-frame masks, their areas, underfill,
    the raw-mask centroid seed, and the foreign-node count each mask engulfs (raw vs grown).
    Growing uncapped lets a cap be applied afterwards as a cheap post-filter, so a cap sweep
    needs only this one grow pass."""
    H, W = frames[center_z].shape[:2]
    recs = {}
    for n, full in nmasks.items():
        ys, xs = np.where(full)
        if ys.size == 0:
            continue
        y1, y2 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        x1, x2 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        win = full[y1:y2, x1:x2]
        zs = [z for z in range(center_z - window, center_z + window + 1) if z in frames]
        crops = [frames[z][y1:y2, x1:x2] for z in zs]
        if len(crops) > 1:
            crops = mb.register_crops(crops)
        em_crop = mb.project_crops(crops, combine=combine)
        mem = mb.membrane_map(em_crop)
        grown, _capped = grow_to_membrane(win, mem, cap=1e9)  # no clamp; cap applied later
        g_full = np.zeros((H, W), bool)
        g_full[y1:y2, x1:x2] = grown
        recs[n] = {
            "raw": full, "grown": g_full,
            "raw_area": int(win.sum()), "grown_area": int(grown.sum()),
            "raw_uf": float(mb.underfill_fraction(win, mem)),
            "grown_uf": float(mb.underfill_fraction(grown, mem)),
            "seed": (float(xs.mean()), float(ys.mean())),  # raw-centroid, trusted core
            "raw_foreign": foreign_count(full, nodes, n, radius),
            "grown_foreign": foreign_count(g_full, nodes, n, radius),
        }
    return recs
```

- [ ] **Step 3: Update `main()`: build the frames window, add the CLI flags, update the call site**

In `main()`, add the two new arguments next to the existing ones:

```python
    ap.add_argument("--mm-window", type=int, default=0,
                    help="temporal projection window radius in z-slices (0 = single-slice, "
                         "today's behaviour)")
    ap.add_argument("--mm-combine", choices=["median", "mean", "max", "min"], default="median",
                    help="combine statistic across the window (only used when --mm-window > 0)")
```

Replace the frame-loading block:

```python
    em, _ = pipeline.load_frame_sam(args.z, scale=SCALE)
    h8, w8 = em.shape[:2]
    em_gray = em.mean(axis=2) if em.ndim == 3 else em
```

with:

```python
    em, _ = pipeline.load_frame_sam(args.z, scale=SCALE)
    h8, w8 = em.shape[:2]
    em_gray = em.mean(axis=2) if em.ndim == 3 else em
    frames = {args.z: em_gray}
    for dz in range(1, args.mm_window + 1):
        for zz in (args.z - dz, args.z + dz):
            try:
                fz, _ = pipeline.load_frame_sam(zz, scale=SCALE)
            except Exception:
                continue
            frames[zz] = fz.mean(axis=2) if fz.ndim == 3 else fz
```

Update the `grow_all` call site:

```python
    recs = grow_all(nmasks, frames, args.z, args.mm_window, args.mm_combine, args.pad,
                    nodes, DEFAULT_RADIUS)
```

- [ ] **Step 4: Verify the window=0 regression**

Run: `py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 --mm-window 0 > /tmp/dmf_after.txt 2>&1`
Diff `/tmp/dmf_baseline.txt` against `/tmp/dmf_after.txt`: the underfill, area%, foreign, bleed_cells,
and new_bleed numbers must match exactly (timing seconds may differ, that's fine). If they differ,
stop and debug before continuing, `--mm-window 0` must be byte-identical to the old path.

- [ ] **Step 5: Smoke-test `--mm-window 1`**

Run: `py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 --mm-window 1 --mm-combine median`
Expected: runs to completion, prints a summary line (numbers may differ from the window=0 baseline,
that's expected and is exactly what Task 3's sweep will characterize).

- [ ] **Step 6: Run the full test suite (no regressions)**

Run: `py -3 -m pytest -q`
Expected: all tests pass. `ruff check experiments/dense_membrane_fill.py` clean.

- [ ] **Step 7: Commit**

```bash
git add experiments/dense_membrane_fill.py
git commit -m "feat(experiments): wire the temporal projection window into dense_membrane_fill"
```

---

### Task 3: `--sweep-temporal` and the gate run

**Files:**
- Modify: `experiments/dense_membrane_fill.py`
- Modify: `docs/explanation/roadmap.md` (record the gate result)
- Modify: `docs/CHANGELOG.md` (record what landed)

**Interfaces:**
- Consumes: `grow_all` (Task 2's signature), `apply_cap`, `contested_px` (both unchanged, already
  in this file).

- [ ] **Step 1: Add `--sweep-temporal`, and note it in the module docstring**

Update the module docstring's usage examples (top of `experiments/dense_membrane_fill.py`) to add:

```
    py -3 experiments/dense_membrane_fill.py --mm-window 1 --mm-combine median
    py -3 experiments/dense_membrane_fill.py --sweep-temporal    # grid window/combine vs baseline
```

This is the only docs update this task needs at the code level (per the spec: not yet wired into
production scoring, so `docs/reference/configuration.md` stays untouched, a docstring note is
enough).

Add the CLI flag next to `--sweep`:

```python
    ap.add_argument("--sweep-temporal", action="store_true",
                    help="grid a few (window, combine) settings and print the same "
                         "bleed/underfill table as --sweep")
```

After the existing `if args.sweep:` block in `main()`, add:

```python
    if args.sweep_temporal:
        max_w = 2
        frames_wide = dict(frames)
        for dz in range(args.mm_window + 1, max_w + 1):
            for zz in (args.z - dz, args.z + dz):
                if zz in frames_wide:
                    continue
                try:
                    fz, _ = pipeline.load_frame_sam(zz, scale=SCALE)
                except Exception:
                    continue
                frames_wide[zz] = fz.mean(axis=2) if fz.ndim == 3 else fz
        print("\n[sweep temporal]  window  combine   mean_uf   area%   foreign  "
              "bleed_cells  new_bleed  contested")
        for w in (0, 1, 2):
            combos = ("median",) if w == 0 else ("median", "mean", "max")
            for combine in combos:
                recs_t = grow_all(nmasks, frames_wide, args.z, w, combine, args.pad,
                                  nodes, DEFAULT_RADIUS)
                ch, m = apply_cap(recs_t, args.cap, args.uf_min)
                cont = contested_px(ch, (h8, w8))
                print(f"[sweept]  {w:>4}  {combine:>8}  {m['uf']:6.3f}  "
                      f"{100*(m['a_after']-m['a_before'])/max(1,m['a_before']):+5.0f}%  "
                      f"{m['foreign']:>6}   {m['bleed_cells']:>3}/{len(recs_t)}     "
                      f"{m['new_bleed']:>4}     {cont:>7}")
        print()
```

- [ ] **Step 2: Run the full test suite and lint (no regressions)**

Run: `py -3 -m pytest -q` and `ruff check experiments/dense_membrane_fill.py`.
Expected: clean.

- [ ] **Step 3: Commit the code change**

```bash
git add experiments/dense_membrane_fill.py
git commit -m "feat(experiments): add --sweep-temporal to grid window/combine against the baseline"
```

- [ ] **Step 4: Run the gate**

Run: `py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 --sweep-temporal`
Capture the full printed table (the `window=0 combine=median` row is the baseline, identical to
Task 2 Step 4's regression numbers).

- [ ] **Step 5: Read the gate and record the result**

Compare every non-baseline row's `foreign` and `bleed_cells` against the `window=0` row, holding
`mean_uf` and `area%` in view (a setting that also blows up area or underfill in the wrong direction
is not a free win). Two possible outcomes, both are a valid landing, the spec never assumed which:

- **The floor drops**: some `(window, combine)` row shows a materially lower `foreign`/`bleed_cells`
  count than the baseline without a comparable increase in area or underfill regressing the wrong
  way. Note which setting, it becomes the candidate for the deferred `MembraneSource` wiring spec.
- **The floor does not move**: every row is within noise of the baseline. Record that the temporal
  lever alone did not clear item 2b.5's gate, and that the deferred intensity/texture blob filter
  (out of scope for this plan) is now the next thing to try, informed by this negative result rather
  than starting cold.

Update `docs/explanation/roadmap.md`'s item 2b.5 (§5 and the §5b queue entry) with whichever outcome
actually happened, and add a `docs/CHANGELOG.md` entry summarizing the experiment and the numbers.
Run the `humanizer` skill on both before committing (per `CLAUDE.md`).

- [ ] **Step 6: Commit the docs**

```bash
git add docs/explanation/roadmap.md docs/CHANGELOG.md
git commit -m "docs(roadmap): record the temporal-projection sweep result against item 2b.5's gate"
```
