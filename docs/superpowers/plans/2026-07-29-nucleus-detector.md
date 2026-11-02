# Nucleus-capture detector implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a nucleus-capture detector for the roadmap's nested-membrane ceiling (problem 7):
spot-check the real, verified NucleoNet model first, build a classical intensity/texture fallback
only if it doesn't generalize to C. elegans EM, and add the one-line GUI labeling workflow that lets
the ground-truth set this detector needs start accumulating.

**Architecture:** Task 1 is an independent, mechanical labeling-workflow change. Task 2 is a
characterization spike (install a new external package, discover its real API, run it once on real
data, render the result) whose outcome is a genuine gate: NucleoNet either generalizes or it doesn't,
nobody knows yet. Task 3 (the classical fallback) only runs if Task 2's gate says so. Task 4 documents
whichever path was actually taken.

**Tech Stack:** numpy, scipy.ndimage (existing deps). Task 2 adds `empanada-dl` (torch-based,
scoped to `experiments/` only, lazy-imported). Task 3, if it runs, is pure numpy/scipy, torch-free.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (`CLAUDE.md`).
- Run the `humanizer` skill on any prose you are about to commit before committing.
- New pure-logic tests stay torch-free and CPU-only; `py -3 -m pytest`.
- Lint with `ruff check .`, touching only files you edit.
- The library (`sam2_utils/`) must never import the drivers or `eval/`.
- No shape/roundness term anywhere in nucleus detection: real neurites can be round too (the
  2026-07-28 visual verdict this whole spec is built around).
- `empanada-dl` and any torch/GPU-heavy inference stay inside `experiments/`, not `sam2_utils/`, not
  added to the CPU-only test suite's dependency footprint, unless a *later*, separate spec
  deliberately promotes it.
- Commit incrementally, one concern per commit.

---

### Task 1: add the `"nucleus"` GUI labeling category

**Files:**
- Modify: `sam2_utils/labels.py:72`
- Test: `tests/test_labels.py` (check if this file exists first; if it does, add to it; if not, this
  task's verification is the existing test suite plus a manual GUI smoke check, see Step 3)

**Interfaces:**
- Produces: `sam2_utils.labels.ERROR_TYPES` gains `"nucleus"` as a valid value. No other task depends
  on this one; it is independent and can run in any order relative to Tasks 2-4.

- [ ] **Step 1: Check for an existing labels test file**

Run: `ls tests/test_labels.py 2>&1 || echo "no such file"`

If it exists, read it to match its existing style for the next step. If not, skip straight to Step 2
and rely on Step 4's manual check instead of a new automated test (a one-tuple-value addition with an
existing `if error_type and error_type not in ERROR_TYPES: raise ValueError` guard already covers the
behavior; a dedicated test only makes sense if the file already exists and this is the natural place
to extend it).

- [ ] **Step 2: Add the new error type**

In `sam2_utils/labels.py`, change:

```python
ERROR_TYPES = ("wrong_object", "under", "over", "bleed", "fragmented", "missing", "other")
```

to:

```python
ERROR_TYPES = ("wrong_object", "under", "over", "bleed", "fragmented", "missing", "nucleus", "other")
```

Keep `"other"` last (matches the existing convention of a catch-all trailing the specific
categories).

- [ ] **Step 3: Verify the GUI dropdown picks it up**

Run: `py -3 -c "from sam2_utils import labels; print(labels.ERROR_TYPES); assert 'nucleus' in labels.ERROR_TYPES"`
Expected: prints the tuple including `'nucleus'`, no assertion error.

This confirms the GUI's `ComboBox(label="error type", choices=list(labels_mod.ERROR_TYPES), ...)` at
`gui.py:1127` will show it, since it reads the tuple directly with no other dispatch keyed to specific
values (verified during design: `grep -rn "ERROR_TYPES" --include=*.py .` shows exactly one producer,
`sam2_utils/labels.py`, and one consumer, `gui.py`'s `ComboBox`).

- [ ] **Step 4: Run the full test suite**

Run: `py -3 -m pytest -q`
Expected: all existing tests still pass (this change only adds a tuple value; nothing that validated
against the old tuple's exact contents should exist, but confirm).
Run: `ruff check sam2_utils/labels.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/labels.py
git commit -m "feat(labels): add a nucleus error type for the nucleus-capture labeling workflow"
```

---

### Task 2: NucleoNet spot-check (characterization spike, real API discovery required)

**Files:**
- Create: `experiments/nucleonet_spotcheck.py`

**Interfaces:**
- Consumes: `pipeline.load_frame_sam(z, scale=8)` (existing), `experiments.dense_overlay`'s
  `build_palette`, `colorize_over_em`, `build_index`, `neuron_masks_at_z`, `stack_labelmap` (existing,
  already used identically by `experiments/dense_membrane_fill.py` and `experiments/dense_video.py`,
  read either file for the exact call pattern before writing this task's rendering code).
- Produces: a rendered comparison image, and, most importantly, **the answer to the gate question**:
  does NucleoNet find plausible C. elegans nuclei on real target-worm EM. This is read by a human
  (the controller/you), not consumed by later code, so there is no function signature contract to
  later tasks. Task 3 is conditional on this task's outcome, not on any function it defines.

**This is a spike, not a TDD task.** `empanada-dl`'s exact Python API was not verified line-by-line
during design (only its existence, license, and headless-capability claim were verified via web
search and the package's README/PyPI page). You must discover the real API yourself before writing
inference code, the same way `experiments/sam3_probe.py` characterized the SAM3 HF API before any
adapter was built for it. Do not guess at function names or fabricate a plausible-looking call that
you have not actually run.

- [ ] **Step 1: Install and inspect**

Run: `py -3 -m pip install empanada-dl`
Expected: installs successfully (it pulls in torch if not already present; this repo's `sam2env`
already has torch for SAM2/SAM3, so this should mostly reuse the existing install).

Then discover the real headless inference API. Try, in order, until one gives you a usable answer:

```
py -3 -c "import empanada; help(empanada)"
py -3 -c "import empanada; print([x for x in dir(empanada) if not x.startswith('_')])"
```

Also check the installed package's own source for an inference entry point (e.g.
`py -3 -c "import empanada, os; print(os.path.dirname(empanada.__file__))"` then browse that
directory for scripts or modules with "infer" or "predict" in the name), and the project's docs at
`https://empanada.readthedocs.io/en/latest/` and `https://github.com/volume-em/empanada` (the base,
non-napari package, not `empanada-napari`) for a documented headless usage example.

You are specifically looking for: how to load the pretrained NucleoNet model (a model name/zoo
identifier, or a checkpoint URL/path to download), and how to run 2D instance segmentation inference
on a single in-memory image array (not a file-based napari workflow), getting back an instance label
map or equivalent.

- [ ] **Step 2: If the API cannot be found or NucleoNet's weights are not obtainable headlessly**

Report status BLOCKED with exactly what you tried and what's missing (e.g. "no headless model-zoo
entry point found, only the napari GUI exposes model selection" or "found the API but NucleoNet
weights require an interactive download step"). Do not fall back to writing speculative code that
calls a function you have not confirmed exists. This is a legitimate, useful outcome, the gate can
also be answered as "NucleoNet is not practically usable headlessly here," which routes to Task 3
the same as a poor-quality detection result would.

- [ ] **Step 3: Write the spot-check script**

Once you have a confirmed, working way to run NucleoNet inference on a single 2D image array, write
`experiments/nucleonet_spotcheck.py`. Structure it like this (fill in the real inference call from
Step 1's findings; everything else below is exact):

```python
"""Spot-check: does NucleoNet find plausible C. elegans nuclei on target-worm EM?

A characterization spike, not production code (matches experiments/sam3_probe.py's role for the
SAM3 adapters). Lazy-imports torch/empanada so nothing outside this script gains a new heavy
dependency. No ground truth exists yet (that's what the new "nucleus" GUI error type, Task 1, starts
collecting), so this is a visual gut-check: render NucleoNet's detections over the raw EM and read
them by eye, the same way Stage 0.1's registration overlay was judged before a formal metric existed.

    py -3 experiments/nucleonet_spotcheck.py                # z=1456, the current-work frame
    py -3 experiments/nucleonet_spotcheck.py --z 1472
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from experiments import dense_overlay as do

INDEX_CACHE = Path("docs/figures/sam3-bakeoff/dense-overlay/_index.json")
OUT_DIR = Path("docs/figures/presentation/nucleonet-spotcheck")
SCALE = 8


def run_nucleonet(em_gray: np.ndarray) -> np.ndarray:
    """Run NucleoNet on a single grayscale EM image, return an instance label map,
    same H x W as the input, 0 = background, 1..N = detected nucleus instances.

    Lazy-imports torch/empanada so this stays out of the CPU-only test path."""
    # FILL IN: the real, confirmed-working call from Step 1's API discovery.
    raise NotImplementedError("fill in from the confirmed empanada-dl API")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--index", default=str(INDEX_CACHE))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    em, _full_hw = pipeline.load_frame_sam(args.z, scale=SCALE)
    em_gray = em.mean(axis=2) if em.ndim == 3 else em

    print(f"[nucleonet] z={args.z}: running NucleoNet inference ...")
    nuc_labels = run_nucleonet(em_gray.astype(np.float32))
    n_detected = int(nuc_labels.max())
    print(f"[nucleonet] z={args.z}: {n_detected} nucleus instances detected")

    lut = do.build_palette(max(n_detected, 1))
    nuc_rgb = do.colorize_over_em(em, nuc_labels, lut, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.8))
    em_rgb = np.stack([em_gray.astype(np.uint8)] * 3, axis=2) if em.ndim != 3 else em
    ax[0].imshow(em_rgb); ax[0].set_title(f"raw EM (z={args.z})", fontsize=11)
    ax[1].imshow(nuc_rgb); ax[1].set_title(f"NucleoNet detections ({n_detected} instances)", fontsize=11)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("NucleoNet spot-check on target-worm EM", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out_dir / f"nucleonet_spotcheck_z{args.z}.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[nucleonet] wrote {f}")


if __name__ == "__main__":
    main()
```

Adjust the `run_nucleonet` body and any imports to match what Step 1 actually found; the rest of the
script (argument parsing, frame loading, rendering) is exact and should not need changes given the
existing `dense_overlay` helpers it reuses.

- [ ] **Step 4: Run it on the current-work frame**

Run: `py -3 experiments/nucleonet_spotcheck.py --z 1456`
Expected: writes `docs/figures/presentation/nucleonet-spotcheck/nucleonet_spotcheck_z1456.png`, prints
the detected instance count. If it errors, that is real signal about headless feasibility, capture the
full error in your report rather than debugging around it indefinitely, if you're stuck after one or
two reasonable fix attempts, report BLOCKED with the error.

If Step 4 succeeds, also try one or two more frames if you know of specific ones with a
previously-noted suspected nucleus-capture case (check `docs/figures/presentation/dense-membrane-fill/`
and any GUI-review notes you can find for a specific z/neuron already flagged); if none are readily at
hand, z=1456 alone is sufficient for this round's read.

- [ ] **Step 5: Look at the rendered image yourself and answer the gate**

This is the actual deliverable. Read the PNG(s) `experiments/nucleonet_spotcheck.py` wrote (the image
tool that reads files can open a PNG directly). Compare the "NucleoNet detections" panel against the
raw EM panel: do the detected instances land on round, membrane-bound, visually nucleus-like
structures on this C. elegans tissue, or does NucleoNet miss obvious candidates, fire on the wrong
structures, or produce nothing coherent?

Write your honest read into the report (Step 6). Do not soften a poor result or oversell a marginal
one; this exact judgment call is what Task 3 is conditioned on.

- [ ] **Step 6: Commit and report**

```bash
git add experiments/nucleonet_spotcheck.py
git commit -m "feat(experiments): spot-check NucleoNet on target-worm EM"
```

The rendered PNG(s) are figures, follow this repo's existing convention (check whether
`docs/figures/presentation/` is gitignored/untracked for other similar figures, e.g.
`dense-membrane-fill/`, before deciding whether to `git add` the PNG itself; if untracked figures are
the norm here, leave it untracked and just reference its path in your report).

In your report, state clearly: **GATE VERDICT: NucleoNet generalizes** (skip Task 3) or **GATE
VERDICT: NucleoNet does not generalize** (do Task 3), or **GATE VERDICT: BLOCKED, could not evaluate**
(if Step 2's blocker applies; also skip to Task 3, since the classical fallback becomes the only
remaining path). Include the image path(s) and your specific visual observations (what you saw,
structure by structure if there are only a few candidates, not just a one-word verdict) so the
controller can independently check your read before accepting the gate call.

---

### Task 3: classical fallback detector (ONLY IF Task 2's gate requires it)

**Skip this task entirely if Task 2's report says "GATE VERDICT: NucleoNet generalizes."** Only
execute it if the verdict was "does not generalize" or "BLOCKED, could not evaluate."

**Files:**
- Create: `sam2_utils/nucleus.py`
- Test: `tests/test_nucleus.py`

**Interfaces:**
- Consumes: `sam2_utils.membrane.membrane_map(em_patch, *, sigmas=...)` (existing, unchanged),
  `sam2_utils.membrane._perimeter(mask)` (existing private helper; import it directly, e.g.
  `from sam2_utils.membrane import _perimeter`, since it is exactly the boundary-extraction logic
  this task needs and duplicating it would violate DRY).
- Produces: `nucleus_dark_blob(mask, em_patch, *, dark_percentile=15, uniformity_max=25.0) -> tuple[bool, dict]`,
  `nucleus_thick_loop(mask, mem_thin, mem_coarse, *, tau=0.5, thick_ratio=0.6) -> tuple[bool, dict]`,
  `nucleus_candidate(mask, em_patch, *, sigmas_thin=(1,2,3), sigmas_coarse=(3,5,7), **kwargs) -> dict`.
  No other task in this plan consumes these; Task 4 only reports on this task's existence and test
  results, not its internals.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nucleus.py`:

```python
import numpy as np
from sam2_utils import nucleus as nuc


def _rect_mask(h=30, w=30, y0=5, y1=25, x0=5, x1=25):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_dark_blob_fires_on_uniform_dark_interior():
    mask = _rect_mask()
    em = np.full((30, 30), 200.0, dtype=np.float32)
    em[5:25, 5:25] = 30.0  # uniformly dark inside the mask
    fired, info = nuc.nucleus_dark_blob(mask, em)
    assert fired is True
    assert info["mean_intensity"] < 100.0
    assert info["uniformity"] < 25.0


def test_dark_blob_does_not_fire_on_patchy_interior_at_same_mean():
    mask = _rect_mask()
    em = np.full((30, 30), 200.0, dtype=np.float32)
    # same mean darkness as the fired case (30.0), but alternating stripes (0.0, 60.0)
    # instead of uniform: this isolates the uniformity term as the deciding factor,
    # should look like textured cytoplasm, not a nucleus
    interior = em[5:25, 5:25]
    interior[::2, :] = 0.0
    interior[1::2, :] = 60.0
    fired, info = nuc.nucleus_dark_blob(mask, em)
    assert fired is False
    assert info["uniformity"] >= 25.0


def test_dark_blob_does_not_fire_on_bright_interior():
    mask = _rect_mask()
    em = np.full((30, 30), 200.0, dtype=np.float32)
    fired, info = nuc.nucleus_dark_blob(mask, em)
    assert fired is False


def test_dark_blob_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    em = np.full((30, 30), 30.0, dtype=np.float32)
    fired, info = nuc.nucleus_dark_blob(mask, em)
    assert fired is False


def test_thick_loop_fires_on_wide_ridge():
    mask = _rect_mask()
    mem_thin = np.zeros((30, 30), dtype=np.float32)
    mem_coarse = np.zeros((30, 30), dtype=np.float32)
    # _rect_mask's true 1px perimeter sits at rows/cols 5 and 24 (mask spans [5:25)).
    # A wide ridge spanning several px on each side of that true perimeter, so it fully
    # covers the perimeter regardless of scale: strong at both the thin and coarse scale.
    for arr, val in ((mem_thin, 1.0), (mem_coarse, 0.9)):
        arr[3:8, 3:27] = val; arr[22:27, 3:27] = val
        arr[3:27, 3:8] = val; arr[3:27, 22:27] = val
    fired, info = nuc.nucleus_thick_loop(mask, mem_thin, mem_coarse)
    assert fired is True


def test_thick_loop_does_not_fire_on_thin_ridge():
    mask = _rect_mask()
    mem_thin = np.zeros((30, 30), dtype=np.float32)
    mem_coarse = np.zeros((30, 30), dtype=np.float32)
    # A normal-width ridge exactly on the mask's true perimeter (rows/cols 5, 24; the
    # same construction test_membrane_metric.py's boundary_on_membrane test already
    # verifies matches _perimeter closely): strong at the thin scale. mem_coarse is left
    # at zero, weak at the coarse scale, as a normal-width membrane predicts.
    mem_thin[5, 5:25] = 1.0; mem_thin[24, 5:25] = 1.0
    mem_thin[5:25, 5] = 1.0; mem_thin[5:25, 24] = 1.0
    fired, info = nuc.nucleus_thick_loop(mask, mem_thin, mem_coarse)
    assert fired is False


def test_thick_loop_empty_mask():
    mask = np.zeros((30, 30), dtype=bool)
    mem = np.zeros((30, 30), dtype=np.float32)
    fired, info = nuc.nucleus_thick_loop(mask, mem, mem)
    assert fired is False


def test_nucleus_candidate_neither_cue_on_a_round_neurite_standin():
    # A round mask with normal-width membrane and bright, non-uniform interior: the
    # explicit regression test for "round does not mean nucleus."
    mask = _rect_mask()
    em = np.full((30, 30), 200.0, dtype=np.float32)
    interior = em[5:25, 5:25]
    interior[::2, :] = 150.0
    interior[1::2, :] = 210.0
    result = nuc.nucleus_candidate(mask, em)
    assert result["is_nucleus"] is False
    assert result["dark_blob"] is False
    assert result["thick_loop"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_nucleus.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'sam2_utils.nucleus'`.

- [ ] **Step 3: Implement `sam2_utils/nucleus.py`**

```python
"""Nucleus-capture detection: is a candidate mask actually the nucleus, not the cell.

Two intensity/texture cues, deliberately never shape (real neurites can be round too, the
2026-07-28 visual verdict this module is built around). Pure functions, torch-free, no frame
or dataset coupling, matching sam2_utils/membrane.py's existing pattern. Design:
docs/superpowers/specs/2026-07-29-nucleus-detector-design.md
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from sam2_utils.membrane import membrane_map, _perimeter

DEFAULT_DARK_PERCENTILE = 15.0
DEFAULT_UNIFORMITY_MAX = 25.0
DEFAULT_TAU = 0.5
DEFAULT_THICK_RATIO = 0.6
DEFAULT_SIGMAS_THIN = (1, 2, 3)
DEFAULT_SIGMAS_COARSE = (3, 5, 7)


def nucleus_dark_blob(mask: np.ndarray, em_patch: np.ndarray, *,
                      dark_percentile: float = DEFAULT_DARK_PERCENTILE,
                      uniformity_max: float = DEFAULT_UNIFORMITY_MAX
                      ) -> tuple[bool, dict]:
    """Small-nucleus cue: a uniformly dark ("boba pearl") mask interior.

    Excludes a 1px border strip (the eroded mask) so membrane pixels on the boundary
    don't skew the interior read. Fires when the interior mean sits below the
    dark_percentile of the SURROUNDING context's intensity distribution (the pixels
    outside the mask, not the interior's own values, comparing against the interior's
    own values would be self-referential and degenerate on a uniform image, e.g. a
    bright mask on a bright background would trivially clear a percentile computed from
    itself) AND the interior standard deviation stays below uniformity_max (uniform, not
    the patchier texture of cytoplasm with scattered organelles at the same mean
    darkness)."""
    img = em_patch.mean(axis=2) if em_patch.ndim == 3 else em_patch
    interior = mask & ndi.binary_erosion(mask)
    vals = img[interior]
    outside = img[~mask]
    if vals.size == 0 or outside.size == 0:
        return False, {"mean_intensity": None, "uniformity": None, "threshold": None}
    mean_val = float(vals.mean())
    std_val = float(vals.std())
    threshold = float(np.percentile(outside, dark_percentile))
    fired = bool(mean_val < threshold and std_val < uniformity_max)
    return fired, {"mean_intensity": mean_val, "uniformity": std_val, "threshold": threshold}


def nucleus_thick_loop(mask: np.ndarray, mem_thin: np.ndarray, mem_coarse: np.ndarray, *,
                       tau: float = DEFAULT_TAU, thick_ratio: float = DEFAULT_THICK_RATIO
                       ) -> tuple[bool, dict]:
    """Big-nucleus cue: the mask's OWN perimeter stays strongly membrane-like even at a
    coarse ridge-detection scale, where a normal-width cell membrane's response drops
    off. mem_thin and mem_coarse are membrane_map(...) outputs at different `sigmas`
    (thin: the module default; coarse: a wider scale tuned for thick ridges), both
    already computed by the caller on the same EM patch, this function only compares
    them along the perimeter."""
    perim = _perimeter(mask)
    p = int(perim.sum())
    if p == 0:
        return False, {"thin_frac": None, "coarse_frac": None, "ratio": None}
    thin_frac = float((perim & (mem_thin > tau)).sum()) / p
    coarse_frac = float((perim & (mem_coarse > tau)).sum()) / p
    ratio = coarse_frac / thin_frac if thin_frac > 0 else 0.0
    fired = bool(thin_frac > 0.5 and ratio >= thick_ratio)
    return fired, {"thin_frac": thin_frac, "coarse_frac": coarse_frac, "ratio": ratio}


def nucleus_candidate(mask: np.ndarray, em_patch: np.ndarray, *,
                      sigmas_thin=DEFAULT_SIGMAS_THIN, sigmas_coarse=DEFAULT_SIGMAS_COARSE,
                      dark_percentile: float = DEFAULT_DARK_PERCENTILE,
                      uniformity_max: float = DEFAULT_UNIFORMITY_MAX,
                      tau: float = DEFAULT_TAU, thick_ratio: float = DEFAULT_THICK_RATIO
                      ) -> dict:
    """Combine both cues. A mask flagged by either is a nucleus candidate."""
    dark_fired, dark_info = nucleus_dark_blob(
        mask, em_patch, dark_percentile=dark_percentile, uniformity_max=uniformity_max)
    mem_thin = membrane_map(em_patch, sigmas=sigmas_thin)
    mem_coarse = membrane_map(em_patch, sigmas=sigmas_coarse)
    thick_fired, thick_info = nucleus_thick_loop(
        mask, mem_thin, mem_coarse, tau=tau, thick_ratio=thick_ratio)
    return {
        "is_nucleus": bool(dark_fired or thick_fired),
        "dark_blob": dark_fired, "dark_blob_info": dark_info,
        "thick_loop": thick_fired, "thick_loop_info": thick_info,
    }
```

While implementing, run the tests iteratively (`py -3 -m pytest tests/test_nucleus.py -k
test_dark_blob -v`, etc.) and adjust the exact threshold constants if a test's synthetic construction
doesn't cleanly clear or miss the bar as written, this is expected tuning, the spec explicitly labels
these thresholds "eyeballed for v1... comparative, not absolute." Keep every test's INTENT unchanged
(dark+uniform fires, patchy-at-same-mean does not, bright does not fire, thick ridge fires, thin ridge
does not fire, round-but-normal mask fires neither cue) even if you adjust a numeric literal to make
the synthetic construction actually land on the right side of a threshold.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_nucleus.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check sam2_utils/nucleus.py tests/test_nucleus.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add sam2_utils/nucleus.py tests/test_nucleus.py
git commit -m "feat(nucleus): add the classical dark-blob and thick-loop nucleus-capture detector"
```

---

### Task 4: docs

**Files:**
- Modify: `docs/explanation/roadmap.md` (item 2e in section 5)
- Modify: `docs/CHANGELOG.md`
- Modify: `sam2_utils/labels.py` (docstring only, if not already covered by Task 1)
- Modify: `docs/reference/configuration.md` (only if it documents `ERROR_TYPES` or similar GUI
  config already; check first, `grep -n "ERROR_TYPES\|error_type" docs/reference/configuration.md`)

**Interfaces:**
- Consumes: Task 1's `ERROR_TYPES` addition, Task 2's gate verdict and spot-check findings, Task 3's
  existence (or absence, if skipped) and test results.

- [ ] **Step 1: Read what actually happened**

Read Task 2's commit message and, if you have it, its report (the controller may hand you a summary
instead of a raw file, in that case use what you're given). Confirm whether Task 3 ran or was skipped.
You need the real gate verdict and the real reasoning behind it, do not write generic placeholder
prose, this is a measurement writeup, same discipline as the temporal-projection docs.

- [ ] **Step 2: Update the roadmap**

In `docs/explanation/roadmap.md`, find item 2e (search for "nucleus detection by intensity/texture,
not shape"). Rewrite it to record:
- That NucleoNet turned out to be real (not an unconfirmed name), with a one-line summary of what it
  is and its license.
- The spot-check's actual verdict (generalizes / does not generalize / blocked) and why, citing the
  specific visual observations from Task 2's report.
- If Task 3 ran: that the classical `sam2_utils/nucleus.py` detector (dark-blob + thick-loop cues) was
  built instead, with its test count, and that formal scoring awaits the new GUI-collected labeled set.
- If Task 3 did not run: that NucleoNet is the detector, unvalidated by formal metric yet, same
  scoring caveat.
- That wiring the (chosen) detector into `multimask_generous` or any other live lever is still a
  separate, future spec (unchanged from the design's stated scope).

Also update the section 5b queue if item 2e is tracked there under a numbered item (search for it);
match the existing DONE / PARTLY DONE / READY / TODO vocabulary (the legend was extended with PARTLY
DONE during the temporal-projection work).

- [ ] **Step 3: Add a CHANGELOG entry**

Follow the existing entries' format (most-recent-first, an anchor id, a `##` heading, prose
paragraphs, a `---` separator; look at the top few entries in `docs/CHANGELOG.md` for the exact
pattern). Cover: the `"nucleus"` GUI label addition, the verified NucleoNet identity and license, the
spot-check's real outcome, and whether the classical fallback was built.

- [ ] **Step 4: Update `sam2_utils/labels.py`'s module docstring, if it names the error types**

Run: `grep -n "wrong_object\|ERROR_TYPES" sam2_utils/labels.py`

If any docstring or comment enumerates the categories by name (beyond the `ERROR_TYPES` tuple
itself, already correct from Task 1), add `nucleus` there too for consistency.

- [ ] **Step 5: Check and update `docs/reference/configuration.md` if needed**

Run: `grep -n "ERROR_TYPES\|error_type\|wrong_object" docs/reference/configuration.md`

If it lists the error-type categories, add `nucleus`. If it doesn't mention them at all, skip this
file, no new documentation surface is needed for a value addition to an already-documented mechanism.

- [ ] **Step 6: Humanize and check for em dashes**

Run the `humanizer` skill on everything you wrote in Steps 2-3 before committing (required by
`CLAUDE.md`, not optional). Then:

Run: `grep -n $'\xe2\x80\x94\|\xe2\x80\x93' docs/explanation/roadmap.md docs/CHANGELOG.md`
Expected: no output (no em or en dashes).

- [ ] **Step 7: Run the full suite one last time**

Run: `py -3 -m pytest -q`
Expected: all tests pass (this task shouldn't change any code, but confirm nothing else drifted).

- [ ] **Step 8: Commit**

```bash
git add docs/explanation/roadmap.md docs/CHANGELOG.md sam2_utils/labels.py docs/reference/configuration.md
git commit -m "docs(roadmap): record the nucleus-capture detector's spot-check result"
```

(Drop any file from the `git add` list that Steps 4-5 determined didn't need a change.)
