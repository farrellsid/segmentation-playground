# Organelle blob suppression implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress small, dark, round organelles in the scale-8 Sato ridge map by detecting them
(Otsu threshold + connected-component shape filtering) and inpainting them out of the EM crop before
`membrane_map()` runs, then measure with the existing `dense_membrane_fill.py` sweep whether this
moves the ~40% bleed-per-fill floor the temporal-projection lever failed to move.

**Architecture:** Two new pure functions (`detect_organelle_blobs`, `suppress_organelles`) in
`sam2_utils/membrane.py`, unit-tested against synthetic cases already verified by direct execution
before this plan was written. `experiments/dense_membrane_fill.py` gains a `--suppress-organelles`
flag wired into the same crop-building step the (now-closed) temporal-projection window used, plus a
`--sweep-organelle` mode. A real calibration pass against actual target-worm EM precedes the gate run.

**Tech Stack:** numpy, scipy.ndimage, skimage (filters.threshold_otsu, measure.regionprops,
restoration.inpaint_biharmonic), all already dependencies. No new dependency, no torch.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (`CLAUDE.md`).
- Run the `humanizer` skill on any prose you are about to commit before committing.
- New tests stay torch-free and CPU-only; `py -3 -m pytest`.
- Lint with `ruff check .`, touching only files you edit.
- No shape/roundness term ever used to EXCLUDE a real structure from protection: the eccentricity
  filter here exists to protect ridge-like membranes from suppression, not to detect them, this is the
  organelle-suppression lever, not the (separately resolved) nucleus detector.
- This is measurement only: no lever, no mask change, no wiring into `MembraneSource` or any live
  segmentation path.
- Commit incrementally, one concern per commit.

---

### Task 1: `detect_organelle_blobs` and `suppress_organelles`

**Files:**
- Modify: `sam2_utils/membrane.py`
- Test: `tests/test_organelle_blobs.py` (new)

**Interfaces:**
- Produces: `detect_organelle_blobs(em_patch: np.ndarray, *, max_area: float = 150.0,
  max_eccentricity: float = 0.85, dilate_px: int = 2) -> np.ndarray` (boolean mask),
  `suppress_organelles(em_patch: np.ndarray, organelle_mask: np.ndarray) -> np.ndarray`
- Consumes: nothing from other tasks (no dependencies).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_organelle_blobs.py`:

```python
import numpy as np
from sam2_utils import membrane as mb


def test_dark_round_blob_fully_detected():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    blob = (yy - 20) ** 2 + (xx - 20) ** 2 <= 3 ** 2
    patch[blob] = 20.0
    mask = mb.detect_organelle_blobs(patch)
    # every true blob pixel is covered (the dilated mask is a superset of the true blob)
    assert bool((blob & mask).sum() == blob.sum())
    # nothing flagged far from the blob
    assert not mask[0:5, 0:5].any()


def test_thin_ridge_not_detected():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    patch[:, 19:21] = 20.0  # a 2px-wide, full-height dark ridge
    mask = mb.detect_organelle_blobs(patch)
    assert mask.sum() == 0


def test_suppress_hard_edged_blob_reaches_background():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    blob = (yy - 20) ** 2 + (xx - 20) ** 2 <= 3 ** 2
    patch[blob] = 20.0
    mask = mb.detect_organelle_blobs(patch)
    cleaned = mb.suppress_organelles(patch, mask)
    assert abs(float(cleaned[20, 20]) - 200.0) < 1e-3


def test_suppress_soft_edged_blob_dilation_helps():
    yy, xx = np.ogrid[:40, :40]
    dist2 = (yy - 20) ** 2 + (xx - 20) ** 2
    soft_blob = (200.0 - 180.0 * np.exp(-dist2 / (2 * 3.0 ** 2))).astype(np.float32)
    mask_undilated = mb.detect_organelle_blobs(soft_blob, dilate_px=0)
    mask_dilated = mb.detect_organelle_blobs(soft_blob, dilate_px=2)
    cleaned_undilated = mb.suppress_organelles(soft_blob, mask_undilated)
    cleaned_dilated = mb.suppress_organelles(soft_blob, mask_dilated)
    # dilation gets meaningfully closer to the true background (200.0) than no dilation
    assert abs(float(cleaned_dilated[20, 20]) - 200.0) < abs(float(cleaned_undilated[20, 20]) - 200.0)


def test_suppress_organelles_empty_mask_is_noop():
    patch = np.full((40, 40), 200.0, dtype=np.float32)
    empty_mask = np.zeros((40, 40), dtype=bool)
    result = mb.suppress_organelles(patch, empty_mask)
    assert np.array_equal(result, patch)


def test_detect_organelle_blobs_uniform_patch_no_exception():
    patch = np.full((40, 40), 150.0, dtype=np.float32)
    mask = mb.detect_organelle_blobs(patch)
    assert mask.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3 -m pytest tests/test_organelle_blobs.py -v`
Expected: FAIL, `AttributeError: module 'sam2_utils.membrane' has no attribute 'detect_organelle_blobs'`.

- [ ] **Step 3: Implement `detect_organelle_blobs` and `suppress_organelles`**

In `sam2_utils/membrane.py`, add after the existing temporal-projection functions
(`register_crops`/`project_crops`, or after `membrane_map` if those aren't present in this checkout,
either placement is fine, this task's functions have no dependency on them):

```python
DEFAULT_BLOB_MAX_AREA = 150.0
DEFAULT_BLOB_MAX_ECCENTRICITY = 0.85
DEFAULT_BLOB_DILATE_PX = 2


def detect_organelle_blobs(em_patch: np.ndarray, *, max_area: float = DEFAULT_BLOB_MAX_AREA,
                           max_eccentricity: float = DEFAULT_BLOB_MAX_ECCENTRICITY,
                           dilate_px: int = DEFAULT_BLOB_DILATE_PX) -> np.ndarray:
    """Detect small, dark, round structures (organelles) in an EM patch, never ridges.

    Otsu-thresholds the patch and labels connected components of the dark side, then
    keeps only components that are both small (area <= max_area) and compact
    (eccentricity <= max_eccentricity, i.e. round, not elongated). This is a genuine
    shape discriminator: a ridge's eccentricity is close to 1, a compact blob's is
    well below that. A Gaussian-scale blob detector (skimage.feature.blob_dog) was
    tried first and rejected: verified directly that it does not discriminate blob
    shape from ridge shape at this resolution (it fired on a synthetic ridge as much
    as a synthetic blob), because scale-8 membranes are only a few pixels wide, the
    same spatial scale small organelles need.

    The returned mask is dilated by dilate_px before being returned (not a separate
    step suppress_organelles has to remember): verified this meaningfully improves
    suppression completeness on a soft-edged (realistic) blob, though it makes no
    visible difference on a hard-edged synthetic one, where the detected region
    already equals the true dark region exactly. A uniform (textureless) patch needs
    no special-case handling: threshold_otsu returns that constant value as the
    threshold (verified directly, it does not raise), every pixel satisfies
    img <= thresh, and the resulting single whole-image component is then rejected
    by the max_area filter below, naturally producing an empty mask."""
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops

    img = em_patch.mean(axis=2) if em_patch.ndim == 3 else em_patch
    img = img.astype(np.float32)
    thresh = threshold_otsu(img)
    dark = img <= thresh
    lbl, _n = ndi.label(dark)
    out = np.zeros(img.shape[:2], dtype=bool)
    for rp in regionprops(lbl):
        if rp.area <= max_area and rp.eccentricity <= max_eccentricity:
            out[lbl == rp.label] = True
    if dilate_px > 0 and out.any():
        out = ndi.binary_dilation(out, iterations=dilate_px)
    return out


def suppress_organelles(em_patch: np.ndarray, organelle_mask: np.ndarray) -> np.ndarray:
    """Replace organelle_mask's True pixels with a locally-consistent inpainted value.

    organelle_mask is expected to already be dilated (detect_organelle_blobs does
    this), so this function does not dilate again. Returns em_patch unchanged
    (no-op) when organelle_mask has no True pixels, avoiding a wasted inpainter
    call."""
    from skimage.restoration import inpaint_biharmonic

    if not organelle_mask.any():
        return em_patch
    img = em_patch.astype(np.float32)
    return inpaint_biharmonic(img, organelle_mask)
```

Check the top of `sam2_utils/membrane.py` for the existing `from scipy import ndimage as ndi` import
(already present, used by `_perimeter` and other existing functions); `detect_organelle_blobs` reuses
it, do not add a second import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3 -m pytest tests/test_organelle_blobs.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check sam2_utils/membrane.py tests/test_organelle_blobs.py`
Expected: no issues.
Run: `py -3 -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add sam2_utils/membrane.py tests/test_organelle_blobs.py
git commit -m "feat(membrane): add detect_organelle_blobs and suppress_organelles"
```

---

### Task 2: wire into `dense_membrane_fill.py`, calibrate against real data, run the gate

**Files:**
- Modify: `experiments/dense_membrane_fill.py`

**Interfaces:**
- Consumes: `mb.detect_organelle_blobs(em_patch, *, max_area=..., max_eccentricity=..., dilate_px=...)`,
  `mb.suppress_organelles(em_patch, organelle_mask)` from Task 1 (exact signatures above).
- Produces: `--suppress-organelles`, `--blob-max-area`, `--blob-max-eccentricity`, `--blob-dilate-px`,
  `--sweep-organelle` CLI flags. No other task in this plan consumes these.

- [ ] **Step 1: Confirm the current exact state matches this plan's assumptions**

`experiments/dense_membrane_fill.py`'s `grow_all` (currently at line 86) has this signature and this
membrane-map call site (currently lines 120-121):

```python
def grow_all(nmasks: dict, frames: dict, center_z: int, window: int, combine: str,
            pad: int, nodes, radius: int):
    ...
        em_crop = mb.project_crops(crops, combine=combine)
        mem = mb.membrane_map(em_crop)
```

Run `grep -n "^def grow_all\|mb.membrane_map\|grow_all(" experiments/dense_membrane_fill.py` and
confirm the output matches this (line numbers may have shifted slightly from other unrelated commits,
that's fine; the signature and the `em_crop = mb.project_crops(...)` then `mem = mb.membrane_map(em_crop)`
sequence should match). If it does not match closely, STOP and report NEEDS_CONTEXT describing what's
different, rather than guessing how to adapt the steps below.

- [ ] **Step 2: Add the CLI flags**

In `main()`, add near the existing `--mm-window`/`--mm-combine` flags (if present) or near `--cap`/
`--uf-min` otherwise:

```python
    ap.add_argument("--suppress-organelles", action="store_true",
                    help="detect and inpaint out small dark round organelles before membrane_map")
    ap.add_argument("--blob-max-area", type=float, default=150.0,
                    help="max connected-component area (px) to count as an organelle, not a ridge")
    ap.add_argument("--blob-max-eccentricity", type=float, default=0.85,
                    help="max eccentricity (0=circle, close to 1=elongated) to count as an organelle")
    ap.add_argument("--blob-dilate-px", type=int, default=2,
                    help="dilate the detected organelle mask by this many px before inpainting")
    ap.add_argument("--sweep-organelle", action="store_true",
                    help="grid a few (max_area, max_eccentricity) settings and print the same "
                         "bleed/underfill table as --sweep")
```

- [ ] **Step 3: Thread four new parameters into `grow_all` and apply suppression at the crop-building step**

Change `grow_all`'s signature (add the four new keyword-only parameters at the end, after `radius`):

```python
def grow_all(nmasks: dict, frames: dict, center_z: int, window: int, combine: str,
            pad: int, nodes, radius: int, *, suppress_organelles_flag: bool = False,
            blob_max_area: float = 150.0, blob_max_eccentricity: float = 0.85,
            blob_dilate_px: int = 2):
```

Change the body's crop-building sequence from:

```python
        em_crop = mb.project_crops(crops, combine=combine)
        mem = mb.membrane_map(em_crop)
```

to:

```python
        em_crop = mb.project_crops(crops, combine=combine)
        if suppress_organelles_flag:
            organelle_mask = mb.detect_organelle_blobs(
                em_crop, max_area=blob_max_area, max_eccentricity=blob_max_eccentricity,
                dilate_px=blob_dilate_px)
            em_crop = mb.suppress_organelles(em_crop, organelle_mask)
        mem = mb.membrane_map(em_crop)
```

Update BOTH existing call sites (there are two: the normal path in `main()`, currently around line
242, and the `--sweep-temporal` loop's call, currently around line 296) to pass the four new values
through as keyword arguments, e.g.:

```python
    recs = grow_all(nmasks, frames, args.z, args.mm_window, args.mm_combine, args.pad,
                    nodes, DEFAULT_RADIUS, suppress_organelles_flag=args.suppress_organelles,
                    blob_max_area=args.blob_max_area, blob_max_eccentricity=args.blob_max_eccentricity,
                    blob_dilate_px=args.blob_dilate_px)
```

Apply the same keyword-argument addition to the second call site inside the `--sweep-temporal` loop
(search for the second `grow_all(` call, matching the args-based pattern above but using whatever
local loop variables that call site already uses for its other positional arguments, unchanged).

- [ ] **Step 4: Smoke-test the flag runs without error**

Run: `py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 --suppress-organelles`
Expected: runs to completion, no exception. The printed numbers are not yet meaningful (defaults are
unverified against real organelle sizes), this step only confirms the wiring itself is correct.

- [ ] **Step 5: Run the full test suite**

Run: `py -3 -m pytest -q`
Expected: all tests pass. `ruff check experiments/dense_membrane_fill.py` clean.

- [ ] **Step 6: Commit the wiring**

```bash
git add experiments/dense_membrane_fill.py
git commit -m "feat(experiments): wire organelle suppression into dense_membrane_fill"
```

- [ ] **Step 7: Real calibration against target-worm EM (required work, not optional)**

F: drive must be mounted (`ls "F:/ZhenLab/Data/output_masks/resolution_experiments/"` should list
directories; if not, stop and report BLOCKED). The defaults (`max_area=150.0`,
`max_eccentricity=0.85`, `dilate_px=2`) come from one synthetic test case, not real organelle
statistics. Before the gate run means anything, render what `detect_organelle_blobs` actually flags
on a real frame and judge it by eye:

Write a small, throwaway verification (not a committed script, a one-off `py -3 -c "..."` or a
temporary script you delete after use is fine) that: loads `pipeline.load_frame_sam(1456, scale=8)`,
takes a representative crop (e.g. `[400:600, 400:600]`, matching the size used during this plan's
design verification), runs `mb.detect_organelle_blobs` on it with the defaults, and renders the
detected mask over the raw EM crop (matplotlib, two panels, same pattern
`experiments/nucleonet_spotcheck.py` uses: `ax[0].imshow(raw)`, `ax[1].imshow(overlay)`). Save the
figure to `docs/figures/presentation/organelle-blob-calibration/` (untracked, matches this repo's
existing convention for exploratory figures) and look at it.

Judge: do the flagged regions look like real organelles (compact, dark, round, plausible size for a
mitochondrion or vesicle at scale-8), or does the detector flag membrane fragments, noise, or nothing
useful? Adjust `max_area`/`max_eccentricity`/`dilate_px` and re-render until the detected regions look
right, using the plan's defaults as the starting point, not the final answer. Record in your report
what you started with, what you changed, why, and the final calibrated values you are using for the
gate run below.

- [ ] **Step 8: Run the real gate**

With the calibrated defaults from Step 7 (pass them explicitly via the CLI flags if they differ from
the plan's starting defaults), run:

```bash
py -3 experiments/dense_membrane_fill.py --z 1456 --uf-min 0.6 --suppress-organelles \
    --blob-max-area <calibrated> --blob-max-eccentricity <calibrated> --blob-dilate-px <calibrated>
```

Capture the full printed output, including the `filled`, `foreign`, `bleed_cells`, `new_bleed`,
`mean_uf`, `area%`, and `contested` numbers (the same table format `--sweep`/`--sweep-temporal`
already print). Compare against the window=0, no-suppression baseline already on record from the
temporal-projection work (foreign 39, bleed_cells 28/117, mean_uf 0.403, area +12%, at the same
z=1456, uf_min 0.6 settings, see `docs/CHANGELOG.md`'s 2026-07-29 temporal membrane projection entry
for the exact baseline numbers to compare against, since the baseline row itself does not need to be
re-run, only cited).

- [ ] **Step 9: Read the gate honestly**

Two outcomes, do not assume either ahead of running it:

- **The floor drops**: organelle suppression's `new_bleed`/`filled` rate (compute this the same way
  the temporal-projection work's final fix computed the bleed-per-fill rate:
  `new_bleed / filled`) is meaningfully lower than the 0.38 baseline rate on record. This is a real,
  positive result, note it and flag this as ready for the roadmap's Phase 2c/2d to build on.
- **The floor does not move, or gets worse**: the rate is at or above 0.38. Also a real, useful
  result: it means organelle suppression, even correctly calibrated, is not the lever that clears
  Phase 2c/2d's gate, and the roadmap's own decision point (skip to a learned membrane map) applies.

Report the actual `new_bleed / filled` rate computed from the real run, not a qualitative impression.

- [ ] **Step 10: Commit the calibration figure reference and gate numbers to your report**

No code commit needed for this step (the calibration figure is untracked, matching repo convention);
just ensure your final report (see Report Format below) captures the calibrated parameter values and
the real gate numbers, since Task 3 (docs) depends on them.

---

### Task 3: docs

**Files:**
- Modify: `docs/explanation/roadmap.md`
- Modify: `docs/CHANGELOG.md`

**Interfaces:**
- Consumes: Task 2's real calibration decisions and gate numbers (the `new_bleed / filled` rate,
  compared against the 0.38 baseline).

- [ ] **Step 1: Read what actually happened**

Read Task 2's report for the real calibrated parameter values and the real gate outcome. Do not write
generic placeholder prose; this is a measurement writeup, matching every other lever this project has
documented (temporal projection, z-consistency, nucleus detection).

- [ ] **Step 2: Update the roadmap**

In `docs/explanation/roadmap.md`, find item 2b.5's writeup (section 5) and the section 6 decision
point that currently reads "the still-untried intensity/texture blob filter... stays the next thing
to try" (search for that phrase). Update both with:
- The real calibrated parameters (`max_area`, `max_eccentricity`, `dilate_px`) and what they were
  adjusted from.
- The real gate numbers: the baseline `new_bleed/filled` rate (0.38, already on record) vs. this
  lever's rate, computed the same way.
- Whichever outcome actually happened (floor drops, or does not), stated plainly, matching this
  project's "measurement first, evidence-gated" documentation culture, never rounded up or softened.
- If the floor did not move: state explicitly that items 2c/2d now fall to the roadmap's own
  documented decision point (skip to a learned membrane map, Phase 3), since that decision point's
  text already commits to this consequence once the classical route is exhausted.
- If the floor did drop: note that items 2c (grow-to-membrane) and 2d (non-overlap arbitration) are
  now unblocked, as a separate future spec, not something this landing itself builds.

- [ ] **Step 3: Add a CHANGELOG entry**

Follow the existing entries' format (anchor id, `##` heading, prose, `---` separator,
most-recent-first in both the file body and the Contents list). Cover the design correction process
(the `blob_dog` attempt and why it was rejected before any plan existed, briefly, the full detail
already lives in the spec) and the real calibration and gate numbers.

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
git commit -m "docs(roadmap): record the organelle blob suppression gate result"
```
