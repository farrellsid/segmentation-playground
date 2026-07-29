# Phase 2b.5: intensity/texture blob filter (organelle suppression)

Status: design, approved 2026-07-29 (autonomous session, no interactive partner available; written
directly to spec, same as the z-consistency metric earlier this session). Scope: the suppression
mechanism and its gate measurement only, no wiring into `multimask_generous`, `MembraneSource`, or
any other live pipeline path.

## Why this, why now

The temporal-projection lever (this same roadmap item, 2b.5) is closed: no `(window, combine)`
setting improved the bleed-per-fill rate the `dense_membrane_fill.py` sweep measures, and a
shift-clamp diagnostic showed the lever's own registration step was a likely contributor to the
regression, not just blur. The roadmap's next-in-line lever for organelle suppression is a different
mechanism entirely: detect the dark, round, small structures (organelles, vesicles) that corrupt the
scale-8 Sato ridge map with false walls and false gaps, and remove them from the image BEFORE
`membrane_map()` runs, rather than combining information across z (temporal projection's approach,
now known to blur/misregister).

Nucleus detection (this roadmap's item 2e, previously bundled with this same item) is resolved
separately and is out of scope here: NucleoNet, a real pretrained EM nucleus segmenter, was
spot-checked and verdicted "generalizes," so nucleus-capture has its own detector now, unrelated to
this classical blob filter. This spec is scoped to the broader organelle-suppression problem the
Sato ridge map still has (mitochondria, vesicles, and other small dark structures that are not
specifically nuclei), which is what items 2c (grow-to-membrane) and 2d (non-overlap arbitration)
still wait on.

## Scope

In:

- A blob detector tuned for small, dark, round structures in EM crops, verified to work on real
  target-worm data before being written into a spec (see "Verified before writing this spec" below).
- A suppression step that removes detected blobs from the image before the existing `membrane_map()`
  runs, replacing each blob's pixels with a locally-consistent inpainted value rather than a naive
  fill, so the result still looks like plausible EM texture to the downstream Sato filter.
- Wiring into `experiments/dense_membrane_fill.py` (the same tool the temporal-projection lever was
  gated with), so the existing bleed-per-fill sweep can measure this lever the same way.
- A real gate run against real target-worm data, honestly reported either way.

Out:

- Wiring a validated suppression step into `eval/merge_metric.py`'s production `MembraneSource`, or
  into `sam2_utils/membrane.py`'s `membrane_map()` as a new default. That is a follow-on once the
  gate passes; premature before that, matching every other lever this project has built (measure
  first, wire in only after the gate says so).
- Distinguishing organelle TYPES (mitochondria vs vesicle vs lipid droplet). This spec suppresses
  "small dark round structure," full stop; type-specific handling (e.g. DropNet for lipid droplets
  specifically) was already scoped out of the nucleus-detector spec for the same broader-problem
  reason and stays out here too.
- Any change to a segmentation mask, a lever, or a live pipeline path. This is a membrane-map input
  transform, measured, not applied to production output.

## Verified before writing this spec (not vibes, and the first draft of this mechanism was wrong)

Checked directly against real target-worm EM crops and hand-built synthetic blob/ridge cases before
committing to this mechanism. The first attempt used `skimage.feature.blob_dog`, and it failed two
separate checks badly enough to change the design, both caught here rather than after a subagent
built and tested it:

- **`blob_dog` does not discriminate blob shape from ridge shape.** A synthetic dark round blob and a
  synthetic dark, thin, full-height ridge (both on a bright background) were both flagged by
  `blob_dog` at comparable or higher density on the ridge (the ridge test flagged 100% of its own
  pixels, more total pixels than the blob test). This directly contradicts the "shape-selective by
  construction" assumption a blob detector is normally credited with: at scale-8, real membranes are
  only a few pixels wide, close to the same spatial scale small organelles need, so a small-sigma DoG
  detector reads a several-pixel-wide dark line as a chain of local blobs. Scale-space blob detection
  alone is not the right tool here.
- **The fix: threshold + connected-component shape filtering, not scale-space blob detection.**
  `skimage.filters.threshold_otsu` on the patch, followed by `scipy.ndimage.label` on the dark side of
  that threshold and `skimage.measure.regionprops` per component, correctly separates the two cases:
  filtering components by `area` (small) and `eccentricity` (low, i.e. compact/round) flags the
  synthetic blob completely (29/29 true blob pixels) and the synthetic ridge not at all (0/80 ridge
  pixels). This is a real shape discriminator (a ridge's eccentricity is close to 1, a compact blob's
  is well below that), not a Gaussian-scale heuristic that turns out to respond to both shapes.
  `threshold_otsu` (a well-established automatic thresholding method, maximizes between-class
  variance) was chosen over a fixed percentile because organelles are a small minority of any given
  crop, so a fixed low percentile (e.g. 15th) sits inside the background range, not the organelle
  range, and would flag nearly the whole patch, verified directly, not assumed. `threshold_otsu` on a
  perfectly uniform (textureless) patch does not raise, and correctly returns nothing flagged.
- **Suppression can fail silently without mask dilation, severity depends on the detector.** With the
  original `blob_dog`-based detector, feeding `inpaint_biharmonic` a mask sized exactly to its reported
  blob radius left the center pixel completely unchanged: `blob_dog`'s sigma-derived radius
  underestimated the true dark region's extent, leaving a thin ring of still-dark pixels immediately
  outside the mask, and biharmonic inpainting uses the pixels immediately outside the mask as its
  boundary condition, so a still-dark boundary makes the solver reconstruct something close to dark in
  the interior too. The corrected Otsu-plus-connected-components detector does not have this specific
  problem on a hard-edged synthetic blob (it labels the exact true dark pixels, not an approximated
  circle, so an undilated mask already reaches full background on that test case). But a second,
  independent check on a more realistic SOFT-edged blob (a Gaussian intensity falloff, closer to real
  EM texture than a hard-edged disk) showed dilation still matters: undilated suppression left the
  center at 140.7 (out of a 20.0 blob center and 200.0 background, i.e. still visibly darker than
  background), while a 2px-dilated mask reached 180.5, meaningfully closer to true background. Neither
  fully reaches 200 in the soft-edge case, because the Gaussian's tail keeps darkening pixels even
  outside the dilated margin, so the inpainting boundary itself is not pure background either.
  **Conclusion: dilation before inpainting is kept as the default (`dilate_px=2`), not because it is
  strictly required for every detector, but because it measurably improves suppression completeness on
  realistic soft-edged organelles, and costs little.**
- `skimage.restoration.inpaint_biharmonic(image, mask)` is available in the installed skimage (0.26.0)
  and, once fed a properly-dilated mask, is the right tool for the suppression half: a principled
  PDE-based inpainter that fills a masked region with values consistent with its surrounding pixels,
  rather than a flat fill that would itself look like a new, different kind of artifact to the Sato
  filter.

## The mechanism

Two pure functions in `sam2_utils/membrane.py`, alongside `membrane_map` (same file: both operate on
a raw EM patch and belong to the same "signal preparation before ridge-filtering" concern; the
temporal-projection functions, `register_crops`/`project_crops`, already live here for the same
reason):

```python
def detect_organelle_blobs(em_patch: np.ndarray, *, max_area: float = 150.0,
                           max_eccentricity: float = 0.85, dilate_px: int = 2) -> np.ndarray
```

Otsu-thresholds the patch (`skimage.filters.threshold_otsu`, dark side of the split, no inversion or
`[0, 1]` normalization needed since Otsu finds its own split point from the patch's own histogram),
labels connected components of the dark side (`scipy.ndimage.label`), and keeps only components
whose `regionprops` `area` is at or below `max_area` and whose `eccentricity` is at or below
`max_eccentricity` (compact and round, not ridge-like). The kept components' union, dilated by
`dilate_px` (verified to meaningfully improve suppression completeness, see above), is the returned
boolean mask, same `H x W` as the input.
An all-uniform patch (no components at all) returns an all-`False` mask, no exception, verified
directly.

```python
def suppress_organelles(em_patch: np.ndarray, organelle_mask: np.ndarray) -> np.ndarray
```

Returns a copy of `em_patch` with `organelle_mask`'s `True` pixels replaced via
`skimage.restoration.inpaint_biharmonic`. The mask handed in is expected to already be dilated (that
happens inside `detect_organelle_blobs`, not here, keeping the dilation margin a detection-time
decision, not something every caller of the suppression step has to remember). If `organelle_mask` has
no `True` pixels, returns `em_patch` unchanged (no-op, mirroring `project_crops`'s single-element
identity behavior, cheap to guarantee and avoids calling the inpainter on an all-`False` mask, which
would otherwise waste a call for no effect).

Both functions are pure, torch-free, take already-loaded arrays only, no frame-store or pipeline
coupling, matching every other function in `sam2_utils/membrane.py`.

## Integration

`experiments/dense_membrane_fill.py` gains a `--suppress-organelles` flag (boolean, default off) and
exposed detector parameters (`--blob-max-area`, `--blob-max-eccentricity`, `--blob-dilate-px`,
defaulting to the verified values above, `150.0`/`0.85`/`2`, to be refined by real calibration in
Task 2 against actual target-worm organelle sizes rather than the synthetic test case's numbers).
When set,
`grow_all`'s per-neuron crop building step runs `detect_organelle_blobs` then `suppress_organelles`
on the EM crop before handing it to `mb.membrane_map()`, in place of the raw crop, the same insertion
point the temporal-projection window's `project_crops` output used. This composes with, but is
independent of, the (now-closed) `--mm-window`/`--mm-combine` temporal flags, since organelle
suppression operates within a single slice and does not need a z-window; passing both is a valid,
untested combination this spec does not require exploring (temporal projection is closed as a lever,
not deleted as code, so nothing prevents someone from trying the combination later, but this spec's
gate is organelle suppression alone against the `--mm-window 0` baseline).

A `--sweep-organelle` flag grids a small set of `(max_area, max_eccentricity)` combinations (exact
grid decided during calibration, since the useful range on real organelles is not yet known
precisely, only verified functional on a synthetic test case above) and prints the same
bleed/underfill table `--sweep` and `--sweep-temporal` already print, `filled` column included from
the start this time (the temporal-projection plan had to retrofit that column after its first gate
run overstated a result; this spec starts with it), so the gate reads directly off the table.

## Calibration (real work, not a formality)

This lever's parameters are verified functional (the mechanism works on synthetic blob/ridge cases)
but not yet calibrated against real organelle sizes and shapes on target-worm tissue. Before running
the full gate sweep, Task 2 must:

1. Run `detect_organelle_blobs` on a real target-worm crop (the same z=1456 frame the rest of this
   project's dense-frame tooling uses) and render the detected mask over the raw EM (matching the
   visual-gut-check pattern `experiments/nucleonet_spotcheck.py` and the registration overlay before
   it both used: no ground truth exists for "which pixels are organelles," so this is judged by eye
   against what a human would call an organelle on the raw EM).
2. Adjust `max_area`/`max_eccentricity`/`dilate_px` until the detected regions look like real
   organelles (compact, dark, round, roughly organelle-sized at scale-8) and not membrane fragments,
   nucleus interior, or noise, using the verified defaults (`150.0`/`0.85`/`2`) as the starting point,
   not the final answer. Pay particular attention to whether `max_eccentricity` needs tightening: the
   synthetic ridge test used a perfectly straight line, real membrane fragments inside a crop may be
   curved enough to read as more compact than the synthetic case, so the real discrimination margin on
   actual EM texture needs an eyeballed check, not just trust in the synthetic result.
3. Only then run `--sweep-organelle` for the actual gate measurement, with the calibrated defaults.

## Testing (CPU, torch-free)

`tests/test_organelle_blobs.py`, synthetic arrays, no frames or torch. The exact test cases below were
run directly before this spec was written (not just described), so the expected values are real,
observed numbers, not assumptions:

- A dark round blob (radius-3 disk) on a bright background: `detect_organelle_blobs` flags all 29 true
  blob pixels (plus a dilation margin); nothing flagged far from the blob.
- A dark, thin, full-height 2px-wide ridge (a membrane stand-in) on a bright background:
  `detect_organelle_blobs` flags zero pixels, the explicit regression test for "shape-selective via
  eccentricity, not darkness-selective," and the test that caught the original `blob_dog`-based design
  failing this exact case (it flagged 100% of the ridge's own pixels).
- `suppress_organelles` on a hard-edged blob mask: the returned patch's value at the blob's center
  reaches the true surrounding background value exactly (verified on the hard-edged synthetic case;
  with the corrected Otsu-based detector this holds even without dilation, since the detected mask
  already equals the true dark region on a hard edge).
- `suppress_organelles` on a SOFT-edged blob (Gaussian intensity falloff, closer to real EM texture):
  dilation measurably improves how close the suppressed center gets to background (verified: 140.7
  undilated vs 180.5 with `dilate_px=2`, against a 20.0 blob center and 200.0 true background), the
  regression test for dilation's real, measured benefit, distinct from the hard-edged case where it
  makes no visible difference.
- `suppress_organelles` with an all-`False` mask: returns the input unchanged (the no-op path) via
  `np.array_equal`, without calling the inpainter.
- `detect_organelle_blobs` on an all-uniform patch (no texture at all): returns an all-`False` mask,
  no exception (verified: `threshold_otsu` on a constant array does not raise, and correctly finds
  nothing to flag), the degenerate case a real EM crop's edge or a padding region could produce.

## Docs to update on landing

- `docs/explanation/roadmap.md`: record the gate result for items 2c/2d's blocking condition (item
  2b.5's writeup, and the section 6 decision point that already reads "the still-untried
  intensity/texture blob filter... stays the next thing to try") with whichever outcome the real
  sweep shows.
- `docs/CHANGELOG.md` on landing.
- No ADR: matches the temporal-projection and z-consistency specs' same reasoning, this is a
  measurement, not a load-bearing tradeoff a newcomer would question. An ADR becomes appropriate only
  if/when a validated suppression step gets wired into `MembraneSource` as a new default.

## Risks and accepted limitations

- The verified defaults (`max_area=150.0`, `max_eccentricity=0.85`, `dilate_px=2`) come from one
  synthetic test case, not real organelle statistics; Task 2's calibration step is required work, not
  optional polish, before the gate run means anything.
- A single global Otsu threshold per crop assumes organelles and cytoplasm separate reasonably well
  in one histogram split. If a crop's brightness varies a lot across its own area (uneven illumination
  or a crop spanning very different tissue), a single threshold may over- or under-segment in different
  parts of the same crop. Not observed in the synthetic tests (which are, by construction, uniform
  outside the test shape), so this is a real risk to watch for during Task 2's calibration against real
  crops, not a confirmed problem.
- `max_eccentricity=0.85` was tuned against one synthetic ridge (a perfectly straight line). A curved
  or short real membrane fragment inside a crop could score more compact (lower eccentricity) than a
  long straight synthetic ridge does, so the real margin between "organelle" and "membrane fragment"
  on actual tissue is untested until Task 2's calibration step runs on real data.
- Inpainting a detected, dilated blob region assumes the surrounding, non-blob pixels are a
  trustworthy signal for what should be there instead; for a very large or very dense cluster of
  organelles (little surrounding "clean" signal to interpolate from), `inpaint_biharmonic` may produce
  a smoothed, unrealistic-looking patch. This is a known general limitation of PDE inpainting on large
  holes, not specific to this use, and the calibration step's job includes checking this does not
  happen pathologically on real target-worm tissue.
- Dilating before inpainting means the suppressed region is always somewhat larger than the detected
  region. A larger `dilate_px` gives more suppression margin but also erodes more real surrounding
  signal; `dilate_px=2` is a starting point Task 2's calibration should sanity-check against real
  organelle spacing (a dilation margin large enough to make two nearby organelles' suppressed regions
  touch or merge is a real failure mode worth watching for), not assume is optimal.
- Like every other membrane-map-adjacent lever this project has built, this is comparative at the
  scale-8 grid, not an absolute organelle segmentation. It only needs to move the bleed-per-fill rate,
  not perfectly classify every organelle.
