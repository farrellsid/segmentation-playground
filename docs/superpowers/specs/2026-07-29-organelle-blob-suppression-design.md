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

## Verified before writing this spec (not vibes)

Checked directly against a real target-worm EM crop (`pipeline.load_frame_sam(1456, scale=8)`,
a 200x200 window) before committing to this mechanism:

- `skimage.feature.blob_dog` (Difference-of-Gaussians blob detector) is already available (skimage is
  an existing dependency via `membrane_map`'s Sato filter) and fast: 0.14s on a 200x200 crop.
- **The input must be normalized to `[0, 1]` before calling `blob_dog`, or `threshold` is
  meaningless.** On the raw dark-inverted crop (values roughly 0-250), sweeping `threshold` from 0.05
  to 0.5 barely changed the blob count (1372 to 1370): the DoG response magnitude scales with the
  input's raw value range, so an unnormalized image saturates past any reasonable threshold. After
  scaling the dark-inverted crop to `[0, 1]`, the same sweep gave a real, usable dynamic range: 1093,
  715, 359, 148, 2, 0 blobs at thresholds 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 respectively. `threshold`
  somewhere in `0.15-0.3` is a reasonable starting search range for this crop size, not a number to
  treat as final, real calibration against known organelle locations is still Task 2's job.
- `skimage.restoration.inpaint_biharmonic(image, mask)` is already available in the installed skimage
  (0.26.0) and is the right tool for the suppression half: a principled PDE-based inpainter that fills
  a masked region with values consistent with its surrounding pixels, rather than a flat fill that
  would itself look like a new, different kind of artifact to the Sato filter.
- **Why a blob detector and not a blanket morphological filter.** A grayscale morphological closing
  (dilate then erode) with a disk structuring element would also remove small dark features, but it
  is shape-blind: it cannot distinguish a small round organelle from a short segment of a genuinely
  thin membrane, and would erode real membrane signal along with the organelles it is meant to
  suppress. A blob detector is shape-selective by construction (DoG responds to compact, roughly
  isotropic dark regions, not to elongated ridges), so it targets organelles specifically without the
  same collateral risk to thin membrane structure. This is the same reasoning the roadmap already
  uses to reject shape-based nucleus detection ("round does not mean nucleus") turned around: here,
  round is exactly the discriminating feature, because the alternative (ridge-like) is the thing being
  protected, not suppressed.

## The mechanism

Two pure functions in `sam2_utils/membrane.py`, alongside `membrane_map` (same file: both operate on
a raw EM patch and belong to the same "signal preparation before ridge-filtering" concern; the
temporal-projection functions, `register_crops`/`project_crops`, already live here for the same
reason):

```python
def detect_organelle_blobs(em_patch: np.ndarray, *, min_sigma: float = 1.0,
                           max_sigma: float = 4.0, threshold: float = 0.2) -> np.ndarray
```

Dark-inverts the patch (`patch.max() - patch`, matching `membrane_map`'s own "membranes are dark on
bright cytoplasm" framing turned around: organelles are also dark, so the same inversion applies),
normalizes to `[0, 1]` by its own max (a per-patch normalization, matching `membrane_map`'s own
99th-percentile normalization pattern so the threshold is stable frame to frame rather than tied to
one crop's absolute brightness), and runs `skimage.feature.blob_dog`. Returns a boolean mask, same
`H x W` as the input, `True` at every pixel covered by a detected blob (each `(y, x, sigma)` result
drawn as a filled disk of radius `sigma * sqrt(2)`, matching scikit-image's own documented radius
convention for DoG blob detection).

```python
def suppress_organelles(em_patch: np.ndarray, organelle_mask: np.ndarray) -> np.ndarray
```

Returns a copy of `em_patch` with `organelle_mask`'s `True` pixels replaced via
`skimage.restoration.inpaint_biharmonic`. If `organelle_mask` has no `True` pixels, returns `em_patch`
unchanged (no-op, mirroring `project_crops`'s single-element identity behavior, cheap to guarantee
and avoids calling the inpainter on an all-`False` mask).

Both functions are pure, torch-free, take already-loaded arrays only, no frame-store or pipeline
coupling, matching every other function in `sam2_utils/membrane.py`.

## Integration

`experiments/dense_membrane_fill.py` gains a `--suppress-organelles` flag (boolean, default off) and
exposed detector parameters (`--blob-min-sigma`, `--blob-max-sigma`, `--blob-threshold`, defaults
from the verified starting range above, to be refined by real calibration in Task 2). When set,
`grow_all`'s per-neuron crop building step runs `detect_organelle_blobs` then `suppress_organelles`
on the EM crop before handing it to `mb.membrane_map()`, in place of the raw crop, the same insertion
point the temporal-projection window's `project_crops` output used. This composes with, but is
independent of, the (now-closed) `--mm-window`/`--mm-combine` temporal flags, since organelle
suppression operates within a single slice and does not need a z-window; passing both is a valid,
untested combination this spec does not require exploring (temporal projection is closed as a lever,
not deleted as code, so nothing prevents someone from trying the combination later, but this spec's
gate is organelle suppression alone against the `--mm-window 0` baseline).

A `--sweep-organelle` flag grids a small set of `(threshold, min_sigma, max_sigma)` combinations
(exact grid decided during calibration, since the useful range is not yet known precisely, only
roughly bounded by the pre-spec check above) and prints the same bleed/underfill table `--sweep` and
`--sweep-temporal` already print, `filled` column included from the start this time (the
temporal-projection plan had to retrofit that column after its first gate run overstated a result;
this spec starts with it), so the gate reads directly off the table.

## Calibration (real work, not a formality)

Unlike the z-to-z consistency metric, this lever's parameters are not yet well-calibrated, the
pre-spec check above only established that the mechanism is real and roughly where a usable threshold
range starts. Before running the full gate sweep, Task 2 must:

1. Run `detect_organelle_blobs` on a real target-worm crop (the same z=1456 frame the rest of this
   project's dense-frame tooling uses) and render the detected blob mask over the raw EM (matching the
   visual-gut-check pattern `experiments/nucleonet_spotcheck.py` and the registration overlay before
   it both used: no ground truth exists for "which pixels are organelles," so this is judged by eye
   against what a human would call an organelle on the raw EM).
2. Adjust `min_sigma`/`max_sigma`/`threshold` until the detected blobs look like real organelles
   (compact, dark, round) and not membrane fragments or noise, using the pre-spec-verified range as
   the starting search point, not the final answer.
3. Only then run `--sweep-organelle` for the actual gate measurement, with the calibrated defaults.

## Testing (CPU, torch-free)

`tests/test_organelle_blobs.py`, synthetic arrays, no frames or torch:

- A dark round blob on a bright background: `detect_organelle_blobs` flags pixels at the blob's
  location; a bright background elsewhere: not flagged.
- A dark, thin, elongated ridge (a membrane stand-in) on a bright background: `detect_organelle_blobs`
  does NOT flag the ridge, or flags it far less than the round-blob case at the same darkness and
  total dark-pixel area, the explicit regression test for "shape-selective, not just
  darkness-selective" (the whole reason a blob detector was chosen over a morphological filter).
- `suppress_organelles` on a blob mask: the returned patch's values at the masked pixels differ from
  the original (something was filled in) and are closer to the surrounding background value than the
  original dark blob value was.
- `suppress_organelles` with an all-`False` mask: returns the input unchanged (the no-op path), and
  does not call the inpainter (a fast, direct check: pass a mask of all `False` and confirm the return
  value equals the input via `np.array_equal`, verifying the short-circuit works rather than assuming
  the inpainter would have been a no-op on empty input anyway).
- `detect_organelle_blobs` on an all-uniform patch (no blobs, no ridges, no texture at all): returns
  an all-`False` mask, no exception (the degenerate case a real EM crop's edge or a padding region
  could produce).

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

- The pre-spec-verified threshold range (`0.15-0.3` on one 200x200 crop) is a starting point, not a
  calibrated default; Task 2's calibration step is required work, not optional polish, before the
  gate run means anything.
- `blob_dog` is an approximation of the more accurate but slower Laplacian-of-Gaussian detector
  (`blob_log`); chosen for speed given the calibration loop needs many quick iterations. If the DoG
  approximation turns out too coarse during calibration (misses real organelles a human can see, or
  over/under-detects in a way threshold tuning cannot fix), switching to `blob_log` is a one-line
  change behind the same function signature, not a redesign.
- Inpainting a detected blob region assumes the surrounding, non-blob pixels are a trustworthy signal
  for what should be there instead; for a very large or very dense cluster of organelles (little
  surrounding "clean" signal to interpolate from), `inpaint_biharmonic` may produce a smoothed,
  unrealistic-looking patch. This is a known general limitation of PDE inpainting on large holes, not
  specific to this use, and the calibration step's job includes checking this does not happen
  pathologically on real target-worm tissue.
- Like every other membrane-map-adjacent lever this project has built, this is comparative at the
  scale-8 grid, not an absolute organelle segmentation. It only needs to move the bleed-per-fill rate,
  not perfectly classify every organelle.
