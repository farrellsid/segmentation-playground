# Phase 2e: nucleus-capture detector (NucleoNet spot-check, classical fallback, labeling workflow)

Status: design, approved 2026-07-29. Scope: the detector only. Wiring detection into
`multimask_generous` or any other live pipeline lever is deferred to its own spec, once a detector
is shown to work.

## Why this, why now

The nested-membrane ceiling (roadmap problem 7, [[nucleus-capture-underfill]]) is a confident
wrong-compartment answer: when a skeleton node lands inside a nucleus, SAM segments the nucleus
instead of the cell, and every existing geometric QC signal reads this as a clean mask (the
perimeter sits on a real membrane, nothing spans, underfill is low because the nuclear envelope
walls in the flood). It is invisible to the pipeline's ruler, fixable only by recognizing the
nucleus semantically. The temporal-projection work just closed (roadmap item 2b.5) as an organelle
suppression route for the membrane map generally; this is the other, deferred half: detect the
nucleus specifically, so a later lever (the already-built `multimask_generous`, currently gated off
globally because it adds bleed everywhere it's not needed) can be applied only where it helps.

## Scope

In:

- A spot-check of NucleoNet (a real, verified, permissively-licensed pretrained EM nucleus
  instance-segmentation model, not the unconfirmed name it started as) against target-worm frames,
  to see whether it generalizes to C. elegans neurite EM without any finetuning.
- A gate: if NucleoNet's detections look right, it is the detector, full stop, no classical
  detector needed. If it doesn't generalize, build the classical fallback this same round.
- The classical fallback: two cheap, intensity/texture cues (never shape alone, per the 2026-07-28
  visual verdict that round neurites exist and are not nuclei) reusing `membrane_map()`.
- Extending the existing napari review GUI's label store with a `"nucleus"` error type, so the
  ground-truth set this detector needs (and does not yet have) builds incrementally during normal
  review sessions instead of requiring a dedicated labeling pass.

Out (deferred to later specs):

- Wiring a validated detector into `multimask_generous`, `build_prompts`, or any other live
  segmentation-affecting code path. This spec produces a measurement, not a lever, matching the
  project's spine (rulers before levers), the same split used for the membrane map (2a/2b) versus
  grow-to-membrane (2c) and, most recently, temporal projection versus the still-deferred
  intensity/texture blob filter this spec is now half of.
- Formal precision/recall scoring. Blocked on the hand-labeled set the GUI change starts collecting;
  this round's read on both NucleoNet and any classical fallback is a visual gut-check, the same
  kind Stage 0.1's registration overlay used before a formal metric existed.
- Lipid-droplet detection (DropNet, NucleoNet's sibling model). Out of scope: the roadmap's organelle
  problem is broader than lipid droplets specifically, and nucleus-capture is the one confirmed,
  named failure mode with a paper trail. Revisit if the classical route's dark-blob cue turns out to
  fire on other organelles and DropNet would disambiguate them.

## NucleoNet: what it actually is (verified, not vibes)

The `nucleus-capture-underfill` memory flagged "NucleoNet, DropNet" as unconfirmed names floated by
a student. Verified via web search before this spec was written:

- **NucleoNet and DropNet: generalist deep learning models for instance segmentation of nuclei and
  lipid droplets from electron microscopy images**, bioRxiv, April 2026
  (`10.64898/2026.04.02.713930v1`), from a team including NCI Center for Cancer Research
  researchers. Panoptic DeepLab models trained on crowdsourced, "large, heterogeneous" annotated EM
  data plus public volume-EM datasets.
- Distributed two ways: the `empanada-napari` v1.2 plugin (GUI), and the headless `empanada-dl`
  PyPI package, which has a scriptable Python inference API, no napari GUI required, so it fits this
  pipeline's batch-script architecture.
- License: BSD-3-Clause (`volume-em/empanada`), permissive, no non-commercial restriction to
  verify, unlike the roadmap's existing nnInteractive caveat.
- Requires torch (GPU optional, CPU inference works, just slower), Python 3.10 to 3.13.
- **Unverified: domain generalization to C. elegans.** Nothing in the available documentation names
  the organism or tissue types in the training set beyond "cancer models" and "in vivo tumors,"
  invertebrate or neurite data is not confirmed either way. This is exactly the domain-gap risk the
  roadmap already tracks for organelle-trained models (micro_sam's generalist degrading neurites),
  so treat NucleoNet's applicability here as unproven until the spot-check says otherwise, not as a
  safe assumption because the paper exists.

## NucleoNet spot-check

A new experiment script, `experiments/nucleonet_spotcheck.py`, following the same pattern as
`experiments/sam3_probe.py` (a characterization spike, not production code, lives in `experiments/`
until validated, promotable to `sam2_utils/` later if it graduates):

1. `pip install empanada-dl` as a spot-check-only dependency (not added to the core project
   dependencies unless the gate passes).
2. Lazy-import torch/empanada inside the script, matching `sam2_utils/sam3_backend.py`'s pattern, so
   nothing outside `experiments/` gains a new heavy dependency.
3. Run NucleoNet's headless inference on the z=1456 "current-work frame" (already has cached EM,
   masks, and the dense-overlay index from the temporal-projection and autofill work, so no new data
   plumbing), plus any additional frames already known from GUI review or the dense-frame visuals to
   contain a suspected nucleus-capture case, if such cases are readily at hand.
4. Render NucleoNet's detected nucleus instances over the raw EM, next to the existing per-slice
   masks, the same overlay style `dense_overlay.py` already uses (EM backdrop, colorized instances,
   alpha blend), so the comparison is a single image, not a table of numbers nobody has ground truth
   to check yet.

## The gate

Read the overlay by eye. Two honest outcomes:

- **NucleoNet finds real, plausible C. elegans nuclei** (round, dark, membrane-bound structures that
  match where a human would call "nucleus" on the raw EM), on target-worm tissue it never trained
  on. Then it is the detector. Stop here; do not also build the classical fallback. Note the result
  in the roadmap and queue the formal scoring (below) once the labeled set exists.
- **NucleoNet does not generalize** (misses obvious nuclei, fires on the wrong structures, or
  produces detections that don't match what a human would call a nucleus on this tissue). Then build
  the classical fallback in the same round, informed by exactly what NucleoNet got wrong, e.g. if it
  under-detects small dark nuclei specifically, that sharpens the classical dark-blob threshold's
  priority over the thick-loop cue, and vice versa.

Do not assume either outcome ahead of running it. Both are useful, real results.

## The classical fallback (if the gate requires it)

Two independent cues, matching the two nucleus types the EM cues memory already documents, neither
based on shape alone. New module `sam2_utils/nucleus.py` (a new file, not folded into
`membrane.py`, because nucleus-specific cues are a distinct concern from the generic bleed/underfill
detectors already there; it imports `membrane_map` from `membrane.py`, reuse not duplication):

- **Dark-blob cue (small nuclei).** Small nuclei read as uniformly dark "boba pearls." Compute the
  candidate mask's interior EM intensity (excluding a thin border strip so membrane pixels don't
  skew it): if the mean sits below a percentile-based threshold (computed from the crop's own
  intensity distribution, matching `membrane_map`'s percentile-normalization for resolution/contrast
  robustness, not a fixed absolute value) AND the interior's variance is low (uniform, not the
  patchier texture of cytoplasm with scattered organelles), flag as a small-nucleus candidate.
- **Thick-loop cue (big nuclei).** Big nuclei have interior texture too close to cytoplasm for the
  dark-blob cue, but their membrane is measurably thicker than an ordinary cell membrane.
  `membrane_map()` already exposes `sigmas` as a parameter: a Sato ridge filter's response is
  strongest at the sigma scale matching the ridge's actual width, so a genuinely thick membrane
  keeps a strong response even at a coarser sigma range where a normal-width membrane's response
  drops off. Run `membrane_map` twice on the same patch, once at the existing default (thin-tuned)
  sigmas, once at a coarser range, and compare the two responses along the candidate mask's own
  perimeter (reusing `_perimeter`'s existing boundary extraction): a perimeter that stays strong at
  the coarse scale, not just the thin scale, is a thick-loop candidate.
- Both cues are independent boolean-plus-score outputs; a mask flagged by either is a nucleus
  candidate. No shape/roundness term anywhere, per the explicit warning that real neurites can be
  round too.

## The labeling workflow

One line: add `"nucleus"` to `ERROR_TYPES` in `sam2_utils/labels.py` (currently `("wrong_object",
"under", "over", "bleed", "fragmented", "missing", "other")`). The existing napari review GUI
(`gui.py:1127`) already wires an "error type" `ComboBox` straight to this tuple with no other
dispatch logic keyed to the specific values, verified by checking every other reference to
`ERROR_TYPES` in the codebase (there is exactly one producer and one consumer). Adding the value is
the entire labeling workflow: no new UI code, no new widget. Ground truth for formal detector scoring
then accumulates as a byproduct of normal review sessions instead of a dedicated labeling pass.

## Testing (CPU, torch-free, for the classical fallback only)

`experiments/nucleonet_spotcheck.py` is a spike script, not covered by the pytest suite (matches
`experiments/sam3_probe.py`'s precedent). If the classical fallback is built:

`tests/test_nucleus.py`, synthetic arrays, no frames or torch:

- A uniformly dark, low-variance interior patch: dark-blob cue fires; a patchy, mixed-intensity
  interior at the same mean darkness: it does not (the uniformity term is load-bearing, not the
  darkness alone).
- A perimeter with a genuinely wide ridge (strong response at both sigma scales): thick-loop cue
  fires; a perimeter with a normal-width ridge (strong at the thin scale, weak at the coarse scale):
  it does not.
- A round mask with neither cue true (a real circular neurite stand-in): not flagged, the explicit
  regression test for the "round does not mean nucleus" requirement.

## Docs to update on landing

- `docs/explanation/roadmap.md`: record the spot-check result (item 2e) and which path was taken
  (NucleoNet, or classical, or both if the picture is mixed).
- `docs/CHANGELOG.md` on landing.
- `docs/reference/configuration.md` and `sam2_utils/labels.py`'s own docstring: the new `"nucleus"`
  error type.
- No ADR yet: this spec does not change a mask, a metric, or a production default, matching the
  temporal-projection spec's same reasoning. An ADR is warranted once a detector is wired into a
  live lever.

## Risks and accepted limitations

- NucleoNet's domain generalization to C. elegans is genuinely unknown until the spot-check runs;
  the paper's existence is not evidence of applicability here, only evidence the model is real and
  worth trying cheaply.
- The classical fallback's thresholds (percentile cutoffs, sigma ranges, variance cutoff) are
  eyeballed for v1 against whatever frames the spot-check used, the same "comparative, not
  absolute" caveat every other `membrane.py` detector already carries.
- No formal accuracy number this round for either path. The GUI labeling addition is the mechanism
  that unblocks one, not a substitute for it.
- If NucleoNet is adopted, it becomes a new torch dependency scoped to `experiments/` only until (if
  ever) a later spec promotes it into `sam2_utils/`; it must not silently widen the core library's
  or the CPU-only test suite's dependency footprint before that decision is made deliberately.
