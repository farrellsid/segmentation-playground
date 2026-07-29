# Phase 2b.5: temporal membrane projection (adjacent-slice organelle suppression)

Status: design, approved 2026-07-29. Scope: the temporal projection lever only. The
intensity/texture blob filter (also part of roadmap item 2b.5) is deferred to its own spec, informed
by whatever failure mode this lever leaves over.

Outcome (2026-07-29): measured negative, temporal projection does not move the bleed-per-fill floor.
The first gate run overstated the size of the regression due to a gate-membership confound; a
population-controlled re-run and a shift-clamp diagnostic gave a smaller, better-supported picture.
See `docs/explanation/roadmap.md` item 2b.5 and queue item 10, and the CHANGELOG's 2026-07-29 temporal
membrane projection entry, for the numbers.

## Why this, why now

The dense-frame grow-to-membrane sweep (`experiments/dense_membrane_fill.py`, 2026-07-27) found the
scale-8 Sato membrane map (`sam2_utils/membrane.py`) is the ceiling on the whole refinement tier: no
runaway cap or underfill gate moves the bleed-per-fill rate off a ~40% floor, because organelles and
vesicles put false walls and false gaps in the single-slice ridge map. The roadmap's proposed fix
(§5, item 2b.5) is to suppress organelles before ridge-filtering, on the premise that they are
transient across z while membranes are nearly stationary: combine a small window of adjacent
z-slices into one projected intensity image, then run the existing `membrane_map()` on that instead
of a single raw slice.

Two levers were bundled in the roadmap note (the temporal projection, and a separate intensity/
texture filter for dark round blobs). This spec builds and gates the temporal projection alone, so
its effect is not conflated with a second, independent change.

## Scope

In:

- A pixel-registration step for a small window of adjacent-z crops (jitter between target-worm
  sections is unmeasured, so this is built defensively rather than skipped).
- A swappable combine step (median, mean, max, min) that reduces the registered window to one
  projected image.
- Wiring both into `experiments/dense_membrane_fill.py` as a `--mm-window` / `--mm-combine` /
  `--sweep-temporal` opt-in, with `--mm-window 0` reproducing today's behaviour exactly (the control
  group for the sweep).

Out (deferred):

- The intensity/texture blob filter for nuclei/organelles (its own spec, after this lever is
  measured).
- Wiring the temporal projection into `eval/merge_metric.py`'s production `MembraneSource`. That is
  a follow-on once a `(window, combine)` setting is shown to move the gate; premature before that.
- Subpixel registration. Deliberately excluded: interpolation blur works against the goal of keeping
  membranes sharp (the mEMbrain lesson: judge a boundary map on sharpness, not zoomed-out neatness).

## Registration: `register_crops`

```
register_crops(crops: list[np.ndarray], *, max_shift: int = 5) -> list[np.ndarray]
```

A pure function in `sam2_utils/membrane.py`, alongside `membrane_map`. `crops` are same-shape 2D
grayscale arrays, one per z in the window, in z order; the center element (`len(crops) // 2`) is the
registration target. For each other crop:

1. Estimate the integer-pixel translation that best aligns it to the center crop (phase correlation
   on the two crops).
2. Clamp the estimated shift to `[-max_shift, max_shift]` per axis, a safety valve against a bad or
   ambiguous correlation peak (e.g. a low-texture crop) producing a wild shift.
3. Apply the shift with `scipy.ndimage.shift(..., order=0, mode="nearest")`, integer order to avoid
   interpolation blur, edge-replicated borders rather than zero-fill.

Returns a new list, same length and shape as the input, the center crop unchanged.

Built defensively because slice-to-slice jitter on the target worm is unmeasured. Registration is
restricted to the small per-neuron crop (the same bbox+pad window `membrane_map` already runs on,
not the whole ~9k x 9k frame), so the cost is negligible next to the Sato filter itself regardless of
whether jitter turns out to matter.

## Projection: `project_crops`

```
project_crops(crops: list[np.ndarray], *, combine: str = "median") -> np.ndarray
```

A pure function, also in `sam2_utils/membrane.py`. Stacks the (already-registered) crops along a new
axis and reduces with a small dispatch table:

- `median` (default): a pixel dark in a minority of the window's slices is pulled back toward the
  brighter cytoplasm value; a pixel dark in most/all slices (a persistent membrane) stays dark. More
  robust to single-slice noise than `max`.
- `mean`: dilutes a transient dark value but by a similar amount regardless of how many slices it
  appears in, weaker separation between organelle and membrane than `median`.
- `max`: the most aggressive organelle suppression (absent in even one slice erases it), at higher
  risk of eroding a real membrane pixel that reads only faintly dark in one frame.
- `min`: kept in the dispatch table for completeness and future sweeps, but expected to be
  counterproductive here: it preserves whichever slice's darkness is strongest, which *keeps* a
  transient organelle rather than suppressing it.

`combine` is a string key into the dispatch table, not a hardcoded branch, so trying a fifth
statistic later is one dict entry, matching the "signature is the interface" pattern `membrane_map`
already documents for its own future upgrade (a trained model dropping in behind the same call).

## Orchestration (stays in the experiment, not the library)

`sam2_utils/membrane.py` gains no frame-loading or z-indexing logic; both new functions take
already-loaded arrays. The orchestration lives in `experiments/dense_membrane_fill.py`:

1. Before the per-neuron loop, load the whole-frame EM for every z in
   `[args.z - window, args.z + window]` **once** (a `{z: em_gray}` cache), reusing
   `pipeline.load_frame_sam`. Missing z (stack start/end) are simply absent from the cache; the
   window degrades asymmetrically rather than erroring.
2. Inside the per-neuron loop (`grow_all`), for each neuron's bbox+pad window, slice the same
   `(y1:y2, x1:x2)` rectangle out of every cached frame that exists, in z order.
3. If more than one crop was sliced: `register_crops` then `project_crops` on them, producing one
   combined crop; if only one exists (whole window missing, e.g. `window=0` or a stack edge), skip
   straight to it unchanged, so `window=0` is byte-identical to today's single-slice path.
4. Pass the resulting crop to the existing, untouched `mb.membrane_map()`.

This keeps the library side (`sam2_utils/membrane.py`) purely about array math, and the frame-store,
bbox, and z-range bookkeeping where it already lives (the experiment script), consistent with
`CLAUDE.md`'s library/driver import-direction rule.

## CLI and sweep

```
py -3 experiments/dense_membrane_fill.py --mm-window 1 --mm-combine median
py -3 experiments/dense_membrane_fill.py --sweep-temporal
```

- `--mm-window` (int, default 0): window radius in z-slices on each side of center. 0 reproduces
  today's behaviour, the sweep's control row.
- `--mm-combine` (choices: `median` default, `mean`, `max`, `min`): only meaningful when
  `--mm-window > 0`.
- `--sweep-temporal`: grids a small set of `(window, combine)` settings (`window` in `{0, 1, 2}`,
  `combine` in `{median, mean, max}`, `min` available but not in the default grid given the
  above analysis) and prints the same bleed/underfill table `--sweep` already prints, so the gate,
  does the ~40% bleed-per-fill floor drop, reads directly off the table next to the `window=0`
  baseline row.

## Testing (CPU, torch-free)

`tests/test_membrane_temporal.py`, synthetic arrays only, no frames or model:

- `register_crops`: a crop with known content translated by a fixed integer offset is recovered
  (the returned shift, or the aligned output, matches within the synthetic construction); a shift
  larger than `max_shift` is clamped, not applied in full; a single-element list is returned
  unchanged (no-op, nothing to align to).
- `project_crops`: a pixel dark in 1-of-3 synthetic crops (bright in the other two) is pulled toward
  the bright value under `median` and `max` but only partially under `mean`; a pixel dark in all
  three stays dark under every combiner; a single-element list returns that element unchanged for
  every combiner.
- End-to-end smoke: three synthetic crops with a persistent dark ridge and one transient dark blob,
  registered and projected, then run through the existing `membrane_map()`, ridge response stays
  high, blob response drops relative to the same call on the single center crop alone.

## Docs to update on landing

- `docs/reference/configuration.md`: the new `mm-window` / `mm-combine` experiment flags, if the
  experiment scripts are documented there; otherwise a note in the experiment's own docstring is
  sufficient given this is not yet wired into production scoring.
- `docs/explanation/roadmap.md`: record the sweep result against item 2b.5's gate (does the floor
  drop) and either close the item or scope the follow-on intensity/texture filter from what is left.
- `docs/CHANGELOG.md` on landing.
- No ADR yet: this spec does not change a mask, a metric, or a production default. An ADR is
  warranted only if/when the temporal projection is wired into `MembraneSource` as the new default
  membrane source.

## Risks and accepted limitations

- Jitter magnitude on the target worm is unmeasured going in; the defensive per-crop registration is
  the mitigation, not a substitute for actually characterizing it. If the sweep results look odd,
  measuring jitter directly (e.g. logging the estimated shifts) is the first thing to check.
- Phase correlation on a small, low-texture crop can still return a low-confidence estimate even
  within the `max_shift` clamp; accepted for v1, the clamp bounds the worst case rather than
  eliminating it.
- A wider window suppresses organelles harder but risks pulling in real cross-section change for
  fast-tapering neurites; this is exactly what `--sweep-temporal` is for; no window size is assumed
  correct ahead of the sweep.
- This lever alone may not be sufficient. If `--sweep-temporal` shows only a partial improvement,
  that is expected, and the residual is precisely the input the deferred intensity/texture filter
  spec needs.
