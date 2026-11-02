# Phase 2 foundation: membrane map + membrane-aware bleed detection

Status: design, approved 2026-07-17. Scope: the foundation of roadmap Phase 2 only.
Refinement (grow-to-membrane) and non-overlap arbitration are deferred to their own specs.

## Why this, why now

Phase 0 gave the project its first trustworthy ruler on the target worm: the skeleton
merge-metric (`eval/merge_metric.py`), which counts foreign skeleton nodes contained in a raw
mask. That is a severe-merge floor. It only fires when a mask swallows a *neighbour's centreline*,
so it is blind to two large classes of error the Phase-1 A/B surfaced: mild bleed that crosses a
membrane into a neighbour but stops short of that neighbour's node, and underfill where a mask
covers only part of its own cell.

The roadmap's Phase 2 answer is a per-pixel membrane signal read from the raw EM. This spec builds
the foundation of that phase and nothing more: the signal itself, and the detection metrics that
grade masks against it. It deliberately builds a signal and a ruler before any lever that changes a
mask, because the project's spine is that you cannot fairly choose a lever until you can measure
one. Refinement and arbitration reuse this signal later, each in its own spec, once this metric can
grade them.

## Scope

In:

- 2a: a ground-truth-free per-pixel membrane signal from the raw EM, behind a swappable interface.
- 2b: membrane-aware detector scalars that extend the Phase-0 merge-metric, scored on the same raw
  per-chain masks, in the same `_sam` grid, written to the same CSV.

Out (each its own later spec):

- 2c grow-to-membrane refinement (changes masks).
- 2d mutex-watershed / multicut non-overlap arbitration (changes the composite labelmap).
- Any trained membrane model (U-Net), and any reuse of the prior lab pipeline's model. v1 is
  classical and training-free; the trained version rides the same interface later.

Nothing in this spec changes a mask. Both deliverables are measurement only.

## 2a: the membrane map

A pure function in a new library module `sam2_utils/membrane.py`:

```
membrane_map(em_patch: np.ndarray, *, sigmas=..., ...) -> np.ndarray  # float32 in [0, 1]
```

- Input is a grayscale EM patch (a crop of one frame at the run's `_sam` scale). Output is a
  per-pixel membrane-ness map normalised to [0, 1], same height and width as the input.
- v1 is a classical dark-ridge filter: a Sato / Frangi-style Hessian ridge response tuned for dark
  membranes on the target worm's contrast, or the EM intensity gradient, whichever scores cleaner
  on a hand-checked frame. skimage and scipy are already dependencies; no torch, so the module and
  its tests stay CPU-only and torch-free.
- The signature is the interface. A trained U-Net or the reused prior-pipeline model drops in
  behind the same call later without touching 2b.

Placement rationale: the map lives in the library (`sam2_utils/`, alongside `qc.py` and
`alignment.py`), not in `eval/`, because the later refinement (2c) and arbitration (2d) work lives
in `pipeline/` and will reuse this exact signal. `eval/` importing the library is allowed;
the library importing `eval/` is not (enforced by `tests/test_import_direction.py`), so the reusable
signal must sit on the library side.

Caching: v1 memoises maps in memory per run only (a small per-z cache inside the scorer). It does
not persist maps to disk. The code module itself is tiny. If a future version caches rasters to
disk, they go under `frames_root` (on F:), never inside the repo.

Honest limitation, stated up front: at the `_sam` scale (typically 8) membrane-ness is coarse, so
thin gaps between adjacent neurites can blur shut. v1 is therefore a relative comparator across
runs, not an absolute boundary ruler. A finer membrane source (a higher-res crop, or a trained
model) is the documented upgrade path and rides the same interface.

## 2b: the detectors

All detectors operate per-frame per-chain on the raw masks that `pipeline.chain_masks_in_sam`
already returns as `(mask, x0, y0)` in the `_sam` grid, against `membrane_map` of the matching EM
patch. They are pure array functions (mask + membrane map in, scalars out), so they are
unit-testable without frames or torch. Detector primitives live in `sam2_utils/membrane.py`
next to the map; the scorer in `eval/` wires them to frames.

Let `M` be the membrane map for a mask `R`, and `B_m = M > tau` the binary membrane.

### A. Interior spanning-membrane (primary, the mild-merge signal)

A correct mask covers one compartment, so a strong membrane should sit on its border, never run
through its interior. Procedure:

1. Remove membrane from the mask: `R_open = R & ~B_m`.
2. Label connected components of `R_open`; keep components with area >= `f * area(R)`.
3. If two or more kept components each touch `R`'s outer border, a membrane spans the mask
   border-to-border: the mask engulfed a cell boundary. `spanning_merge = True`.
4. Graded output `bled_fraction` = area of the smaller border-touching kept component / area(R).

Soma handling falls out of the border-to-border criterion with no special case: a nucleus is a
*closed* interior loop, so removing it leaves one border-touching cytoplasm region plus one
*enclosed* region that does not touch `R`'s border. One border-touching region, so not flagged.
That is precisely the nested-membrane case the roadmap warns about, handled by construction.

### B. Boundary-on-membrane fraction (secondary, boundary quality)

Fraction of the mask perimeter within tolerance `t` px of a membrane pixel:
`boundary_on_membrane = |perimeter(R) & dilate(B_m, t)| / |perimeter(R)|`. Low means the edge floats
through cytoplasm, which is the shared signature of both leaking bleed and underfill, so B is
direction-agnostic and sits alongside A and C rather than replacing either.

### C. Underfill fraction (third scalar, lowest confidence)

A `k`-bounded outward geodesic flood from the mask, membranes as walls: the cytoplasm area
reachable from `R` within `k` px without crossing `B_m`, as a fraction of area(R). High means the
mask stopped short of its enclosing membrane, i.e. there was room to grow, i.e. underfill.

Two caveats stated in the code and the docs:

1. Most gap-sensitive of the three at coarse `_sam`: a broken ridge lets the flood leak into a
   neighbour and overestimate underfill. The `k` bound keeps a leak local rather than global, but
   this scalar improves the most when the finer membrane source arrives, so it ships labeled lowest
   confidence.
2. Closest to refinement: 2c *applies* this grow-to-membrane flood; 2b only *measures* the gap. The
   scope line stays clean. We report the number, we never write the flooded mask back.

Parameters `tau`, `f`, `t`, `k` are resolution-aware defaults, chosen against a couple of
target-worm frames and exposed on the CLI. The metric is comparative across runs, not an absolute
score.

## Integration

Extend `eval/merge_metric.py`; keep every Phase-0 column and the existing node-only behaviour intact.

New per-frame columns joined onto the existing per-frame DataFrame: `spanning_merge` (bool),
`bled_fraction` (float), `boundary_on_membrane` (float), `underfill_fraction` (float). When the
membrane signal is unavailable (see graceful degradation), these are `NaN`.

New summary quantities, alongside the Phase-0 ones:

- `mild_bleed_rate` (headline): fraction of frames with `spanning_merge` True but zero foreign
  nodes. This is exactly the error class Phase 0 is blind to, expressed as one number.
- `mean_boundary_on_membrane`, `mean_underfill_fraction`, `spanning_merge_rate`.

### The EM patch, the one real dependency

Phase 0 never read frames; 2b must. A small `MembraneSource` in `eval/` owns it: given
`(z, x0, y0, h, w, scale)` it loads the cached frame `frames_cache_s{scale}/z{file_z}.jpg`
(`alignment.catmaid_z_to_file_z` maps `z`), crops to the mask's `(x0, y0)` offset, calls
`membrane_map`, and memoises per z. `frames_root` defaults to `config.FRAMES_ROOT`.

Graceful degradation is a hard requirement: if `frames_root` is absent, or a frame or its window
is missing, the membrane columns are `NaN` and the tool still emits the full Phase-0 metric
unchanged. The existing headless scorer must never regress because frames are not co-located with a
merged tree.

## CLI and output

```
py -3 -m eval.merge_metric --root <tree> [--root ...] \
    [--frames-root <dir>] [--no-membrane] [--tau <m>] [--tol <t>]
```

`--no-membrane` forces the Phase-0-only path (fast, no frame reads). Output is the same
`_merge_metric.csv` per tree with the new columns; `format_summary` gains membrane lines. Retro
scoring the four Phase-1 A/B trees (`tier2_s1forced_neg`, `generous_only`, `perslice_only`,
`perslice`) then grades their *mild* bleed and underfill, not just node hits, in one command.

## Testing (CPU, torch-free)

`tests/test_membrane_metric.py`, all synthetic arrays, no frames or torch:

- A straight ridge across a square mask: `spanning_merge` True, `bled_fraction` ~ the smaller side.
- A closed nucleus loop inside a mask: `spanning_merge` False (the soma guard, by construction).
- A boundary drawn on a ridge: high `boundary_on_membrane`; a boundary in blank cytoplasm: low.
- A mask inset from a surrounding ring of membrane: `underfill_fraction` high; a mask filling to the
  ring: low.
- A missing frame: membrane columns `NaN`, Phase-0 columns intact (the graceful-degradation path).
- `membrane_map` smoke test: a hand-built patch with one dark ridge yields a high response along the
  ridge and low response off it, output in [0, 1] with the input's shape.

## Docs to update on landing

- `docs/reference/configuration.md`: the new metric parameters (`tau`, `f`, `t`, `k`).
- `docs/reference/cli.md`: the new `merge_metric` flags.
- `docs/reference/code-map.md`: `sam2_utils/membrane.py` as the home of the membrane signal and the
  detector primitives.
- `docs/explanation/roadmap.md`: mark Phase 2's foundation (2a + 2b) landed; note 2c refinement and
  2d arbitration still queued.
- A new ADR for the border-to-border spanning criterion: it is the load-bearing choice that makes
  the soma case fall out for free, and a newcomer would question why "membrane inside the mask" is
  not itself the merge signal.
- `docs/CHANGELOG.md` on landing.

## Risks and accepted limitations

- Threshold calibration is eyeballed for v1. Defaults are picked against a few target-worm frames;
  the metric is comparative, not absolute. Accepted for v1.
- Coarse `_sam` membrane blurs thin inter-neurite gaps, undercounting mild merges and inflating
  underfill. Accepted; the finer-membrane upgrade rides the same interface.
- Tier-2 chains are downsampled to `_sam` before scoring (matching `chain_masks_in_sam`), losing
  their native resolution for this metric. Consistent with Phase 0. Noted.
- `underfill_fraction` is the least reliable scalar at v1 (gap-sensitive) and overlaps 2c
  conceptually. Shipped labeled lowest confidence, measured only, never applied.
