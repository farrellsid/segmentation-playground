# 0016. Membrane map and border-to-border bleed detection

Status: Accepted

Refines [0015](0015-target-worm-merge-metric-ruler.md).

## Context

ADR 0015 gave the project its first trustworthy ruler on the target worm: the skeleton merge-metric,
which counts foreign skeleton nodes contained in a raw mask. That ruler is honest about its own
limit, it is a **severe-merge floor**: foreign-node containment only fires once a mask reaches a
neighbouring cell's centreline. Two large classes of error sit below that floor and were invisible to
every A/B run against it:

- **Mild bleed.** A mask crosses a real membrane into the neighbouring cell but stops short of that
  neighbour's node. The Phase-1 negatives/resolution round could not be graded on this axis at all,
  it came back flag-neutral and merge-metric-neutral while some masks were visibly bleeding.
- **Underfill.** A mask covers only part of its own cell, stopping short of the membrane that should
  bound it. Neither the QC flag rule nor the merge-metric has any opinion on this.

Both need a per-pixel boundary signal read from the raw EM. Nothing built so far reads the EM inside
scoring, so this is new ground, not a refinement of an existing feature.

## Decision

Add a ground-truth-free membrane map and three detector primitives that grade a mask against it,
implemented as `sam2_utils/membrane.py`, plus a `MembraneSource` loader and membrane-aware scoring
wired into `eval/merge_metric.py`. Design: `docs/superpowers/specs/2026-07-17-phase2-membrane-map-bleed-detection-design.md`.

- **`membrane_map(em_patch)`**: a per-pixel membrane-ness map in `[0, 1]`, v1 a classical dark-ridge
  filter (Sato, black ridges), normalised by its own 99th percentile so a fixed threshold stays
  stable frame to frame. The signature is the interface: a trained model can drop in behind it later
  without touching the detectors or the scorer.
- **`spanning_membrane(mask, mem)`**: the primary mild-bleed detector. Strip membrane pixels from the
  mask, label what is left, and check whether two or more of the surviving regions each touch the
  mask's own border. Two border-touching regions mean a membrane ridge cut the mask in two,
  border to border, so the mask engulfed a real cell boundary.
- **`boundary_on_membrane(mask, mem)`**: the fraction of the mask's perimeter that sits within
  tolerance of a membrane pixel. A direction-agnostic boundary-quality check, low on both bleed and
  underfill.
- **`underfill_fraction(mask, mem)`**: a bounded outward flood from the mask through non-membrane
  pixels, reported as a fraction of the mask's own area. High means there was room to grow before
  hitting a membrane, i.e. the mask stopped short.

**Why border-to-border, not "any membrane inside the mask."** The naive rule, flag a mask that
contains any membrane pixel at all, breaks on the nucleus. A soma's nucleus is a real, dark-ridged
membrane sitting entirely inside the cell, and it is a *closed interior loop*, not a ridge that spans
the mask. Stripping it away leaves one region touching the mask's border (the cytoplasm) and one
enclosed region that touches nothing (the nucleus interior), so only one region touches the border
and the mask is not flagged. Requiring two or more *border-touching* survivors is what makes the soma
case fall out for free, with no special-cased nucleus detection anywhere in the code. This is also
why the roadmap's nested-membrane ceiling (a soma's nucleus stealing the point prompt) does not
double-count as a false bleed positive here: the two problems are independent, and this detector does
not conflate them.

## Consequences

- **Comparative, not absolute, at the `_sam` grid.** The scale-8 working resolution blurs thin gaps
  between adjacent neurites, so a broken ridge can leak a false spanning read or overestimate
  underfill. Treat all four scalars as comparators across runs of the same worm at the same scale,
  not as an absolute boundary ruler. A finer membrane source, a higher-resolution crop or a trained
  model, is the documented upgrade path behind the same `membrane_map` signature.
- **Underfill is the lowest-confidence of the three.** It is the most sensitive to a broken ridge,
  because a leak lets the flood spread into a neighbour rather than stopping at the true boundary.
  The bounded flood radius keeps a leak local, but the number should be read as a lead, not a verdict,
  until the finer signal lands.
- **The headline is `mild_bleed_rate`**: the fraction of scored frames with a spanning membrane and
  no foreign node, mild bleed that the Phase-0 floor could never see. `spanning_merge_rate`,
  `mean_boundary_on_membrane`, and `mean_underfill_fraction` round out the summary; all four are
  `None` when the membrane pass is unavailable or skipped, so a caller can tell "not measured" apart
  from "measured as zero."
- **This is measurement only.** Nothing here changes a mask. Two follow-on uses are deliberately
  deferred to their own specs so they can be measured against this same ruler once it exists:
  **2c**, grow-to-membrane refinement, which would apply the same flood that 2b only measures; and
  **2d**, non-overlap arbitration, replacing the composite's first-writer-wins with a membrane-aware
  resolve. Both reuse this signal rather than inventing a second one.
- The Phase-0 severe-merge floor from ADR 0015 is unchanged and still the primary bleed ruler for
  the cases it does catch; this ADR extends the ruler downward into the mild-bleed and underfill
  range it was always honest about missing.
