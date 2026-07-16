# 0015. Target-worm skeleton merge-metric as the GT-free bleed ruler

Status: Accepted

Refines [0010](0010-erl-voi-eval-ruler.md).

## Context

ADR 0010 adopted ERL and split/merge VOI scored against the cross-worm ground truth. Two facts
surfaced since that made it insufficient for the dominant failure mode, bleed (a mask spilling into a
neighbouring cell):

- The QC flag rule has no bleed signal, so flag-rate A/Bs are blind to it. A whole round of
  negatives and resolution experiments came back flag-neutral while the masks were visibly bleeding.
- The cross-worm GT is doubly biased for boundary accuracy on our copy: it is a *different animal*
  (measures generalization, not in-distribution accuracy) and its VAST masks are *inset from the
  shared membrane by design* (an unpublished, incomplete segmentation), so region IoU and precision
  against it penalize a correctly-membrane-filling mask. The observed ~2 to 3 percent precision was
  largely these two confounds, not only real bleed.

So neither the flags nor the cross-worm boundary metrics could grade bleed on the target worm.

## Decision

Add a ground-truth-free ruler scored against the *target* worm's own CATMAID skeletons, implemented
as `eval.merge_metric`. For each run's RAW per-chain masks (pre non-overlap), per z:

- **foreign-node containment** (a merge): does the mask contain another neuron's skeleton node. This
  is an unambiguous, independently-measured bleed, because a chain is seeded only from its own node.
- **own-node dropout** (an omission): does the mask lose its own node.

Aggregate to per-run `foreign_frame_rate`, `dropout_rate`, and total foreign nodes. This is the
primary bleed/dropout ruler for target-worm A/Bs.

Scope it honestly: it is a **severe-merge floor**, not an ERL benchmark for us. Because every mask is
seeded from its own skeleton, the split/ERL side is partly circular (we were handed the topology), so
ERL numbers are not reported as if from a from-scratch segmenter. And foreign-node containment only
fires when a mask reaches a neighbour's centreline, so mild bleed that stops short is invisible; that
is the Phase 2 membrane map's job.

## Consequences

- A real GT-free verdict on the target worm: the merge-metric graded the negatives and resolution
  rounds the flags could not, showing negatives do cut severe bleed (foreign_frame_rate 0.471 to 0.357
  on the crop baseline) and that full-res crop plus negatives is best on both axes.
- The cross-worm boundary metrics from 0010 are demoted for our copy: topology signals (VOI_merge,
  ERL) remain usable there, but region IoU and precision against the eroded, different-worm GT are
  treated as biased and not headline. 0010's ruler is not wrong, its cross-worm boundary numbers are
  just not trustworthy for our GT copy.
- Mild bleed and boundary accuracy still need a boundary-accurate signal (the Phase 2 membrane map)
  and a small boundary-accurate target-worm benchmark (Phase 3). The staged plan and the erosion
  finding are in the [roadmap](../explanation/roadmap.md); the CLI is documented in
  [cli.md](../reference/cli.md).
