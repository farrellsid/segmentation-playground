# 0010. ERL and split/merge VOI as the eval ruler

Status: Accepted

## Context

For most of the project the only quality signal was the QC flag rule, and every A/B was scored on
that same rule. The flag rule is known to be unreliable, so the project was tuning against a noisy
yardstick. Obtaining cross-worm ground truth (a different worm with matching EM and confirmed
segments) made a real measurement possible for the first time.

## Decision

Adopt the connectomics-standard metrics as the ruler and deprecate flag-rate as an A/B metric. The
primary measures for this sparse, per-neuron pipeline are per-neuron region IoU, precision, and
recall, plus Expected Run Length (ERL): the error-free traced distance from a random point, with any
segment that contains a merge assigned zero length. ERL is skeleton-based and 3D, so it fits a subset
of neurons naturally and reuses the CATMAID skeletons the project already has.

Compute split/merge VOI and ARAND as well, but treat them as secondary here. They are built for
dense whole-volume segmentation; on a sparse scored-neuron subset, VOI_merge is blind to bleed into
unscored neighbours and the numbers are not comparable to dense benchmarks. They become primary only
with a dense labelmap.

## Consequences

- A/B comparisons get a real ruler: per-neuron ERL and a split/merge breakdown against confirmed
  segments, weighting merges more heavily than splits.
- The ruler depends on the skeleton-to-GT registration, which both places prompts and samples node
  labels, so a loose registration poisons every number. Verifying the registration is the gate
  before trusting any score.
- The cross-worm GT measures generalization, not in-distribution accuracy, so gains are spot-checked
  on the target worm. The full reasoning, the staged plan, and the sources are in the
  [roadmap](../explanation/roadmap.md); the harness is in `eval/`.
