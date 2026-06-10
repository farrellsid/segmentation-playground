# eval/ — Stage 0 evaluation harness (scaffold, not yet implemented)

**This is an empty home for the Stage 0 work. No metric code lives here yet.**

The evaluation harness is the gate everything else waits on: per
[`../FUTURE_DIRECTIONS.md`](../FUTURE_DIRECTIONS.md) §5 **Stage 0** and §4.1, we must *fix the ruler
before any further accuracy tuning*. Flag-rate (what every A/B to date leaned on) is being
deprecated as an A/B metric.

## What goes here (when built)

A harness that scores the **current pipeline's output against the cross-worm ground truth** — the
confirmed segments of the new worm, with matching EM, obtained in the June 2026 step-back. It should
produce, per neuron:

- **Expected Run Length (ERL)** — expected error-free traced length from a random point, with any
  merged segment assigned zero length. Skeleton-based, and CATMAID skeletons already exist, so it's
  essentially free to compute. (FUTURE_DIRECTIONS §4.1; Januszewski et al., Nature Methods 2018.)
- **Variation of Information split into VOI_split + VOI_merge**, so mergers can be weighted more
  heavily than splits (mergers are far costlier to fix by hand). Start with a **merge:split cost
  ratio of ~5:1 or higher**. (FUTURE_DIRECTIONS §4.1.)

Inputs: confirmed-segment masks + CATMAID skeletons (the confirmation markers say which segments are
manually verified); landing spot for the GT is [`../data/groundtruth/`](../data/groundtruth/).

**Advance gate (Stage 0 → Stage 1):** the harness can produce a per-neuron ERL and a split/merge
breakdown on the confirmed segments.

> Caveat (FUTURE_DIRECTIONS §3, §7): the cross-worm GT measures **generalization**, not
> in-distribution accuracy — treat it as a domain-adaptation benchmark and spot-check on the target
> worm.

See [`../FUTURE_DIRECTIONS.md`](../FUTURE_DIRECTIONS.md) §4.1 and §5 Stage 0 for the full reasoning
and sources.
