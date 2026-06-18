# 0012. Node-anchored multimask selection

Status: Accepted

## Context

SAM2's mask decoder always computes three candidate masks for a prompt and, by default, the pipeline
takes the single top one. On the cross-worm ground truth the dominant failure is bleed: the chosen
anchor mask grows past the neurite into neighbours, so precision is low and VOI_merge dominates
VOI_split.

The 2025 preprint *Lightweight open-source fine-tuning of SAM2* (Bhat et al.; see
[references/](../../references/)) reports that the correct mask is almost always among SAM2's
candidates even when it is not the highest-confidence one, and recovers it with a biologically informed
selection step rather than trusting the confidence score. Their anchor is a Hoechst nucleus centre:
keep the candidate that contains it. We have no Hoechst stain, but every prompt carries a skeleton
node (the positive seed) and its nearest same-z neighbours (the negatives already fed to SAM2). The
node is our anchor.

## Decision

Select among the three candidates by a lexicographic key, behind `multimask_anchor`:
`(contains the positive node, plausible area, single connected component, SAM IoU)`. The selection is
near-free (the decoder computes all three regardless; only CPU scoring is added) and only moves the
video-seed box, never the seed point.

Add an opt-in anti-bleed term, `multimask_exclude_neg`: among candidates that contain the positive
node, prefer one that contains none of the negative neighbour nodes. A mask that swallows a neighbour's
node is bleeding into it, so excluding negatives is the direct counter to our failure mode. The two
flags are separate so each can be measured against the baseline; with `multimask_exclude_neg` off the
ranking is identical to the original. The `eval` preset turns `multimask_anchor` on.

## Consequences

- The selector reuses the anchor gate's geometry helpers (`_point_in_mask`, `_largest_cc_frac`) and the
  same contain radius and area bounds, so the multimask pick and the gate agree in one space.
- Both flags default off, so existing runs reproduce; the cross-worm baseline (`out_gt_multichain`) was
  single-mask, giving a clean A/B for whether selection, then negative-exclusion, improve the bleed.
- The negatives are the nearest same-z nodes, which on a multi-node neuron could include the same
  neuron's other branch. SAM2 already receives those as negatives, so excluding them in selection
  agrees with the prompt; the decomposed cross-worm chains are mostly one node per slice, where the
  negatives are other neurons.
- Two adjacent ideas from the same paper are out of scope here and tracked in the
  [roadmap](../explanation/roadmap.md): parameter optimization (their search tunes the
  `SAM2AutomaticMaskGenerator`, a mode this pipeline does not use) and mask-decoder fine-tuning.
