# 0008. Box-plus-point seed for auto, mask for the GUI

Status: Accepted

## Context

The video predictor needs a seed on the anchor frame. The options are a bounding box, positive and
negative points, a mask, or combinations. A box delegates making the anchor mask to SAM2's decoder. A
mask seed bakes a curated boundary into memory, which propagates faithfully when the boundary is
right but propagates the error when it is slightly wrong. SAM2 also makes a mask seed mutually
exclusive with points and box on the same frame, so "mask plus points" is not a real option.

## Decision

For the automatic batch, seed with a box plus a positive point. A seed ablation ranked this best;
points-only was worst, and a mask-only seed did not beat the box at the scale-8 anchor, where the
mask is a coarse few-pixel blob no more informative than its box.

For the GUI, drop the box and seed with the mask. There a human paints a high-quality anchor, which
is the one regime where the mask seed wins.

Box-from-radius is rejected: the CATMAID radius column is mostly placeholder values.

## Consequences

- The two seed choices are consistent, not contradictory: keep the box where the anchor is rough and
  automatic, use the mask where the anchor is human-verified.
- Negatives in the seed help cluttered or concave chains and hurt clean ones, so they are a targeted
  knob, default off.
- A confidence-gated mask-versus-box seed (use the mask when the anchor is trustworthy, else the box)
  is the natural extension and is co-built with the GUI's human-anchor path.
