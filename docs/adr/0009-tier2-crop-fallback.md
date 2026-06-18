# 0009. Tier-2 per-chain crop with image-score fallback

Status: Accepted

## Context

A neurite is only a few pixels wide at the scale-8 SAM2 grid, which limits mask quality. The anchor
crop (tier 1) sharpens only the one-frame seed; it cannot change propagation resolution. To get
genuinely higher-resolution masks, the whole chain has to propagate at higher resolution.

The risk found in testing: a low-motion chain with a tiny xy extent produces an over-zoomed window
where SAM2 loses inter-frame context and propagation collapses to empty masks. The over-zoomed anchor
still passes the geometry gate (it is a clean blob that contains the node), so a geometry check does
not catch the failure.

## Decision

Add a tier-2 per-chain crop (`_pcrop`): size one window to the chain's extent and propagate the whole
chain inside it at higher resolution. Guard it two ways. Floor the window extent so a low-motion
chain cannot over-zoom. Add a per-chain fallback that reverts to the scale-8 path when the crop
anchor's SAM2 image score is below a threshold, since the image score (not the geometry) is the
signal that discriminates the over-zoom collapse.

Keep tier-2 off by default globally, but turn it on automatically as a second pass for chains the
first pass flagged. A wider A/B improved three chains, regressed none, and cut the queue, with the
fallback firing on the chains whose crop anchor was poor.

## Consequences

- Flagged chains get higher-resolution masks where it helps, and revert cleanly where it does not.
- Tier-2 chains store masks in `_pcrop` and persist the crop window, so QC, the viewer, and the GUI
  rebuild the space. See [0006](0006-canonical-mask-space.md).
- A tier-2 chain that still flags is kept, not reverted: the human gets a crisp paint surface in the
  GUI.
- The fallback threshold is a target-worm default. On the cross-worm ground truth it was found
  mis-calibrated (crop anchors scored just under the floor), which is a known tuning item in the
  [roadmap](../explanation/roadmap.md).
