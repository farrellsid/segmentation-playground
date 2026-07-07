# Mask-seeded propagation and a reusable coarse pass

Status: design, not yet implemented. Split out of the tiling spec because it is a broad,
cheap win on its own and the tiling design consumes it.

## Problem

The video predictor is seeded from a box: the anchor image-predict produces a mask, we take
its largest connected component, and reduce it to a bounding box (`box_from_mask`) that seeds
`propagate`. The box throws away the mask's shape, and a box is a weaker prompt than the mask
it came from. The earlier research favored mask and box prompts over points, and a mask is the
richest seed available.

Two needs come together here:

1. Seed the propagation from the anchor *mask* itself, not just its bounding box.
2. Let a later, higher-resolution pass reuse a chain's already-saved coarse mask as its seed,
   so the coarse pass runs once and refinements do not repeat it.

## What already exists

`PropagationSession.add_mask` seeds a frame with a mask; the review GUI's resume path uses it,
so the capability is proven. Chains already checkpoint: `save_masks` writes the per-frame PNGs
and `state.json`, and the batch resume skips finished chains. So a chain's coarse mask is
already a persisted, reusable artifact; nothing new is needed to store it.

## Design

Two config knobs on `PipelineConfig`, both defaulting to today's behavior:

- `seed_mode: str = "box"` -> `"box"` (current), `"mask"`, or `"mask+box"`. In `"mask"` and
  `"mask+box"`, after the anchor image-predict returns its mask, seed the video with that mask
  via `add_new_mask` (plus the box in `"mask+box"`) instead of box-only. The largest-CC step is
  kept for the mask so a stray blob does not seed a second object.
- `seed_from: str = "anchor_predict"` -> `"anchor_predict"` (current) or `"saved_masks"`. In
  `"saved_masks"`, the run reads a chain's saved masks from a named prior run and seeds the
  anchor frame from that mask rather than re-predicting it. This is what a refinement or tiling
  pass uses to skip the coarse pass.

The seeding change is confined to the phase that builds the video seed; propagation, QC, save,
and aggregation are unchanged.

## Components and isolation

- `predict.py` / `propagate.py`: the seed-construction path gains the mask branch.
- A small loader that returns a chain's saved anchor mask (reusing `qc._load_binary` and the
  `crop_window` remap already in `crop.chain_masks_in_sam`), so "read a saved mask" stays one
  definition.
- `orchestrator.py` wires `seed_mode` / `seed_from` into the seed step.

## Testing

- Pure seed-selection: given an anchor mask, `seed_mode="mask"` produces a mask seed and
  `"box"` produces the current box, torch-free.
- Round-trip: seeding the anchor frame from a known saved mask reproduces that mask at the
  anchor (the propagation itself needs a GPU, so it is a smoke, not a unit test).

## Scope

v1: `seed_mode="mask"` from the anchor image mask, the broad cheap win, measured against GT vs
the box baseline. The `seed_from="saved_masks"` path is the hook the coarse-to-fine tiling
design consumes; it lands with tiling if not before.
