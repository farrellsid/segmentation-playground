# GUI box prompts for the image phase

Date: 2026-06-22
Status: approved, ready to implement

## Problem

The review GUI only lets the human seed a re-prediction with point prompts (the
`prompts` Points layer). SAM2's image predictor also takes a bounding box, which
often captures a thin neurite's full extent better than a few clicks. The data model
already carries a box (`pipeline.Prompts.box_sam`, xyxy in `_sam` space) and the
propagation seed already accepts one, but `image_predict` ignores it, so a human has
no way to draw a box and have it affect the re-predicted mask.

## Goals

- Let the reviewer draw a bounding box on the current frame.
- Feed that box into the image-phase re-predict (`R`) together with any point prompts.
- Pre-load the chain's saved box on open, the way point prompts pre-load.

## Non-goals

- The box does not become a direct propagation seed. Resume propagation (`G`) still
  seeds with the mask, as today. The box only shapes the image-phase mask.
- No multiple boxes per frame, and no box reset control. Reset-prompts stays
  points-only; clear a box by deleting the rectangle in the Shapes layer.

## Decisions

Settled during brainstorming:

- A drawn box feeds `image_predict` alongside points (not box-only, not a propagation
  seed).
- The saved box (`state.prompts.box_sam`) pre-loads at the anchor frame on open.
- Reset-prompts stays points-only.

## Design

### Pipeline: `image_predict` forwards the box

`pipeline/predict.py:image_predict` currently passes only `point_coords` and
`point_labels` to `image_predictor.predict`. Forward `prompts.box_sam` as the `box`
argument, and pass `point_coords=None`/`point_labels=None` when there are no points so
a box-only call is valid:

```python
has_pts = len(pts) > 0
box = None if prompts.box_sam is None else np.asarray(prompts.box_sam, dtype=float)
image_predictor.predict(
    point_coords=pts if has_pts else None,
    point_labels=labs if has_pts else None,
    box=box,
    multimask_output=multimask,
)
```

This is a no-op for the batch: the orchestrator derives `box_sam` from the predicted
mask (`box_from_mask`) *after* `image_predict`, so `box_sam` is None at predict time
and the call is byte-for-byte the old one. The multimask selection path is unchanged
(the GUI re-predict uses `multimask=False`).

### GUI: a "box" Shapes layer

In `gui.py`:

- A napari Shapes layer `box`, `ndim=3`, with the same `scale` as the prompts/mask
  layers, so a drawn rectangle's data coordinates are already `_sam` (or `_pcrop` for
  a tier-2 chain), needing no transform, exactly like the point prompts.
- A `draw box (B)` button and a `b` key (the only free single key) activate the box
  layer in `add_rectangle` mode. One box per frame; the last drawn wins.
- Three module-level pure helpers, torch-free and napari-free, so they unit-test
  without a GPU or a viewer:
  - `_rect_to_xyxy(verts)`: a rectangle's `(N, 3)` vertices `(t, y, x)` to
    `(x0, y0, x1, y1)`.
  - `_xyxy_to_rect(xyxy, t)`: the inverse, a `(4, 3)` vertex array at frame `t`.
  - `_box_on_frame(shapes_data, frame_idx)`: the xyxy of the last rectangle on a
    frame, or None.
  `_box_for_frame(frame_idx)` calls `_box_on_frame(self._box.data, frame_idx)`.
- `_seed_prompts_from_state` also pre-loads `state.prompts.box_sam` as a rectangle at
  the anchor frame.
- `rerun_image_phase` sets `prompts.box_sam = self._box_for_frame(frame_idx)` and
  relaxes its guard to "at least one positive point or a box on this frame".

### Coordinate spaces

The box layer shares the per-chain `_sam`-to-EM-world `scale` with the prompts, mask,
and skeleton layers, so a rectangle drawn on the canvas has `_sam` (or `_pcrop`) data
coordinates. That is the space `image_predict` expects, since the GUI re-predict runs
on the displayed frame the human drew on, the same round-trip the point prompts use.

## Testing

- A new test for the three pure box helpers: xyxy-to-verts-to-xyxy round-trip, the
  last-wins rule, frame filtering, and the empty case. Imports `gui` (torch and napari
  load lazily, so the module imports without either), matching `test_anchor_select`.
- An `image_predict` box-forwarding test with a fake predictor that records the
  `box` and `point_coords` keyword arguments: box-only passes `point_coords=None`,
  box-plus-points passes both, and no box passes `box=None`. Skipped when torch is
  absent, since `image_predict` enters a `torch.inference_mode()` block.

## Docs to update in the same change

- `docs/how-to/review-flagged-chains.md`: the layer table and a short box paragraph.
- The `gui.py` module header note that says the box seed was dropped: clarify that the
  box is now an image-phase prompt while resume still seeds with the mask.
