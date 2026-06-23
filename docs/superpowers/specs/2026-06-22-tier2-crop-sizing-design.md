# Tier-2 crop sizing: bigger buffer, collapse default, and a GUI recrop plan

Date: 2026-06-22
Status: Part A landed; Part B Phase 1 (grow-crop recrop) landed; Part B Phase 2
(draw-on-full-frame re-centering) still planned

## Problem

Tier-2 per-chain crops come out too small in two ways:

1. The window is sized from the first-pass `_sam` mask (unioned with the skeleton
   bbox). When that mask under-fills the cell, the window clips the true membrane.
   Even windows that look fine from the skeleton turn out too tight in practice.
2. When the first pass collapsed (no usable mask to size from), the code silently
   falls back to skeleton-only sizing, which can leave a small, off-center window.

A bigger buffer fixes the first; a predictable node-centered default fixes the second.
Separately, when a window is still too small after both, the reviewer wants to recrop
from the napari GUI. That is planned here, not built yet.

## Decisions

From brainstorming:

- The numbers are in `_tif` (full-res) px, the space the existing crop knobs use.
- The 512/side buffer is applied generally (raise `chain_crop_pad_tif`), not only to
  the mask-derived box, because windows that seem fine from the skeleton also clip.
- The collapse default is a fixed window centered on the anchor node.

## Part A: sizing fixes (implement now)

### A1. Raise the general pad

`pipeline/config.py`: `chain_crop_pad_tif` default `64 -> 512`. Every tier-2 window
gets 512 `_tif` px of margin per side. The adaptive `crop_scale` bump (capped by
`chain_crop_max_px`, the input-edge limit) absorbs the larger extent by reading
coarser, the documented coverage-over-resolution trade. No other code changes: the
pad already flows through `chain_crop_window`, and the crop tests pin the pad
explicitly, so they stay green.

### A2. Collapse to a node-centered window

`pipeline/config.py`: new knob `chain_crop_collapse_size_tif: int = 1024` (the
collapse window edge in `_tif` px; 0 disables and keeps skeleton sizing).

`pipeline/crop.py`: new pure helper

```python
def node_crop_window(node_xy_tif, *, size_tif, image_hw_tif, crop_scale, max_px, sam_scale):
    """A fixed size_tif x size_tif CropWindow centered on a node (_tif xy), clipped to
    the frame, with the same adaptive crop_scale guard as chain_crop_window."""
```

It builds a `size_tif`-square box around the node and hands it to
`alignment.CropWindow.around_box`, bumping `crop_scale` coarser if `size_tif` would
exceed `max_px` at the target scale. Exported from `pipeline/__init__.py` like the
other crop functions.

`pipeline/orchestrator.py`, in the `chain_crop_from_mask` block: distinguish two
"no box" cases.

- The masks dir exists but `mask_union_box_px` returns `None`: the first pass
  collapsed. Build the window with `node_crop_window` on the anchor node (looked up in
  `annotate_df` by `state.anchor_node_id`), when `chain_crop_collapse_size_tif > 0`.
- The masks dir is absent (no prior pass at all): keep skeleton sizing, as today.

### Testing

- `node_crop_window`: the window is `size_tif`-square (before clipping), centered on
  the node, clipped to the frame at an edge, and bumps `crop_scale` when `size_tif`
  exceeds `max_px`.
- Re-run the existing `test_chain_crop_from_mask.py` suite to confirm the pad bump
  does not regress the union/skeleton sizing tests.

Torch-free, like the rest of the crop tests.

### Docs

- `docs/reference/configuration.md`: the new default and the new knob.
- `docs/explanation/design-notes.md`: a backlog/notes entry, including the recrop plan
  below.

## Part B: GUI recrop (plan only)

When a tier-2 window is still too small after Part A, let the reviewer recrop from the
GUI. The design reuses the tier-2 machinery rather than reimplementing cropping in the
driver.

1. Thread an `override_crop_window: Optional[CropWindow]` through `run_chain_once` and
   the orchestrator anchor phase. When present, skip the sizing logic and use it. This
   is pure plumbing and also makes the window externally testable.
2. The GUI builds the new window. Two phases, simplest first:
   - Phase 1: a "grow crop by N (`_tif` px)" spinbox (default 512) and a recrop button.
     It grows the chain's current `crop_window` by N per side, clipped to the frame.
     This handles "still too small" without rendering the full frame.
   - Phase 2: show the full-res / `_sam` frame as context and let the reviewer draw a
     rectangle (a Shapes box in `_tif`) for a re-centered window, reusing the
     box-prompt layer already in the GUI.
3. The GUI re-runs in the new window through the existing run path with
   `chain_crop=True` plus the override window: `prepare_chain_crop_frames` rebuilds the
   `_pcrop` view (the slow per-frame full-res windowed read, already has a progress
   bar), the anchor phase re-seeds in the new crop, and propagation, save, and QC write
   a consistent on-disk chain. The GUI then reopens the chain.

This keeps the library/driver boundary intact: the GUI composes pipeline functions and
does not grow its own cropping logic. The cost is the per-frame full-res read on
recrop, which is inherent to changing the window.

## Non-goals

- No change to the `_sam` (non-tier-2) path.
- No change to the `chain_crop_fallback` to `_sam` behavior.
- Part B is not implemented in this change.
