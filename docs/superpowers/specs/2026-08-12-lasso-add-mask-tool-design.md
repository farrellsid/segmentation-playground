# Lasso-add: a freehand extend tool for mask correction in gui.py

Status: design, approved 2026-08-12.

## Why

Today's mask correction in `gui.py` is a plain napari Labels-layer paint/erase brush on `self._mask`
(gui.py:475). That is fine for local touch-ups but tedious for the most common real correction: a
long stretch of boundary that should extend further (a neurite that should reach past where the
automatic pipeline stopped it). Extending a long stretch by brush means repainting the whole new
strip by hand.

The user described a "paper.io"-style tool seen elsewhere: draw one open curve near the edge of your
existing territory, and the enclosed area between your curve and the boundary fills in, no need to
retrace the part of the edge that did not change.

Two things narrowed the scope during design:

- The correction need is specifically long boundary stretches, not local blobs (a plain closed loop
  you draw and union in already covers local blobs fine on its own).
- Only addition is needed. Subtraction already has a tool: the existing eraser brush.

Addition-only removes the hardest part of a literal paper.io port. A true port needs the drawn curve's
two ends snapped onto the existing mask boundary (via contour matching), then a choice between the two
resulting boundary arcs to close the loop, both real sources of ambiguity (which arc, what snap
tolerance, what to do with multiple mask components on one frame). None of that is needed for
addition: a plain closed freehand loop, unioned into the mask (`mask |= fill(loop)`), already
reproduces the desired result whenever the loop starts and ends on or near the existing mask, because
union is a no-op wherever the loop overlaps mask that is already there. Whatever closes the loop
(napari's own straight snap-back-to-start segment) only ever passes through already-covered area in
the normal case of extending along one stretch of edge, so it adds nothing unwanted.

## Interaction

`l` arms a new `self._lasso` napari Shapes layer in its native `add_polygon_lasso` mode (confirmed
present in the installed napari 0.7.0, `napari.layers.shapes._shapes_constants.Mode.add_polygon_lasso`,
via `napari.layers.shapes._shapes_mouse_bindings.add_path_polygon_lasso`). This mirrors the existing
`b` keybind that arms `self._box` for a box draw (gui.py:1298-1299, `_new_box_layer`).

Click-drag a freehand loop that starts on the visible mask, bulges out to cover the new area, and ends
back on the mask. Releasing the mouse auto-finishes the polygon; napari's own lasso implementation
finishes automatically at mouse-up whenever the drag accumulated enough vertices to be treated as a
continuous stroke rather than a click (confirmed from source: `if len(vertices) > 2:
layer._finish_drawing()` inside the drag generator). No separate confirm keypress. Each stroke commits
immediately, so multiple strokes can be drawn in sequence to build up a longer extension.

`self._lasso` is created alongside `self._prompts`/`self._box` in `open_chain` (gui.py, near line
480-481), using the same `scale=lscale` convention so a drawn loop's data coordinates land in the same
`_sam`/`_pcrop` pixel space as the mask, matching how `self._box`'s rectangle already round-trips
through `_box_for_frame`.

## Commit logic

A callback on `self._lasso.events.data` fires whenever napari appends a finished polygon. On each
fire:

1. Read the newly added shape's vertices for the current frame (`self.viewer.dims.current_step[0]`),
   in the same `(t, y, x)` convention `_prompts_for_frame`/`_box_for_frame` already use, dropping the
   `t` column.
2. If fewer than 3 vertices, treat as a degenerate stroke (a stray click, not a drag) and discard
   without touching the mask.
3. Rasterize the polygon interior into a boolean `(H, W)` array at the mask's own resolution via
   `cv2.fillPoly` (already a project dependency, used in `pipeline/crop.py`).
4. Union that array into the current frame's labels through `self._mask.data_setitem(indices,
   self.data.obj_id)`, not a raw `self._mask.data[...] =` write. `data_setitem` is the Labels layer's
   own public API for a programmatic edit that still records an undo step (confirmed present on
   `napari.layers.Labels` in the installed version), so `Ctrl+Z` reverts one lasso stroke exactly like
   it already reverts one paint stroke, no separate undo bookkeeping needed here.
5. Remove the just-committed shape from `self._lasso` (`self._lasso.data = self._lasso.data[:-1]` or
   equivalent), so the layer stays visually empty and ready for the next stroke rather than
   accumulating drawn loops on screen.

## Edge cases (handled by construction, not special-cased)

- **No existing mask on this frame.** Union with an empty mask is just the loop's own interior, so
  the tool doubles as a from-scratch add on a frame with nothing painted yet. Not the primary use
  case, but a natural side effect worth keeping rather than guarding against.
- **A self-crossing loop.** `cv2.fillPoly` fills by its own standard rule; the result may not exactly
  match what a self-crossing stroke "meant," but it will not crash or corrupt the mask. Not worth
  detecting or rejecting for a first version.
- **A loop that does not touch the existing mask at all** (drawn somewhere else in the frame
  entirely). Still just unions in a new, disconnected patch. No special handling, and no boundary
  snapping means there is nothing that could reject this case even if it seemed worth rejecting.

## Explicitly not building

Boundary/contour snapping, arc selection between two candidate closures, a cut/subtract mode, or a
modifier key to choose between add and cut. Subtraction stays on the existing eraser brush, per the
user's own scoping call during design.

## Testing plan

- A CPU-only unit test in `tests/` builds a small synthetic `(H, W)` boolean mask and a synthetic
  lasso polygon (a plain list of `(y, x)` vertices), calls the rasterize-and-union step directly
  (extracted as its own testable function rather than only living inside the napari event callback),
  and asserts the resulting mask equals the expected union. No napari `Viewer` needed for this part,
  matching the project's existing CPU-only test convention.
- A degenerate-stroke test (fewer than 3 vertices) asserts the mask is unchanged.
- A manual smoke test in the live GUI against a real chain: draw a real stroke, confirm the mask
  extends as expected, confirm `Ctrl+Z` reverts exactly that stroke and nothing else. This step cannot
  be automated (napari's real mouse-drag event stream), so it stays a manual pre-ship check rather
  than a unit test, the same pattern this session's other GUI fixes (anchor-only, auto-provisioning)
  already followed.
