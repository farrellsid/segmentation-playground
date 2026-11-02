# Multi-neuron 3D mesh view (napari)

Status: design, approved 2026-10-04.

## Why

[The multi-neuron 2D mask viewer spec](2026-10-04-multi-neuron-2d-viewer-design.md) built the 2D
half of the multi-neuron visualization goal, several corrected neurons' masks composited per
z-slice, each toggleable. That spec deferred a matching 3D mesh view to a later spec. This is that
spec: adding 3D `Surface` layers to the same `view_neurons.py`, in the same napari window as the
existing 2D `Labels` layers, so both can be inspected together.

## Scope: same tool, opt-in

3D is added to `view_neurons.py` behind a new `--mesh` flag, not a separate script and not always
on. Mesh generation (marching cubes plus smoothing) is real compute next to just loading masks for
2D, so a plain 2D-only open stays fast by default; `--mesh` opts into paying that cost.

## Reusing `neuron_volume()` and `meshing.volume_to_mesh()`, both unmodified

For each requested neuron, `render_review.neuron_volume()` is called exactly as the 2D path
already calls it (one call per neuron, shared, not duplicated), returning `(vol, origin, spacing)`.
`vol` and `spacing` feed directly into
`sam2_utils.meshing.volume_to_mesh(vol, spacing=spacing, preset=preset)`, the same function
`render_review.py`'s own single-neuron PLY export already uses, unmodified. `preset` defaults to
`"faithful"`, matching that existing default and its stated reasoning: these meshes exist to catch
mistakes, and smoothing hides exactly the z-to-z jitter that reveals a bad slice.

## The axis-order fix

`volume_to_mesh()` returns vertices in `(x, y, z)` order, reversed specifically for Blender's
convention (documented in its own docstring). Every other array in this viewer, `vol`, `origin`,
`spacing`, and the 2D layers built from them, is `(z, y, x)`. Before a neuron's mesh becomes a
napari layer, its vertices are reversed back to `(z, y, x)` (`verts[:, ::-1]`), so this viewer
stays internally consistent. Getting this backwards would silently mirror or transpose every mesh
with no visible crash, so a test pins it against an asymmetric synthetic shape (see Testing).

## Placement: translate only, no scale

`volume_to_mesh()` already bakes `spacing` into its output vertex coordinates (real-world
nanometers), unlike the raw-voxel `vol` the 2D path scales separately. So a mesh layer only needs
`translate = origin * spacing` (the same formula the 2D layers already use for their own
placement), not a `scale` kwarg.

## Color: reusing this repo's existing per-object palette

`napari.Viewer.add_surface` takes a `vertex_colors` argument directly (confirmed against the
installed napari's real signature, not assumed), so each neuron's mesh gets a flat color: every
vertex set to the same RGB, pulled from `sam2_utils.video_viz._PALETTE`, indexed by that neuron's
1-based selection order (the same index already used for the 2D layer's ID). That palette already
exists in this repo specifically because a naive tab10 cycle has an invisible grey entry at index
7, which once cost a real render a whole neuron's worth of frames; `_PALETTE` has that entry
already filtered out. A mesh's color is not forced to match its own 2D layer's color exactly,
napari's internal Labels colormap is a separate, harder thing to mirror and not worth it for v1,
just that every neuron's mesh is visually distinct from every other neuron's mesh.

## Naming and overlap

Each mesh layer is named `f"{neuron} mesh"`, distinct from the 2D layer's plain `neuron` name, so
both can be open for the same neuron without napari auto-renaming one. Overlapping meshes are left
to napari's own translucent blending (the default `Surface` layer opacity), the same "shown, not
resolved" principle the 2D spec already established; this viewer still does not produce one
canonical merged geometry.

## Testing: exercising the real boundary this time

The 2D task's final review noted that nothing in its tests ran a real `neuron_volume()` output
into the layer-building function, each side of that boundary was mocked independently. This spec's
tests close that gap for the new function: rather than also mocking `meshing.volume_to_mesh`, the
test builds a small real synthetic binary volume (an asymmetric shape, so an axis-order bug would
actually be caught, not an accidental cube) and monkeypatches only `render_review.neuron_volume` to
return it. The real `volume_to_mesh` then runs for real inside the test, same as it does in
production, so the axis-reversal and the `translate` math are both exercised end to end, not
assumed. This mirrors how `tests/test_render_review_mesh.py` already tests `neuron_volume()` for
real against a fixture tree; the same discipline now covers the mesh step too.

## Non-goals

- No contact mapping yet, still its own later spec.
- No color-matching between a neuron's 2D and 3D representation, explicitly cut above.
- No mesh-detail control beyond the existing `preset` choice `meshing.py` already offers; no new
  presets.
- No file export (PLY) from this viewer, `render_review.py`'s existing single-neuron export stays
  the way to get a Blender-ready file; this is a live-viewer-only feature.
