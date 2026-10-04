# Multi-neuron 2D mask viewer (napari)

Status: design, approved 2026-10-04.

## Why

The segmentation project's next stage, after per-neuron review finishes, is visualizing several
corrected neurons together (toggle each on/off for inspection) and, after that, mapping contact
between neurons for studying synapses and gap junctions. These two stages are linked: the contact
method converts each neuron's 3D mesh into a point cloud and checks point density near a boundary,
so it needs the same per-neuron 3D geometry a viewer would need to render, in a shared, real-scale
coordinate frame, and it reuses the viewer's own enable/disable state as a contacts-of-interest
filter. Because of that link, this design keeps those two downstream needs in mind even though it
only builds the 2D viewer; contact mapping itself is a separate, later brainstorm.

This spec covers the 2D half only (several neurons' corrected masks, composited per z-slice, each
toggleable). 3D (a matching mesh view) and contact mapping are both deferred, see Deferred below.

## Data source

Lucinda sends back corrected neurons as a bundle (the same per-chain layout
`<neuron>/chain_NN/masks/mask_<z>.png` plus `state.json` that `export_bundle.py` built in the first
place). The viewer takes a tree root and a list of neuron names; the root can be either a bundle
folder directly or a master tree that already had `import_bundle.py` run against it, both have the
identical layout, so neither is a hard requirement of this tool. Nothing here downloads or imports
a bundle automatically, that stays a manual step the user already knows how to do.

## Reusing `neuron_volume()` unmodified

`render_review.py`'s existing `neuron_volume(root, neuron)` already does the hard part: it walks
every chain of a neuron, remaps tier-2 `_pcrop` and legacy `_sam` masks onto the shared `_sam`
pixel grid via `pipeline.chain_masks_in_sam`, crops to that neuron's own bounding box to keep
memory bounded, preserves real z-gaps instead of silently compacting them (mask dropout is a real,
measured failure mode elsewhere in this pipeline), and returns `spacing` already in real-world
nanometers (`SPACING_SAM8_NM`). That spacing is exactly what the contact-mapping method will
eventually need for a meaningful radius check, so no new coordinate work is needed here, it already
exists.

One real gap, found during planning: `neuron_volume()` computes the cropped volume's origin
(`z_min, y0, x0` in the shared grid) internally, but only ever returned `(vol, spacing)`, discarding
it. This design needs that origin too, to place each neuron correctly via napari's `translate`. The
function is extended to return `(vol, origin, spacing)` instead of reimplementing the same remap and
cropping logic a second time elsewhere, a small, mechanical change (its one real caller and its
existing test file's tuple-unpacking sites), not a rewrite.

## Multi-neuron compositing: separate translated layers, no merge

`neuron_volume()` returns each neuron's volume already positioned in the same global `_sam`/z
coordinate system, just locally cropped to its own bounding box. Rather than pasting every
selected neuron into one shared dense array, each neuron becomes its own napari `Labels` layer,
placed in the viewer via napari's `translate` parameter (that neuron's own `z0, y0, x0` origin).
napari composites overlapping layers on its own. This avoids building and holding a monolithic
multi-neuron array in memory, and it is what makes per-neuron toggling a native feature instead of
something this tool has to implement.

One thing `neuron_volume()` does not give for free: its output is a binary 0/1 volume, and napari's
Labels colormap is a deterministic hash keyed by the label *value*, the same globally across every
layer. Two neurons each shown as their own 0/1 layer would both render as label 1 and come out the
same color. Before building a neuron's layer, its volume is multiplied by a small per-neuron
integer ID (1, 2, 3, ... in selection order), so napari's own default colormap then gives each
neuron a distinct, consistent color with no custom palette.

## Toggle is native layer visibility

One layer per neuron means enabling/disabling a neuron for inspection is napari's own layer
visibility checkbox, already a real, queryable Python value (`layer.visible`), not a custom toggle
this tool has to build or expose separately. A later contact-mapping script can read "which
neurons are currently enabled" directly as `[l.name for l in viewer.layers if l.visible]`.

## Overlaps are shown, not resolved

Where two neurons' masks touch or overlap, napari blends both layers' colors visually. This tool
does not attempt to resolve the overlap into one canonical label; that is a separate, already
tracked problem (`resolve_overlaps_argmax`/`resolve_overlaps_watershed` in `sam2_utils/perframe.py`,
part of the existing dense-multineuron-overlay backlog item). This tool's job is inspection, not
producing a merged ground truth.

## Entry point

A new small script, `view_neurons.py` at the repo root, matching the naming convention `gui.py`/
`gui_neuron.py` already use:

```
py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR,RIBL
```

No neuron-picker GUI for v1, a plain CLI flag is enough; see Deferred below for a possible later
picker.

## Testing

napari cannot be meaningfully unit-tested headlessly, so the testable unit is a pure function that
builds each neuron's layer spec (name, ID-multiplied volume, translate offset) from a tree root and
a neuron list, calling `neuron_volume()` per neuron and returning plain data, no napari import. This
follows
the same pattern already used elsewhere in this repo (`gui.py`'s `_peek_chain` is pure and tested;
the viewer construction around it is not). `main()` is a thin wrapper that takes those specs and
adds the real napari layers, not covered by automated tests.

## Deferred (sequenced after this spec, not cut)

- **3D mesh view.** Matching napari `Surface` layers built from the existing `meshing.py` PLY
  output, the natural next step once this 2D viewer proves the toggle/compositing mechanics work.
- **Contact/proximity mapping.** Its own future brainstorm; consumes this work's per-neuron
  geometry and the enable/disable state, per the two linkage points described in Why above.
- **Overlap arbitration.** Already tracked in the existing dense-multineuron-overlay backlog item;
  this viewer only shows overlaps visually and does not resolve them.
- **A neuron-picker GUI.** CLI flag only for v1. A dropdown multi-select, matching
  `render_review.py`'s existing PyQt launcher pattern, could be added later if the CLI becomes
  real friction, not a proven need yet.

## Cut (not planned; would need a fresh ask to revisit)

- **A merged, VAST-style dense labelmap export or cache.** Considered and dropped once the
  per-neuron translated-layer approach turned out not to need a merged array at all. Would only be
  worth building for a real cross-tool-sharing need or a proven slow-reload problem, neither exists
  yet; a handful of neurons loads directly from `neuron_volume()` fast enough today.
- **VAST (the desktop app) as the renderer.** Considered and rejected: it has no Python/scripting
  API, would need a one-way exporter into its `.vsv` format (nothing in this repo writes that
  format today, only reads VAST-exported labelmaps for an unrelated GT-comparison purpose), and it
  cannot expose the toggle state the contact-mapping handoff needs as a queryable value.
- **Custom per-neuron color assignment.** napari's default per-layer colormap is enough for v1.
