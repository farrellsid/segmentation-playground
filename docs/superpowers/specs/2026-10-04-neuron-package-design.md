# Saved neuron package (VAST-style labelmap)

Status: design, approved 2026-10-04.

## Why

The multi-neuron viewer (`view_neurons.py`) has always read live from a tree or bundle, recomputing
every neuron's volume via `neuron_volume()` on every launch. The original visualization goal
([[post-segmentation-visualization-and-contact-mapping-goals]]) asked a storage question that stayed
open through both the 2D and 3D specs: how to keep a "neat and proper" local copy of the neurons
you've actually downloaded and want to look at again, without re-deriving them from the source tree
every time. The 2D spec explicitly cut a merged on-disk format for v1 ("no proven need yet"); that
need is the thing this spec addresses now that both 2D and 3D viewing work and the question of
longer-term organization is the active one.

Scope for this spec is the 2D labelmap only, matching 2D, then if it's successful, do 3D, the same
sequencing already used for the viewer itself. A mesh package is a later, separate spec.

## Format: this repo's own VAST-GT convention, not a new one

`eval/groundtruth.py` already reads VAST-exported labelmaps: one 16-bit single-channel PNG per
z-slice (PIL mode `I;16`), pixel value is the segment ID, `0` is background. This package reuses
that exact convention for the write side too, confirmed against the proven read path
(`eval.groundtruth._read_label_png`, which already auto-casts any PIL-openable single-channel image
to `uint16`) and the one place this repo already writes that mode correctly
(`tests/test_eval_predict_gt.py:82`, `Image.fromarray(arr.astype(np.uint16), mode="I;16").save(path)`).
This is "VAST-style" in the sense that matters: dense per-slice ID labelmaps, the same shape this
repo already has a trusted reader for, not a literal round-trip guarantee into the VAST desktop app
itself, which is unverified and not a goal here.

A package is one directory:

```
<package_dir>/
  manifest.json
  labels/
    000000.png   # I;16, one file per z-slice in the package's own shared bounding box
    000001.png
    ...
```

`manifest.json`:

```json
{
  "schema_version": 1,
  "built_from": {"tree": "<source tree/bundle path>", "neurons": ["AIAL", "AIAR"]},
  "origin": [z0, y0, x0],
  "spacing": [50.0, 128.0, 128.0],
  "shape": [n_z, n_y, n_x],
  "neurons": [{"name": "AIAL", "id": 1}, {"name": "AIAR", "id": 2}]
}
```

`origin`/`spacing` use the exact same `(z, y, x)` convention and units as everywhere else in this
viewer (`neuron_volume()`'s own return, the 2D and 3D specs' `translate` formulas). `id` is the same
1-based, selection-order integer the live `--tree` path already assigns via `build_layer_specs`.

## Building: one shared canvas, not per-neuron crops

Unlike the live viewer (deliberately NOT a merged array, see the 2D spec), a saved package IS a
merged, dense array by definition, that is what makes it a labelmap rather than a set of separate
cropped volumes. Building one calls `neuron_volume()` per neuron exactly as `build_layer_specs`
does, unchanged, then computes the UNION bounding box across every neuron's own `(origin, vol.shape)`
in the shared coordinate space (the same union-of-blocks math `neuron_volume()` itself already uses
internally, just applied across neurons instead of across chains), allocates one `uint16` array of
that shared shape, and pastes each neuron's ID-multiplied volume into its correct offset.

**Overlap policy: last neuron in the selection list wins, not blended, not resolved.** A dense
single-valued array cannot show two neurons' claims on one pixel the way the live viewer's separate
layers can (which blend visually). This is a real, deliberate simplification, not the overlap
arbitration work already tracked elsewhere (`resolve_overlaps_argmax`/`resolve_overlaps_watershed`
in `sam2_utils/perframe.py`, part of the dense-multineuron-overlay backlog). The manifest's own
`built_from.neurons` order is also therefore the overlap-priority order, documented plainly so it is
never a silent surprise.

Each z-slice of the shared array is written as `labels/{i:06d}.png` via
`Image.fromarray(plane.astype(np.uint16), mode="I;16").save(...)`, same pattern as the one proven
instance of this write already in the repo.

## Loading: same layer-spec shape as the live path

`load_package(package_dir)` returns the exact same `list[dict]` shape `build_layer_specs()` already
returns (`{"name", "data", "translate", "scale"}`), so `main()` treats a loaded package and a live
`--tree` read identically, no branching needed past the input-selection step. It reads
`manifest.json`, reads every `labels/*.png` back into one array, and for each neuron entry in the
manifest, extracts `(array == id)` as that neuron's own boolean layer, multiplied back by `id` (so
napari's colormap still gives it the same color it would have live), with `translate`/`scale` set
from the manifest's `origin`/`spacing`. Unlike the live path, every neuron's layer here is the FULL
package shape, not individually re-cropped to that one neuron's own bounding box; packages are
already small (a handful of neurons), and re-deriving each neuron's own tighter crop from the merged
array is unneeded complexity for what it would save.

## CLI: extending `view_neurons.py`, not a new script

Two new flags, following the same "extend, don't proliferate scripts" choice already made for 2D
and 3D:

- `--save-package <dir>`: after building the live specs from `--tree`/`--neurons` as normal, ALSO
  write them out as a package to `<dir>`, then continue opening the viewer as usual. Build and save
  and view in one run.
- `--package <dir>`: load directly from a saved package instead of `--tree`/`--neurons` (mutually
  exclusive with them; `main()` refuses both or neither being given).

`--mesh` and `--package` are mutually exclusive: a package never carries mesh data in this spec
(2D only), and its manifest does not record enough to re-run `neuron_volume()` and `volume_to_mesh()`
from scratch (it has no per-chain information, only the already-merged result). `main()` refuses
`--mesh` together with `--package` with a clear error rather than silently ignoring one of them.
Meshes still work exactly as before with `--tree`/`--neurons --mesh`.

## Non-goals

- No 3D/mesh package yet, 2D first, 3D only if this proves out, per the user's own stated
  sequencing.
- No real overlap resolution, last-neuron-wins is a documented simplification, not arbitration.
- No guarantee of opening correctly in the actual VAST desktop application; the format matches this
  repo's own existing VAST-GT reader convention, which is the proven, relevant bar, not a VAST
  compatibility test that was never run.
- No launcher.py integration yet, deferred to its own later sub-project per
  [[post-segmentation-visualization-and-contact-mapping-goals]].
