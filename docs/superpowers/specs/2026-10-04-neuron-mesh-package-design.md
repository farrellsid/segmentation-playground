# Saved neuron mesh package (extends the VAST-style labelmap package)

Status: design, approved 2026-10-04.

## Why

The saved-package feature ([[2026-10-04-neuron-package-design]]) shipped 2D-only on purpose: "no
3D/mesh package yet, 2D first, 3D only if this proves out." The 2D package has since been built,
reviewed, and smoke-tested against real data (`AIA_for_lucinda`, round-tripped correctly). This
spec is the 3D follow-on that non-goal pointed at: caching each neuron's already-computed mesh
(marching cubes + smoothing, real compute per the 3D mesh view spec) inside the same package, so a
later `--package` run doesn't have to rebuild it.

## Format: extend the same package, not a second one

A package directory still looks like this, with one new optional piece:

```
<package_dir>/
  manifest.json
  labels/
    000000.png
    ...
  meshes/              # only present if saved with --mesh
    AIAL.ply
    AIAR.ply
```

`manifest.json` gains one optional top-level key, present only when the package was saved with
`--mesh`:

```json
{
  "schema_version": 1,
  "built_from": {...},
  "origin": [...], "spacing": [...], "shape": [...],
  "neurons": [...],
  "meshes": [
    {"name": "AIAL", "id": 1, "translate": [z0, y0, x0], "file": "AIAL.ply"},
    {"name": "AIAR", "id": 2, "translate": [z0, y0, x0], "file": "AIAR.ply"}
  ]
}
```

A package saved without `--mesh` has no `"meshes"` key and no `meshes/` directory; it looks exactly
as it does today. `id` is the same 1-based selection-order integer the live `--mesh` path already
uses for `video_viz._color_for`, carried over so a reloaded mesh gets the same color it would have
live. No vertex colors are stored in the `.ply`, they are derived data (deterministic from `id`),
not independent state.

## Saving: reuse the already-built mesh specs, one `.ply` per neuron

When `--mesh` and `--save-package` are both given, `save_package` also receives the already-built
`mesh_specs` (from `build_mesh_specs`, unmodified, no second compute pass) and, for each one, writes
`meshes/<neuron name>.ply` via `sam2_utils.meshing.write_ply` (the existing, proven writer, reused
as-is).

**Axis order:** `build_mesh_specs` reverses `volume_to_mesh`'s native `(x, y, z)` output to
`(z, y, x)` for consistency with the rest of this module. This repo's one other `.ply` writer
(`render_review.py`'s `neuron_mesh`, meant for opening in Blender) writes `volume_to_mesh`'s
vertices unreversed, in their native `(x, y, z)` order. To keep exactly one `.ply` convention across
the whole repo rather than two visually-identical-but-axis-swapped ones, the package writer reverses
each mesh spec's vertices back to `(x, y, z)` before calling `write_ply`, undoing
`build_mesh_specs`'s own reversal. This is a real extra step, not a no-op, and is called out here so
it isn't "simplified away" later by someone who only looks at one side of the round trip.

If `save_package` is called without mesh specs (the existing 2D-only call sites, `--save-package`
without `--mesh`), behavior is unchanged: no `meshes/` directory, no `"meshes"` manifest key.

## Loading: a new `load_mesh_specs`, parallel to `load_package`

A new function, not a change to `load_package`'s existing return shape (2D specs stay 2D specs;
no caller needs a mixed return). This mirrors the live path, where `build_layer_specs` and
`build_mesh_specs` are already two independent functions over the same neuron list, not one
function with a mode flag.

`load_mesh_specs(package_dir)` reads `manifest.json`'s `"meshes"` entries (raising a clear error if
the key is absent: the package was saved without `--mesh`), reads each `.ply` back via a new
`sam2_utils.meshing.read_ply` (the `write_ply` counterpart, parses exactly the format that function
emits, not a general-purpose PLY reader, same "hand-written, a dependency wouldn't earn itself"
reasoning already in that module), reverses vertices from the file's `(x, y, z)` back to this
module's internal `(z, y, x)`, and rebuilds `vertex_colors` from each entry's `id` via
`video_viz._color_for`. Returns the same shape `build_mesh_specs()` already returns:
`{"name", "vertices", "faces", "vertex_colors", "translate"}`.

## `sam2_utils/meshing.py`: add `read_ply`

`write_ply` writes a fixed ASCII layout: a short header naming vertex/face counts, then one
`x y z` line per vertex, then one `3 a b c` line per face. `read_ply(path)` parses exactly that
shape back into `(verts, faces)` as `(np.float64, np.int64)` arrays, the inverse of `write_ply`.
Like `write_ply`, this is scoped to round-tripping this repo's own output, not general PLY files
from elsewhere; it does not need to handle binary PLY, arbitrary property lists, or non-triangle
faces, because `write_ply` never produces any of those.

## CLI: relax the `--package`/`--mesh` conflict

`view_neurons.py` currently refuses `--package` combined with `--mesh` outright. That blanket
refusal is removed. The new rules:

- `--tree`/`--neurons --mesh --save-package DIR`: as today, builds and shows the live mesh, and now
  also saves it into the package (previously `--save-package` silently ignored `--mesh`'s output;
  this closes that gap).
- `--package DIR --mesh`: loads 2D specs via `load_package` as today, and now also calls
  `load_mesh_specs(DIR)`. If the package has no `"meshes"` key, this fails with a clear error
  ("package has no saved mesh data; re-save with `--tree ... --mesh --save-package`"), not a silent
  skip of the flag.
- `--package DIR` without `--mesh`: unchanged, 2D only, any `meshes/` data present is simply not
  read.

## Non-goals

- No overlap arbitration for meshes. Unlike the 2D labelmap (one dense array, so overlap needed a
  policy), each neuron's mesh stays its own `Surface` layer on both the live and package path;
  there is nothing to merge, so nothing to resolve.
- No mesh compression or decimation beyond what `volume_to_mesh`'s existing presets already do.
- No change to `render_review.py`'s own `neuron_mesh` / Blender-export path. That is a separate,
  pre-existing feature for a different purpose (opening in Blender) and is not touched by this
  spec; the shared `write_ply`/`read_ply` functions are reused by both, not merged into one.
- No guarantee of opening a package's `.ply` files correctly in the actual VAST desktop application,
  same non-goal already stated for the 2D package (VAST doesn't do meshes; "VAST-style" describes
  the labelmap half only).
