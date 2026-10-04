# Multi-neuron 3D mesh view Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `view_neurons.py` optionally show each neuron's 3D mesh alongside its existing 2D labelmap, in the same napari window, toggleable the same way.

**Architecture:** A new pure function, `build_mesh_specs`, mirrors `view_neurons.py`'s existing `build_layer_specs`: it calls `render_review.neuron_volume()` and `sam2_utils.meshing.volume_to_mesh()`, both already built and unmodified, reverses `volume_to_mesh()`'s Blender-ordered `(x, y, z)` vertices back to this project's `(z, y, x)` convention, places each neuron via the same `translate = origin * spacing` formula the 2D layers use, and colors each neuron's mesh with this repo's existing per-object palette (`sam2_utils.video_viz._color_for`). `main()` gains an opt-in `--mesh` flag that calls it and adds a napari `Surface` layer per neuron.

**Tech Stack:** Python, numpy, napari, `scikit-image` (already a dependency via `meshing.volume_to_mesh`), pytest.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (repo-wide rule, `CLAUDE.md`).
- Run the `humanizer` skill on any committed prose (docs, commit messages) before committing it.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. The new tests must not import napari; only `build_mesh_specs` (a pure function) is unit tested, the napari wiring in `main()` is not.
- Lint only the files touched: `ruff check <file>`, never a whole-tree `ruff format`.
- 3D is opt-in via `--mesh`, never built by default.
- `neuron_volume()` and `meshing.volume_to_mesh()` are both reused exactly as they exist today; neither is modified by this plan.
- `volume_to_mesh()`'s returned vertices are `(x, y, z)` (Blender order); they must be reversed to `(z, y, x)` before becoming layer data, to match every other array in this viewer.
- A mesh layer uses `translate = origin * spacing` (same formula as the 2D layers) and no `scale` kwarg, since `volume_to_mesh()` already bakes `spacing` into its vertex coordinates.
- Each neuron's mesh color comes from `sam2_utils.video_viz._color_for(i)`, `i` being the same 1-based selection-order index `build_layer_specs` already uses for that neuron's label ID. Not required to visually match that neuron's 2D layer color.
- A neuron whose mesh comes back with zero faces (no surface) is skipped entirely, not added as an empty layer.
- No file export (PLY), no new mesh-detail presets beyond `meshing.PRESETS`, no `--preset` CLI flag (the spec only requires `preset="faithful"` as the function's own default, not an exposed CLI flag), no color-matching between a neuron's 2D and 3D layers. All explicitly cut in the spec.

Spec: [docs/superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md](../specs/2026-10-04-multi-neuron-3d-mesh-view-design.md).

---

### Task 1: `build_mesh_specs` and the `--mesh` flag

**Files:**
- Modify: `view_neurons.py` (module docstring, imports, new `build_mesh_specs` function, `main()`)
- Modify: `tests/test_view_neurons.py` (module docstring, new fixture helper, 4 new tests)
- Modify: `docs/reference/cli.md` (the existing `## view_neurons.py` section's flag table)

**Interfaces:**
- Consumes: `render_review.neuron_volume(root, neuron) -> (vol, origin, spacing)` (existing, unchanged), `sam2_utils.meshing.volume_to_mesh(vol, *, spacing, preset="faithful") -> (verts_xyz, faces)` (existing, unchanged), `sam2_utils.video_viz._color_for(obj_id: int)` (existing, unchanged, returns an RGB float triple).
- Produces: `build_mesh_specs(root, neurons: list[str], *, preset: str = "faithful") -> list[dict]`, each dict `{"name": str, "vertices": np.ndarray (N, 3) float, "faces": np.ndarray (M, 3) int, "vertex_colors": np.ndarray (N, 3) float, "translate": tuple[float, float, float]}`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_view_neurons.py`, replace the module docstring (lines 1-8):

Old:
```python
"""build_layer_specs: per-neuron napari layer specs from a tree or bundle root.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, build_layer_specs here) is tested directly; the real napari
wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
```

New:
```python
"""build_layer_specs / build_mesh_specs: per-neuron napari layer specs from a
tree or bundle root, 2D labelmap and 3D mesh respectively.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, these two functions here) is tested directly; the real
napari wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
```

Then add a new fixture helper right after `_fake_neuron_volume` (after line 25, before
`def test_assigns_distinct_ids_in_selection_order`):

```python
def _asymmetric_box_volume():
    """A solid box, a different voxel extent on each axis (z=2, y=4, x=6), with a
    1-voxel margin on every side so marching cubes produces a single closed
    surface. The asymmetry matters: with spacing=(50, 128, 128), the real-world
    span ends up strictly x > y > z, so a test can tell a correctly axis-reversed
    vertex array (z, y, x) apart from volume_to_mesh's native (x, y, z) one by
    which column has the largest span, not by exact coordinates."""
    vol = np.zeros((6, 8, 10), dtype=np.uint8)
    vol[1:3, 1:5, 1:7] = 1
    return vol
```

Then add these 4 tests at the end of the file, before `if __name__ == "__main__":` (before
line 91):

```python
def test_mesh_vertices_are_reversed_back_to_z_y_x_order(monkeypatch):
    vol = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume",
                        _fake_neuron_volume({"AIAL": (vol, (0, 0, 0), spacing)}))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL"])

    assert len(specs) == 1
    verts = specs[0]["vertices"]
    z_span, y_span, x_span = (np.ptp(verts[:, i]) for i in range(3))
    assert x_span > y_span > z_span, (
        f"expected x-span > y-span > z-span (the fixture box is widest in x, "
        f"narrowest in z), got z={z_span:.1f} y={y_span:.1f} x={x_span:.1f}; this "
        f"fails if vertices are still in volume_to_mesh's native (x, y, z) order")


def test_mesh_translate_is_origin_converted_to_real_world_nm(monkeypatch):
    vol = _asymmetric_box_volume()
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (10, 20, 30), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL"])

    assert specs[0]["translate"] == (500.0, 2560.0, 3840.0), (
        "translate must be origin * spacing elementwise, in real-world nm, no scale "
        "kwarg is used for mesh layers since volume_to_mesh already bakes spacing in")


def test_each_neuron_mesh_gets_a_distinct_color_from_the_existing_palette(monkeypatch):
    vol_a = _asymmetric_box_volume()
    vol_b = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol_a, (0, 0, 0), spacing),
        "AIAR": (vol_b, (0, 0, 0), spacing),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL", "AIAR"])

    expected_a = view_neurons.video_viz._color_for(1)
    expected_b = view_neurons.video_viz._color_for(2)
    assert not np.allclose(expected_a, expected_b), (
        "fixture sanity check: the palette must actually give IDs 1 and 2 different "
        "colors, or this test cannot tell a real bug from a coincidence")
    assert np.allclose(specs[0]["vertex_colors"][0], expected_a)
    assert np.allclose(specs[1]["vertex_colors"][0], expected_b)
    assert specs[0]["vertex_colors"].shape == (len(specs[0]["vertices"]), 3)


def test_a_neuron_with_no_masks_is_skipped_not_an_empty_mesh(monkeypatch):
    empty = np.zeros((0, 0, 0), dtype=np.uint8)
    has_masks = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "EMPTYN": (empty, (0, 0, 0), spacing),
        "AIAL": (has_masks, (0, 0, 0), spacing),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["EMPTYN", "AIAL"])

    assert [s["name"] for s in specs] == ["AIAL mesh"], (
        "a neuron with no masks must be skipped entirely, not added as an empty mesh")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: the 4 new tests FAIL with `AttributeError: module 'view_neurons' has no attribute 'build_mesh_specs'` (or similar, since the function does not exist yet). The 4 existing tests for `build_layer_specs` still PASS, unaffected.

- [ ] **Step 3: Add `build_mesh_specs` and the `--mesh` flag to `view_neurons.py`**

Replace the module docstring (lines 1-13):

Old:
```python
"""Open several corrected neurons together in napari, each toggleable on/off.

One napari Labels layer per neuron, each positioned by its own translate/scale
so they land correctly in the shared real-world (nanometre) coordinate space,
reusing render_review.neuron_volume for the per-neuron remap/crop (extended, not
rewritten, to also return each neuron's bounding-box origin, see
render_review.py's own docstring).
Overlapping neurons are shown blended by napari's own layer compositing; this
tool does not resolve overlaps into one canonical label (see
docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
"""
```

New:
```python
"""Open several corrected neurons together in napari, each toggleable on/off.

One napari Labels layer per neuron, each positioned by its own translate/scale
so they land correctly in the shared real-world (nanometre) coordinate space,
reusing render_review.neuron_volume for the per-neuron remap/crop (extended, not
rewritten, to also return each neuron's bounding-box origin, see
render_review.py's own docstring).
Overlapping neurons are shown blended by napari's own layer compositing; this
tool does not resolve overlaps into one canonical label (see
docs/superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

With --mesh, each neuron's 3D surface is also built (reusing
sam2_utils.meshing.volume_to_mesh, unmodified) and shown as its own napari
Surface layer alongside the 2D one, opt-in since mesh generation is real
compute next to just loading masks (see
docs/superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
"""
```

Replace the imports (lines 14-24):

Old:
```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review
```

New:
```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review
from sam2_utils import meshing, video_viz
```

Add `build_mesh_specs` right after `build_layer_specs` (after its closing `return specs`, before
`def main():`):

```python
def build_mesh_specs(root, neurons: list[str], *, preset: str = "faithful") -> list[dict]:
    """One mesh layer spec per neuron whose volume produced a real surface.

    Reuses render_review.neuron_volume (a second, independent call per neuron;
    not shared or cached with build_layer_specs's own call) and
    meshing.volume_to_mesh, both unmodified. volume_to_mesh returns vertices in
    (x, y, z) order for Blender; they are reversed back to (z, y, x) here to stay
    consistent with every other array in this module. Vertices already carry
    real-world nanometres (volume_to_mesh bakes `spacing` in), so only
    `translate` (origin * spacing, the same formula build_layer_specs uses) is
    needed, no `scale`. Each neuron's mesh gets a flat color from this repo's
    existing per-object palette (video_viz._color_for), indexed by the same
    1-based selection order build_layer_specs uses for its label IDs, so a
    neuron's 2D and 3D layers are not forced to match exactly but are both
    individually distinct from every other neuron's.
    """
    specs = []
    for i, neuron in enumerate(neurons, start=1):
        vol, origin, spacing = render_review.neuron_volume(root, neuron)
        if not vol.size or not vol.any():
            print(f"[view] {neuron}: no masks, skipping mesh")
            continue
        verts_xyz, faces = meshing.volume_to_mesh(vol, spacing=spacing, preset=preset)
        if len(faces) == 0:
            print(f"[view] {neuron}: no surface produced, skipping mesh")
            continue
        verts_zyx = verts_xyz[:, ::-1]
        translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))
        color = video_viz._color_for(i)
        vertex_colors = np.tile(color, (len(verts_zyx), 1))
        specs.append({"name": f"{neuron} mesh", "vertices": verts_zyx, "faces": faces,
                      "vertex_colors": vertex_colors, "translate": translate})
    return specs
```

Replace `main()` (the whole function):

Old:
```python
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D mask viewer (napari)")
    ap.add_argument("--tree", required=True,
                    help="bundle folder or output tree root (same layout either way)")
    ap.add_argument("--neurons", required=True,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL")
    args = ap.parse_args()

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    specs = build_layer_specs(args.tree, neurons)
    if not specs:
        raise SystemExit(
            f"[view] none of {neurons} have any masks under {args.tree}")

    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    print(f"[view] opened {len(specs)} neuron(s): {', '.join(s['name'] for s in specs)}")
    napari.run()
```

New:
```python
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D/3D mask viewer (napari)")
    ap.add_argument("--tree", required=True,
                    help="bundle folder or output tree root (same layout either way)")
    ap.add_argument("--neurons", required=True,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL")
    ap.add_argument("--mesh", action="store_true",
                    help="also build and show each neuron's 3D mesh (slower: marching "
                         "cubes + smoothing per neuron, not built unless asked for)")
    args = ap.parse_args()

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    specs = build_layer_specs(args.tree, neurons)
    mesh_specs = build_mesh_specs(args.tree, neurons) if args.mesh else []
    if not specs and not mesh_specs:
        raise SystemExit(
            f"[view] none of {neurons} have any masks under {args.tree}")

    import napari
    viewer = napari.Viewer(title="Multi-neuron mask viewer")
    for spec in specs:
        viewer.add_labels(spec["data"], name=spec["name"],
                          translate=spec["translate"], scale=spec["scale"])
    for spec in mesh_specs:
        viewer.add_surface((spec["vertices"], spec["faces"]), name=spec["name"],
                           vertex_colors=spec["vertex_colors"], translate=spec["translate"])
    opened = [s["name"] for s in specs] + [s["name"] for s in mesh_specs]
    print(f"[view] opened {len(opened)} layer(s): {', '.join(opened)}")
    napari.run()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: PASS, all 8 tests (4 existing `build_layer_specs` tests, 4 new `build_mesh_specs` tests).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same count as the current baseline, plus the 4 new tests).

- [ ] **Step 6: Document the new flag**

In `docs/reference/cli.md`, replace the existing `## view_neurons.py` section's flag table and
description:

Old:
```markdown
## view_neurons.py

Opens several corrected neurons together in one napari window, each as its own
toggleable Labels layer, composited per z-slice in the shared real-world coordinate
space. Reuses `render_review.neuron_volume` for the per-neuron remap/crop; does not
resolve overlaps between neurons into one canonical label. See
[../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md).

| Flag | Effect |
|------|--------|
| `--tree <dir>` | Bundle folder or output tree root to read neurons from (same per-chain layout either way). |
| `--neurons <names>` | Comma-separated neuron names to open together, e.g. `AVAL,AVAR,RIBL`. |
```

New:
```markdown
## view_neurons.py

Opens several corrected neurons together in one napari window, each as its own
toggleable Labels layer, composited per z-slice in the shared real-world coordinate
space. Reuses `render_review.neuron_volume` for the per-neuron remap/crop; does not
resolve overlaps between neurons into one canonical label. With `--mesh`, also builds
each neuron's 3D surface (reusing `sam2_utils.meshing.volume_to_mesh`) as its own
toggleable Surface layer in the same window. See
[../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md)
and
[../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md](../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md).

| Flag | Effect |
|------|--------|
| `--tree <dir>` | Bundle folder or output tree root to read neurons from (same per-chain layout either way). |
| `--neurons <names>` | Comma-separated neuron names to open together, e.g. `AVAL,AVAR,RIBL`. |
| `--mesh` | Also build and show each neuron's 3D mesh. Slower than the 2D-only default, mesh generation is real compute per neuron. |
```

Now invoke the `humanizer` skill on this updated section and apply any fix it suggests before
committing.

- [ ] **Step 7: Lint the touched files**

Run: `ruff check view_neurons.py tests/test_view_neurons.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 8: Manual smoke check against a real bundle (optional but recommended before treating this as done)**

This needs a real bundle and a display, so it is not part of the automated test suite; skip it if
either is unavailable, but run it before relying on this in practice. Use the same real bundle the
2D viewer was smoke-tested against: extract `C:\Users\User\Downloads\AIA.zip` (if not already
extracted from the 2D task's own smoke check) to a scratch folder, giving `<scratch>\AIA\AIAL`,
`<scratch>\AIA\AIAR`, then:

```
py -3 view_neurons.py --tree "<scratch>\AIA" --neurons AIAL,AIAR --mesh
```

Expected: a napari window opens with four layers, `AIAL`, `AIAR` (2D Labels, as before) and
`AIAL mesh`, `AIAR mesh` (3D Surface), each independently toggleable, the two meshes visibly
different colors, both roughly aligned with their corresponding 2D layer's position when the 3D
display mode is toggled on (napari's own `ndisplay` control, not something this plan adds).

- [ ] **Step 9: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py docs/reference/cli.md
git commit -m "view: add an opt-in 3D mesh layer per neuron

--mesh builds each selected neuron's surface via the existing
meshing.volume_to_mesh (unmodified) and shows it as its own toggleable napari
Surface layer alongside the 2D one. Reverses volume_to_mesh's Blender-ordered
(x, y, z) vertices back to this viewer's (z, y, x) convention, places each
mesh via the same translate formula the 2D layers use, and colors each
neuron with the existing video_viz palette instead of a new one.

docs/superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md."
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** every section of the spec maps to this one task: the `--mesh` opt-in flag,
  reusing both existing functions unmodified, the axis-order fix, translate-only placement, the
  reused palette, the `f"{neuron} mesh"` naming, and the strengthened test strategy (a real
  `volume_to_mesh` call inside the test, not a second mock) are all present in the Step 3/Step 1
  code above.
- **Placeholder scan:** no TBD/TODO/"handle edge cases" anywhere in this plan; both the production
  code and the test code are complete.
- **Type consistency:** `build_mesh_specs`'s return dict shape (`name`, `vertices`, `faces`,
  `vertex_colors`, `translate`) is used identically in `main()`'s new loop (`viewer.add_surface`
  call) and in all 4 new tests; no key name or type drifts between them.
