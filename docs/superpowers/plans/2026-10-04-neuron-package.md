# Saved neuron package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `view_neurons.py` save the neurons you've built once to a small, portable, VAST-style labelmap package on disk, and reopen that package later without recomputing anything from the source tree.

**Architecture:** Two new pure functions in `view_neurons.py`, `save_package` and `load_package`, both operating on the exact same layer-spec shape `build_layer_specs` already returns (`{"name", "data", "translate", "scale"}`), so the rest of the tool never needs to know whether a layer came from a live tree read or a saved package. `save_package` merges the already-built specs into one shared dense array and writes it as one 16-bit indexed PNG per z-slice, this repo's existing VAST-GT convention (`eval/groundtruth.py` already reads this exact format), plus a `manifest.json`. `load_package` reverses that. `main()` gains `--save-package`/`--package` flags.

**Tech Stack:** Python, numpy, Pillow (`PIL.Image`, already a repo dependency), pytest.

## Global Constraints

- No em dashes anywhere: code, comments, docs, or commit messages (repo-wide rule, `CLAUDE.md`).
- Run the `humanizer` skill on any committed prose (docs, commit messages) before committing it.
- Tests are CPU-only and torch-free: `py -3 -m pytest`. No napari import anywhere in the new tests; `save_package`/`load_package` are pure and napari-free, same as `build_layer_specs`/`build_mesh_specs`.
- Lint only the files touched: `ruff check <file>`, never a whole-tree `ruff format`.
- Package format: `<dir>/manifest.json` plus `<dir>/labels/{i:06d}.png`, one 16-bit single-channel PNG per z-slice (PIL mode `I;16`, pixel value is the neuron ID, `0` is background), written via `Image.fromarray(plane.astype(np.uint16), mode="I;16").save(path)`, the exact pattern already proven in `tests/test_eval_predict_gt.py:82`.
- `manifest.json` schema: `{"schema_version": 1, "built_from": {...}, "origin": [z0, y0, x0], "spacing": [z, y, x], "shape": [n_z, n_y, n_x], "neurons": [{"name": str, "id": int}, ...]}`.
- `save_package` takes the already-built `specs` list, not a tree/neurons pair; it never calls `render_review.neuron_volume()` itself, no second, redundant pass over the source data.
- Overlap policy inside `save_package`: later entries in `specs` overwrite earlier ones on any shared pixel. This is a documented simplification, not overlap arbitration; do not add any blending or priority logic beyond plain overwrite-in-list-order.
- `load_package`'s returned layers use the package's full shared shape for every neuron (not re-cropped to that one neuron's own tighter bounding box the way the live `--tree` path does).
- `--package` is mutually exclusive with `--tree`/`--neurons`, with `--mesh`, and with `--save-package`; `main()` refuses invalid combinations with a clear `SystemExit` message rather than silently ignoring one flag. Exactly one of (`--package`) or (`--tree` and `--neurons`) must be given.
- No 3D/mesh package support in this plan, 2D only.
- No launcher.py changes in this plan, that is a separate, later sub-project.

Spec: [docs/superpowers/specs/2026-10-04-neuron-package-design.md](../specs/2026-10-04-neuron-package-design.md).

---

### Task 1: `save_package`, `load_package`, and the CLI flags

**Files:**
- Modify: `view_neurons.py` (module docstring, imports, two new functions, `main()`)
- Modify: `tests/test_view_neurons.py` (module docstring, new tests)
- Modify: `docs/reference/cli.md` (the existing `## view_neurons.py` section's flag table)

**Interfaces:**
- Consumes: `build_layer_specs(root, neurons) -> list[dict]` (existing, unchanged), the exact spec dict shape it returns.
- Produces: `save_package(specs: list[dict], out_dir, *, built_from: dict) -> Path` and `load_package(package_dir) -> list[dict]` (the same spec shape `build_layer_specs` returns).

- [ ] **Step 1: Write the failing tests**

In `tests/test_view_neurons.py`, replace the module docstring (lines 1-8):

Old:
```python
"""build_layer_specs / build_mesh_specs: per-neuron napari layer specs from a
tree or bundle root, 2D labelmap and 3D mesh respectively.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, these two functions here) is tested directly; the real
napari wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
```

New:
```python
"""build_layer_specs / build_mesh_specs / save_package / load_package: per-neuron
napari layer specs from a tree, bundle, or saved package, 2D labelmap, 3D mesh,
and the VAST-style package round-trip respectively.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, these functions here) is tested directly; the real napari
wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
```

Then add these imports right after the existing `import view_neurons` line:

```python
from eval.groundtruth import _read_label_png
```

Then add this fixture helper, after `_asymmetric_box_volume` (before
`def test_assigns_distinct_ids_in_selection_order`):

```python
def _two_specs(*, overlap: bool = False):
    """Two small, hand-built layer specs in the exact shape build_layer_specs()
    returns, for testing save_package/load_package without needing a real tree
    or neuron_volume() at all. With overlap=True, the second neuron's volume
    occupies the exact same block as the first's, so the overlap policy
    (last-in-list wins) is directly testable. With overlap=False, the second
    neuron is offset two voxels along Z and one along X: both have z-depth 2,
    so z=0..1 is AIAL-only and z=2..3 is AIAR-only, no shared z-slice; the
    1-voxel X offset also makes the union canvas one voxel wider than either
    neuron's own 3-voxel block, so AIAL's own z=0 slice has a real background
    column at the far edge, not just foreground wall to wall. That's what lets
    test_saved_png_is_readable_by_the_existing_vast_gt_reader genuinely check
    both "background is 0" and "AIAL's id is present" on one real slice,
    instead of a slice that happens to have no background pixel at all."""
    spacing = (50.0, 128.0, 128.0)
    vol_a = np.ones((2, 3, 3), dtype=np.uint16)
    data_a = vol_a * np.uint16(1)
    spec_a = {"name": "AIAL", "data": data_a,
             "translate": (0.0, 0.0, 0.0), "scale": spacing}

    vol_b = np.ones((2, 3, 3), dtype=np.uint16)
    data_b = vol_b * np.uint16(2)
    if overlap:
        # same origin as AIAL, same shape, so every voxel overlaps
        spec_b = {"name": "AIAR", "data": data_b,
                 "translate": (0.0, 0.0, 0.0), "scale": spacing}
    else:
        # z: offset by 2 voxels (real-world 2*50=100nm), so AIAL occupies
        # z=0..1 and AIAR occupies z=2..3, no shared z-slice.
        # x: offset by 1 voxel (real-world 1*128=128nm), so the union canvas
        # is 4 voxels wide in x while each neuron's own block is only 3,
        # leaving a real background column in every slice.
        spec_b = {"name": "AIAR", "data": data_b,
                 "translate": (100.0, 0.0, 128.0), "scale": spacing}
    return [spec_a, spec_b]
```

Then add these 7 tests at the end of the file, before `if __name__ == "__main__":`:

```python
def test_save_and_load_package_round_trips_names_and_placement(tmp_path):
    specs = _two_specs()

    out_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={"tree": "fake-root", "neurons": ["AIAL", "AIAR"]})
    loaded = view_neurons.load_package(out_dir)

    assert [s["name"] for s in loaded] == ["AIAL", "AIAR"]
    for spec in loaded:
        assert spec["scale"] == (50.0, 128.0, 128.0)
        assert np.unique(spec["data"]).tolist()[0] == 0, "background must still be 0"


def test_loaded_data_matches_each_neurons_own_id_and_shape(tmp_path):
    specs = _two_specs()

    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    loaded = view_neurons.load_package(out_dir)

    aial = next(s for s in loaded if s["name"] == "AIAL")
    aiar = next(s for s in loaded if s["name"] == "AIAR")
    assert set(np.unique(aial["data"]).tolist()) == {0, 1}
    assert set(np.unique(aiar["data"]).tolist()) == {0, 2}
    # both neurons' layers share the package's one full shape
    assert aial["data"].shape == aiar["data"].shape


def test_overlap_uses_the_later_spec_in_the_list(tmp_path):
    specs = _two_specs(overlap=True)

    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    loaded = view_neurons.load_package(out_dir)

    aial = next(s for s in loaded if s["name"] == "AIAL")
    aiar = next(s for s in loaded if s["name"] == "AIAR")
    # Both specs occupied the exact same voxels. AIAR was second in the input
    # list, so it must have won every one of them: AIAL's reconstructed layer
    # has nothing left (id 1 never appears), AIAR's has the full block (id 2
    # everywhere the original block was).
    assert not np.any(aial["data"] == 1), "AIAL must have lost every overlapping voxel"
    assert np.sum(aiar["data"] == 2) == 2 * 3 * 3, "AIAR must have kept the whole block"


def test_saved_png_is_readable_by_the_existing_vast_gt_reader(tmp_path):
    specs = _two_specs()
    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})

    # Real interop check, not just round-tripping through our own loader: this
    # repo's existing VAST-GT reader (eval/groundtruth.py, unrelated feature,
    # unmodified) must be able to read a package PNG directly.
    arr = _read_label_png(out_dir / "labels" / "000000.png")
    assert arr.dtype == np.uint16
    assert set(np.unique(arr).tolist()) == {0, 1}, (
        "z-slice 0 should hold only background (0) and AIAL's id (1), "
        "AIAR starts one slice later in this fixture's un-overlapped layout")


def test_main_rejects_package_combined_with_tree(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(tmp_path),
        "--tree", "fake-root", "--neurons", "AIAL"])
    with pytest.raises(SystemExit, match="cannot be combined with --tree"):
        view_neurons.main()


def test_main_rejects_package_combined_with_mesh(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(tmp_path), "--mesh"])
    with pytest.raises(SystemExit, match="cannot be combined with --mesh"):
        view_neurons.main()


def test_main_requires_package_or_tree_and_neurons(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view_neurons.py"])
    with pytest.raises(SystemExit, match="need either --package"):
        view_neurons.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: the new tests FAIL. `test_save_and_load_package_round_trips_names_and_placement` and the
others calling `save_package`/`load_package` fail with `AttributeError: module 'view_neurons' has no
attribute 'save_package'`. The three `test_main_rejects_*`/`test_main_requires_*` tests fail because
`main()`'s current argparse setup has `--tree`/`--neurons` as `required=True`, so calling it with
neither (or with `--package` instead) fails with argparse's own `SystemExit` (exit code 2, a usage
error) rather than the specific messages these tests check for with `match=`.

- [ ] **Step 3: Add `save_package` and `load_package` to `view_neurons.py`**

Replace the module docstring (lines 1-20):

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

With --mesh, each neuron's 3D surface is also built (reusing
sam2_utils.meshing.volume_to_mesh, unmodified) and shown as its own napari
Surface layer alongside the 2D one, opt-in since mesh generation is real
compute next to just loading masks (see
docs/superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md).

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
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

With --save-package, the built 2D layers are also saved to disk as a VAST-style
labelmap package (one 16-bit PNG per z-slice plus manifest.json, this repo's
existing VAST-GT convention from eval/groundtruth.py, not a new format), so a
later run can reopen it with --package instead of rebuilding from --tree
(see docs/superpowers/specs/2026-10-04-neuron-package-design.md). --package is
2D-only and cannot be combined with --mesh.

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --save-package <pkg dir>
    py -3 view_neurons.py --package <pkg dir>
"""
```

Replace the imports (lines 21-32):

Old:
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

New:
```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_review
from sam2_utils import meshing, video_viz
```

Add `save_package` and `load_package` right after `build_mesh_specs` (after its closing
`return specs`, before `def main():`):

```python
def save_package(specs: list[dict], out_dir, *, built_from: dict) -> Path:
    """Write `specs` (the same shape build_layer_specs() returns) to `out_dir`
    as a VAST-style labelmap package: one 16-bit single-channel PNG per
    z-slice (pixel value is the neuron ID, 0 is background, the same
    convention eval/groundtruth.py already reads for VAST GT data), plus
    manifest.json.

    Reuses each spec's already-computed id-multiplied `data`, `translate`, and
    `scale` directly; never calls render_review.neuron_volume itself, no
    second pass over the source data. `built_from` is recorded in the
    manifest exactly as given, not interpreted here.

    Overlap policy: later entries in `specs` overwrite earlier ones on any
    shared pixel. A dense single-valued array cannot show two neurons' claims
    on one pixel the way separate live layers blend visually; this is a real,
    deliberate simplification (see the design doc), not overlap arbitration.
    """
    if not specs:
        raise SystemExit("[view] nothing to save, no neuron had any masks")
    out_dir = Path(out_dir)
    spacing = specs[0]["scale"]
    origins = [tuple(int(round(t / s)) for t, s in zip(spec["translate"], spacing))
              for spec in specs]
    z0 = min(o[0] for o in origins)
    y0 = min(o[1] for o in origins)
    x0 = min(o[2] for o in origins)
    z1 = max(o[0] + spec["data"].shape[0] for o, spec in zip(origins, specs))
    y1 = max(o[1] + spec["data"].shape[1] for o, spec in zip(origins, specs))
    x1 = max(o[2] + spec["data"].shape[2] for o, spec in zip(origins, specs))
    shape = (z1 - z0, y1 - y0, x1 - x0)

    canvas = np.zeros(shape, dtype=np.uint16)
    neuron_entries = []
    for spec, origin in zip(specs, origins):
        data = spec["data"]
        dz, dy, dx = origin[0] - z0, origin[1] - y0, origin[2] - x0
        sz, sy, sx = data.shape
        region = canvas[dz:dz + sz, dy:dy + sy, dx:dx + sx]
        mask = data > 0
        region[mask] = data[mask]
        neuron_entries.append({"name": spec["name"], "id": int(data.max())})

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(shape[0]):
        Image.fromarray(canvas[i].astype(np.uint16), mode="I;16").save(
            labels_dir / f"{i:06d}.png")

    manifest = {
        "schema_version": 1,
        "built_from": built_from,
        "origin": [z0, y0, x0],
        "spacing": list(spacing),
        "shape": list(shape),
        "neurons": neuron_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def load_package(package_dir) -> list[dict]:
    """The same layer-spec shape build_layer_specs() returns, read back from a
    package save_package() wrote.

    Every neuron's layer here is the full package shape, not re-cropped to
    that one neuron's own tighter bounding box the way the live --tree path
    crops each neuron individually; packages are already small (a handful of
    neurons), and re-deriving each neuron's own crop from the merged array is
    unneeded complexity for what it would save.
    """
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    spacing = tuple(float(s) for s in manifest["spacing"])
    origin = tuple(int(o) for o in manifest["origin"])
    shape = tuple(int(s) for s in manifest["shape"])
    translate = tuple(float(o) * float(s) for o, s in zip(origin, spacing))

    canvas = np.zeros(shape, dtype=np.uint16)
    for i in range(shape[0]):
        with Image.open(package_dir / "labels" / f"{i:06d}.png") as im:
            arr = np.asarray(im)
        canvas[i] = arr if arr.dtype == np.uint16 else arr.astype(np.uint16)

    specs = []
    for entry in manifest["neurons"]:
        neuron_id = int(entry["id"])
        data = np.where(canvas == neuron_id, np.uint16(neuron_id), np.uint16(0))
        specs.append({"name": entry["name"], "data": data, "translate": translate,
                      "scale": spacing})
    return specs
```

Replace `main()` (the whole function):

Old:
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

New:
```python
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-neuron 2D/3D mask viewer (napari)")
    ap.add_argument("--tree", default=None,
                    help="bundle folder or output tree root (same layout either way); "
                         "required together with --neurons, unless --package is given")
    ap.add_argument("--neurons", default=None,
                    help="comma-separated neuron names, e.g. AVAL,AVAR,RIBL; required "
                         "together with --tree, unless --package is given")
    ap.add_argument("--mesh", action="store_true",
                    help="also build and show each neuron's 3D mesh (slower: marching "
                         "cubes + smoothing per neuron, not built unless asked for); "
                         "cannot be combined with --package")
    ap.add_argument("--save-package", default=None, metavar="DIR",
                    help="also save the --tree/--neurons selection as a VAST-style "
                         "labelmap package to this directory")
    ap.add_argument("--package", default=None, metavar="DIR",
                    help="load from a previously saved package instead of --tree/--neurons")
    args = ap.parse_args()

    using_tree = args.tree is not None or args.neurons is not None
    if args.package and using_tree:
        raise SystemExit("[view] --package cannot be combined with --tree/--neurons")
    if args.package and args.mesh:
        raise SystemExit("[view] --package cannot be combined with --mesh (packages "
                         "are 2D-only for now)")
    if args.package and args.save_package:
        raise SystemExit("[view] --save-package only applies when building from "
                         "--tree/--neurons, not --package")
    if not args.package and not (args.tree and args.neurons):
        raise SystemExit("[view] need either --package, or both --tree and --neurons")

    mesh_specs = []
    if args.package:
        specs = load_package(args.package)
        if not specs:
            raise SystemExit(f"[view] package at {args.package} has no neurons")
    else:
        neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
        specs = build_layer_specs(args.tree, neurons)
        if args.mesh:
            mesh_specs = build_mesh_specs(args.tree, neurons)
        if not specs and not mesh_specs:
            raise SystemExit(
                f"[view] none of {neurons} have any masks under {args.tree}")
        if args.save_package:
            save_package(specs, args.save_package,
                         built_from={"tree": args.tree, "neurons": neurons})
            print(f"[view] saved package to {args.save_package}")

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
Expected: PASS, all 15 tests (4 existing `build_layer_specs`, 4 existing `build_mesh_specs`, 7 new).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `py -3 -m pytest -v`
Expected: PASS (same count as the current baseline, plus the 7 new tests).

- [ ] **Step 6: Document the new flags**

In `docs/reference/cli.md`, replace the existing `## view_neurons.py` section's flag table and
description:

Old:
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

New:
```markdown
## view_neurons.py

Opens several corrected neurons together in one napari window, each as its own
toggleable Labels layer, composited per z-slice in the shared real-world coordinate
space. Reuses `render_review.neuron_volume` for the per-neuron remap/crop; does not
resolve overlaps between neurons into one canonical label. With `--mesh`, also builds
each neuron's 3D surface (reusing `sam2_utils.meshing.volume_to_mesh`) as its own
toggleable Surface layer in the same window. With `--save-package`, also saves the
built neurons to disk as a VAST-style labelmap package (reopen later with `--package`
instead of rebuilding from `--tree`). See
[../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md),
[../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md](../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md),
and
[../superpowers/specs/2026-10-04-neuron-package-design.md](../superpowers/specs/2026-10-04-neuron-package-design.md).

| Flag | Effect |
|------|--------|
| `--tree <dir>` | Bundle folder or output tree root to read neurons from (same per-chain layout either way). Required with `--neurons`, unless `--package` is given instead. |
| `--neurons <names>` | Comma-separated neuron names to open together, e.g. `AVAL,AVAR,RIBL`. Required with `--tree`, unless `--package` is given instead. |
| `--mesh` | Also build and show each neuron's 3D mesh. Slower than the 2D-only default, mesh generation is real compute per neuron. Cannot be combined with `--package`. |
| `--save-package <dir>` | Also save the built `--tree`/`--neurons` selection to this directory as a VAST-style labelmap package. |
| `--package <dir>` | Load from a previously saved package instead of `--tree`/`--neurons`. 2D only, cannot be combined with `--mesh`. |
```

Now invoke the `humanizer` skill on this updated section and apply any fix it suggests before
committing.

- [ ] **Step 7: Lint the touched files**

Run: `ruff check view_neurons.py tests/test_view_neurons.py`
Expected: clean. Fix any warnings before committing.

- [ ] **Step 8: Manual smoke check (optional but recommended before treating this as done)**

This needs a real bundle, it is not part of the automated test suite; skip it if unavailable, but
run it before relying on this in practice. Use the same real bundle the 2D/3D tasks were smoke-tested
against:

```
py -3 view_neurons.py --tree "<scratch>\AIA" --neurons AIAL,AIAR --save-package "<scratch>\AIA_pkg"
```

Expected: the window opens as usual, plus a console line `[view] saved package to <scratch>\AIA_pkg`,
and `<scratch>\AIA_pkg\manifest.json` plus `<scratch>\AIA_pkg\labels\*.png` exist on disk. Then,
separately:

```
py -3 view_neurons.py --package "<scratch>\AIA_pkg"
```

Expected: a napari window opens with `AIAL` and `AIAR` layers, visually matching what the `--tree`
run showed (same two neurons, same relative position to each other), confirming the save/load round
trip actually looks right, not just that the tests pass.

- [ ] **Step 9: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py docs/reference/cli.md
git commit -m "view: save/load a VAST-style neuron package

--save-package writes the built 2D layers to disk as one 16-bit PNG per
z-slice plus manifest.json, this repo's existing VAST-GT convention
(eval/groundtruth.py already reads this exact format), not a new one.
--package reopens a saved package without recomputing from the source tree.
2D only; --mesh and --package are mutually exclusive.

docs/superpowers/specs/2026-10-04-neuron-package-design.md."
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** every section of the spec maps into this one task: the I;16 PNG format reusing
  `eval/groundtruth.py`'s own convention, the manifest schema, the shared-canvas build with
  last-wins overlap, the `load_package` shape matching `build_layer_specs`, both new CLI flags, and
  the `--mesh`/`--package` mutual exclusion the spec's own self-review tightened from "left for the
  plan to resolve" into an explicit rule.
- **Placeholder scan caught a real issue while drafting this plan:** an early draft of
  `test_overlap_uses_the_later_spec_in_the_list` had a malformed placeholder assertion line; it was
  removed and the test rewritten cleanly before this plan was committed, not left in for an
  implementer to fix.
- **Hand-verifying the test fixture's math caught a second, more substantive issue:** the first
  version of `_two_specs()`'s non-overlap case only offset the second neuron along Z, so the union
  canvas's Y/X extent exactly matched each neuron's own 3-voxel block with no margin, which would
  have made `test_saved_png_is_readable_by_the_existing_vast_gt_reader`'s `{0, 1}` assertion false
  (z-slice 0 would have held only `{1}`, no background pixel at all). Fixed by also offsetting the
  second neuron one voxel in X, so the union canvas is genuinely wider than either neuron's own
  footprint and the "background is 0, AIAL's id is 1" claim is actually true on a real slice, not an
  accident of a degenerate fixture.
- **Type consistency:** `save_package`'s `specs` parameter and `load_package`'s return value both
  use the identical dict shape (`name`, `data`, `translate`, `scale`) `build_layer_specs` already
  established; `main()`'s new branches (`--package` vs `--tree`/`--neurons`) both feed that same
  `specs` variable into the unchanged `viewer.add_labels` loop, no new shape introduced.
