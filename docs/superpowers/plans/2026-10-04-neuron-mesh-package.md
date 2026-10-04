# Saved Neuron Mesh Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the saved neuron package (`view_neurons.py --save-package`/`--package`) to
optionally carry each neuron's 3D mesh too, so a package built with `--mesh` can be reopened with
`--package ... --mesh` without recomputing marching cubes.

**Architecture:** Add a hand-written PLY reader (`read_ply`) as the counterpart to the existing
`write_ply` in `sam2_utils/meshing.py`. In `view_neurons.py`, extend `save_package` to optionally
write each mesh spec to `meshes/<name>.ply` and record it in `manifest.json`'s new `"meshes"` key;
add a new `load_mesh_specs` function, parallel to `load_package`, that reads it back; relax the
CLI's `--package`/`--mesh` conflict.

**Tech Stack:** Python, numpy, this repo's existing hand-written ASCII PLY format.

## Global Constraints

- Vertex axis order on disk is `volume_to_mesh`'s native `(x, y, z)` (the same convention
  `render_review.py`'s `neuron_mesh` already writes), not this module's internal `(z, y, x)`. Every
  write reverses `(z, y, x)` -> `(x, y, z)` before calling `write_ply`; every read reverses back.
- No vertex colors are stored on disk. They are derived at load time from each mesh entry's `id`
  via `video_viz._color_for`, the same function the live path already uses.
- A package saved without `--mesh` must look exactly as it does today: no `meshes/` directory, no
  `"meshes"` manifest key. `save_package` keeps working with no `mesh_specs` argument at all.
- `load_mesh_specs(package_dir)` returns the exact same shape `build_mesh_specs()` already returns:
  `{"name", "vertices", "faces", "vertex_colors", "translate"}`.
- A mesh entry's `id` in the manifest must be the same id its matching 2D `neurons` entry already
  has in the same manifest (both ultimately trace back to the neuron's 1-based position in the
  original `--neurons` list), not a position recomputed from `mesh_specs`' own (possibly shorter,
  differently-filtered) list.

---

### Task 1: `read_ply` in `sam2_utils/meshing.py`

**Files:**
- Modify: `sam2_utils/meshing.py` (add `read_ply`, right after `write_ply`, before the `PRESETS`
  section)
- Test: `tests/test_meshing.py`

**Interfaces:**
- Produces: `read_ply(path) -> Tuple[np.ndarray, np.ndarray]`, verts as `float64` `(N, 3)`, faces as
  `int64` `(M, 3)`. Used by `view_neurons.load_mesh_specs` in Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_meshing.py`, right after the existing `TestWritePly` class (after line 100,
before the `from sam2_utils.meshing import PRESETS, volume_to_mesh` import on line 103):

```python
class TestReadPly:
    def test_round_trips_a_write_ply_file(self, tmp_path):
        v, f = _tube(n_rings=4, per_ring=6)
        p = write_ply(tmp_path / "t.ply", v, f)
        v2, f2 = read_ply(p)
        assert v2.dtype == np.float64
        assert f2.dtype == np.int64
        assert np.allclose(v2, v, atol=1e-4), (
            "write_ply rounds to 4 decimal places in its ASCII output, so an exact "
            "equality check would be too strict")
        assert np.array_equal(f2, f)

    def test_reads_back_the_exact_counts_from_the_header(self, tmp_path):
        v, f = _tube(n_rings=6, per_ring=8)
        p = write_ply(tmp_path / "t.ply", v, f)
        v2, f2 = read_ply(p)
        assert len(v2) == len(v)
        assert len(f2) == len(f)
```

Update the import on line 23 to also pull in `read_ply`:

```python
from sam2_utils.meshing import taubin_smooth, write_ply, read_ply
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_meshing.py -v -k TestReadPly`
Expected: FAIL (collection error) with `ImportError: cannot import name 'read_ply'`

- [ ] **Step 3: Implement `read_ply`**

In `sam2_utils/meshing.py`, insert immediately after `write_ply` (after the `return path` on the
line following the header/body-writing code, i.e. right after what is currently line 84, before
the blank lines leading into the `PRESETS` dict):

```python
def read_ply(path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an ASCII PLY written by `write_ply`. Parses exactly that fixed layout (a
    short header naming vertex/face counts, then one "x y z" line per vertex, then one
    "3 a b c" line per triangle face); this is not a general-purpose PLY reader, same
    "a dependency wouldn't earn itself" reasoning as `write_ply` itself."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    n_vertex = n_face = 0
    end_header = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex "):
            n_vertex = int(line.split()[-1])
        elif line.startswith("element face "):
            n_face = int(line.split()[-1])
        elif line == "end_header":
            end_header = i
            break
    body = lines[end_header + 1:]
    verts = np.array(
        [[float(x) for x in line.split()] for line in body[:n_vertex]],
        dtype=np.float64)
    faces = np.array(
        [[int(x) for x in line.split()[1:]] for line in body[n_vertex:n_vertex + n_face]],
        dtype=np.int64)
    return verts, faces
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_meshing.py -v`
Expected: all PASS, including the two new `TestReadPly` tests and every pre-existing test in the
file (confirms the new function and import did not break `TestWritePly`/`TestTaubinSmooth`).

- [ ] **Step 5: Commit**

```bash
git add sam2_utils/meshing.py tests/test_meshing.py
git commit -m "feat: add read_ply, the write_ply counterpart"
```

---

### Task 2: Mesh data in `save_package`/`load_mesh_specs`, and CLI wiring

**Files:**
- Modify: `view_neurons.py` (module docstring, `save_package`, new `load_mesh_specs`, `main`)
- Modify: `docs/reference/cli.md` (`## view_neurons.py` section)
- Test: `tests/test_view_neurons.py`

**Interfaces:**
- Consumes: `meshing.write_ply`/`meshing.read_ply` (Task 1), `meshing.volume_to_mesh` (existing),
  `video_viz._color_for` (existing), `build_mesh_specs` (existing, unmodified).
- Produces: `save_package(specs, out_dir, *, built_from, mesh_specs=None) -> Path` (extended
  signature, old 2-positional-plus-`built_from` call sites keep working unchanged since
  `mesh_specs` defaults to `None`). `load_mesh_specs(package_dir) -> list[dict]`, the same shape
  `build_mesh_specs()` returns.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_view_neurons.py`, after `test_a_later_neurons_own_background_does_not_erase_an_earlier_neurons_foreground`
(after line 283) and before `test_saved_png_is_readable_by_the_existing_vast_gt_reader`:

```python
def _two_mesh_specs():
    """Two small, hand-built mesh specs in the exact shape build_mesh_specs() returns,
    matching the two neuron names _two_specs() uses, so a test can save both the 2D
    and mesh specs into the same package and check the mesh entries' ids line up with
    the 2D neurons' ids. Uses the asymmetric box fixture (via build_mesh_specs itself,
    not hand-built vertices) so the written .ply's own axis order is independently
    checkable the same way test_mesh_vertices_are_reversed_back_to_z_y_x_order already
    checks build_mesh_specs's in-memory output."""
    import view_neurons as vn
    vol = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)

    def fake(root, neuron, **kw):
        return {"AIAL": (vol, (0, 0, 0), spacing), "AIAR": (vol, (0, 0, 0), spacing)}[neuron]

    import render_review
    real_neuron_volume = render_review.neuron_volume
    render_review.neuron_volume = fake
    try:
        mesh_specs = vn.build_mesh_specs("fake-root", ["AIAL", "AIAR"])
    finally:
        render_review.neuron_volume = real_neuron_volume
    return mesh_specs


def test_save_package_writes_one_ply_per_neuron_with_matching_2d_ids(tmp_path):
    specs = _two_specs()
    mesh_specs = _two_mesh_specs()

    out_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={}, mesh_specs=mesh_specs)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert (out_dir / "meshes" / "AIAL.ply").exists()
    assert (out_dir / "meshes" / "AIAR.ply").exists()
    mesh_by_name = {m["name"]: m for m in manifest["meshes"]}
    neuron_by_name = {n["name"]: n for n in manifest["neurons"]}
    assert mesh_by_name["AIAL"]["id"] == neuron_by_name["AIAL"]["id"], (
        "a mesh entry's id must match its neuron's own 2D id, so a reloaded mesh "
        "gets the same color it would have live, not a renumbered one")
    assert mesh_by_name["AIAR"]["id"] == neuron_by_name["AIAR"]["id"]
    assert mesh_by_name["AIAL"]["file"] == "AIAL.ply"


def test_save_package_without_mesh_specs_writes_no_meshes_key_or_dir(tmp_path):
    specs = _two_specs()

    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "meshes" not in manifest
    assert not (out_dir / "meshes").exists()


def test_saved_ply_uses_native_x_y_z_order_matching_render_reviews_own_ply_writer(tmp_path):
    specs = _two_specs()
    mesh_specs = _two_mesh_specs()
    out_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={}, mesh_specs=mesh_specs)

    verts_on_disk, _faces = view_neurons.meshing.read_ply(out_dir / "meshes" / "AIAL.ply")
    x_span, y_span, z_span = (np.ptp(verts_on_disk[:, i]) for i in range(3))
    assert x_span > y_span > z_span, (
        "the fixture box is widest in x, narrowest in z; the .ply on disk must store "
        "volume_to_mesh's native (x, y, z) order, same as render_review.py's neuron_mesh, "
        "not this module's internal (z, y, x) order")


def test_load_mesh_specs_round_trips_vertices_faces_translate_and_color(tmp_path):
    specs = _two_specs()
    mesh_specs = _two_mesh_specs()
    out_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={}, mesh_specs=mesh_specs)

    loaded = view_neurons.load_mesh_specs(out_dir)

    assert [s["name"] for s in loaded] == ["AIAL mesh", "AIAR mesh"]
    original_by_name = {s["name"]: s for s in mesh_specs}
    for spec in loaded:
        original = original_by_name[spec["name"]]
        assert np.allclose(spec["vertices"], original["vertices"], atol=1e-4)
        assert np.array_equal(spec["faces"], original["faces"])
        assert spec["translate"] == original["translate"]
        assert np.allclose(spec["vertex_colors"][0], original["vertex_colors"][0]), (
            "color must be re-derived from the stored id and match the live color "
            "exactly, since nothing else determines a mesh's color")


def test_load_mesh_specs_raises_a_clear_error_when_package_has_no_mesh_data(tmp_path):
    specs = _two_specs()
    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})

    with pytest.raises(SystemExit, match="no saved mesh data"):
        view_neurons.load_mesh_specs(out_dir)
```

Add `import json` near the top of `tests/test_view_neurons.py` if not already present (it currently
is not; check the existing imports at lines 11-22 first, since `json.loads` is used above).

Now update the CLI tests. Replace `test_main_rejects_package_combined_with_mesh` (lines 309-314,
the behavior it checks is being removed) with:

```python
def test_main_allows_package_combined_with_mesh_when_package_has_mesh_data(monkeypatch, tmp_path):
    specs = _two_specs()
    mesh_specs = _two_mesh_specs()
    pkg_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={}, mesh_specs=mesh_specs)

    fake_napari = types.ModuleType("napari")
    fake_napari.Viewer = lambda **kw: _FakeViewer()
    fake_napari.run = lambda: None
    monkeypatch.setitem(sys.modules, "napari", fake_napari)
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(pkg_dir), "--mesh"])

    view_neurons.main()  # must not raise


def test_main_package_with_mesh_fails_clearly_when_package_has_no_mesh_data(monkeypatch, tmp_path):
    specs = _two_specs()
    pkg_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(pkg_dir), "--mesh"])

    with pytest.raises(SystemExit, match="no saved mesh data"):
        view_neurons.main()
```

The first test calls `view_neurons.main()`, which reaches `import napari` right before building the
real viewer. `main()`'s `import napari` is a plain import inside the function body, so it is a
*local* name, not a `view_neurons.napari` module attribute; `monkeypatch.setattr(view_neurons,
"napari", ...)` would not work (there is no such attribute to patch) and `monkeypatch.setattr
(view_neurons.napari, ...)` would raise `AttributeError` outright for the same reason. The correct
way to stand in for a module that a function imports internally is to put a fake module object into
`sys.modules["napari"]` before calling `main()`: Python's `import` statement checks `sys.modules`
first, so `main()`'s own `import napari` picks up the fake module instead of ever touching the real
package, with no GUI window opened. `test_main_package_with_mesh_fails_clearly_when_package_has_no_mesh_data`
needs no such fake: its `SystemExit` is raised by `load_mesh_specs` before `main()` ever reaches its
`import napari` line.

To keep this test file's existing "pure, no real napari window" discipline (see the file's own
module docstring at the top), add two things near the top of the file, right after `_two_specs()`
(before `test_assigns_distinct_ids_in_selection_order`):

```python
class _FakeViewer:
    """Stands in for napari.Viewer so test_main_* can exercise main() end-to-end
    (argument parsing, load_package/load_mesh_specs, the final viewer.add_* calls)
    without opening a real GUI window."""
    def add_labels(self, *a, **kw):
        pass

    def add_surface(self, *a, **kw):
        pass
```

Add `import types` to the imports at the top of the file (alongside the existing `import pathlib` /
`import sys`), needed for the `types.ModuleType("napari")` fake module above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: the new tests FAIL — `test_save_package_writes_one_ply_per_neuron_with_matching_2d_ids`
and the others with an `AttributeError` or `TypeError` (`save_package` does not accept
`mesh_specs`, `load_mesh_specs` does not exist); `test_main_allows_package_combined_with_mesh...`
fails because `main()` still raises on `--package` + `--mesh`. Pre-existing tests still PASS.

- [ ] **Step 3: Update the module docstring**

Replace the module docstring's `--save-package`/`--package` paragraph and usage examples (current
lines 18-28) with:

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
(see docs/superpowers/specs/2026-10-04-neuron-package-design.md). Combined with
--mesh, each neuron's mesh is also cached into the same package (one .ply per
neuron plus a "meshes" entry in manifest.json, see
docs/superpowers/specs/2026-10-04-neuron-mesh-package-design.md); --package
--mesh then loads those meshes back instead of recomputing them, and fails
clearly if the package was saved without --mesh.

    py -3 view_neurons.py --tree <bundle or output tree root> --neurons AVAL,AVAR,RIBL
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --save-package <pkg dir>
    py -3 view_neurons.py --tree <root> --neurons AVAL,AVAR --mesh --save-package <pkg dir>
    py -3 view_neurons.py --package <pkg dir>
    py -3 view_neurons.py --package <pkg dir> --mesh
"""
```

- [ ] **Step 4: Extend `save_package`**

Replace the `save_package` function's signature and its final manifest-writing block. The current
function (starting `def save_package(specs: list[dict], out_dir, *, built_from: dict) -> Path:`)
keeps everything through building `canvas`/`neuron_entries`/writing the PNGs unchanged. Change the
signature line to:

```python
def save_package(specs: list[dict], out_dir, *, built_from: dict,
                 mesh_specs: list[dict] | None = None) -> Path:
```

Add one sentence to the docstring, right after the existing overlap-policy paragraph:

```python
    `mesh_specs` (the same shape `build_mesh_specs()` returns), when given and non-empty, is
    also written: one `.ply` per neuron under `meshes/`, plus a `"meshes"` entry in the manifest.
    Omitted or empty, the package is 2D-only, exactly as before this parameter existed.
```

Then replace the existing manifest-building/writing block (currently):

```python
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
```

with:

```python
    manifest = {
        "schema_version": 1,
        "built_from": built_from,
        "origin": [z0, y0, x0],
        "spacing": list(spacing),
        "shape": list(shape),
        "neurons": neuron_entries,
    }

    if mesh_specs:
        id_by_name = {entry["name"]: entry["id"] for entry in neuron_entries}
        meshes_dir = out_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        mesh_entries = []
        for spec in mesh_specs:
            name = spec["name"].removesuffix(" mesh")
            if name not in id_by_name:
                raise SystemExit(
                    f"[view] mesh for {name!r} has no matching 2D neuron entry in this "
                    f"package; save_package needs mesh_specs built from the same neuron "
                    f"list as specs")
            verts_xyz = spec["vertices"][:, ::-1]
            ply_path = meshes_dir / f"{name}.ply"
            meshing.write_ply(ply_path, verts_xyz, spec["faces"])
            mesh_entries.append({
                "name": name,
                "id": id_by_name[name],
                "translate": [float(t) for t in spec["translate"]],
                "file": ply_path.name,
            })
        manifest["meshes"] = mesh_entries

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
```

- [ ] **Step 5: Add `load_mesh_specs`**

Insert this new function right after `load_package` (after its closing `return specs` and before
`def main() -> None:`):

```python
def load_mesh_specs(package_dir) -> list[dict]:
    """The same layer-spec shape build_mesh_specs() returns, read back from a package
    save_package() wrote with mesh_specs. Raises a clear SystemExit if the package was
    saved without --mesh (no "meshes" key in its manifest), rather than silently
    returning an empty list.
    """
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    mesh_entries = manifest.get("meshes")
    if not mesh_entries:
        raise SystemExit(
            f"[view] package at {package_dir} has no saved mesh data; re-save with "
            f"--tree ... --mesh --save-package to include meshes")

    specs = []
    for entry in mesh_entries:
        verts_xyz, faces = meshing.read_ply(package_dir / "meshes" / entry["file"])
        verts_zyx = verts_xyz[:, ::-1]
        translate = tuple(float(t) for t in entry["translate"])
        color = video_viz._color_for(entry["id"])
        vertex_colors = np.tile(color, (len(verts_zyx), 1))
        specs.append({"name": f"{entry['name']} mesh", "vertices": verts_zyx,
                      "faces": faces, "vertex_colors": vertex_colors,
                      "translate": translate})
    return specs
```

- [ ] **Step 6: Wire it into `main()`**

Replace this block in `main()`:

```python
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
```

with:

```python
    using_tree = args.tree is not None or args.neurons is not None
    if args.package and using_tree:
        raise SystemExit("[view] --package cannot be combined with --tree/--neurons")
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
        if args.mesh:
            mesh_specs = load_mesh_specs(args.package)
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
                         built_from={"tree": args.tree, "neurons": neurons},
                         mesh_specs=mesh_specs)
            print(f"[view] saved package to {args.save_package}")
```

Also update the `--mesh` and `--package` argparse help strings (in the `ap.add_argument` calls
above this block):

```python
    ap.add_argument("--mesh", action="store_true",
                    help="also build and show each neuron's 3D mesh (slower: marching "
                         "cubes + smoothing per neuron, not built unless asked for); "
                         "with --package, loads a saved mesh instead of recomputing it")
    ap.add_argument("--save-package", default=None, metavar="DIR",
                    help="also save the --tree/--neurons selection as a VAST-style "
                         "labelmap package to this directory; with --mesh, also caches "
                         "each neuron's mesh into the same package")
    ap.add_argument("--package", default=None, metavar="DIR",
                    help="load from a previously saved package instead of --tree/--neurons; "
                         "add --mesh to also load that package's cached meshes")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_view_neurons.py tests/test_meshing.py -v`
Expected: all PASS, including every new test from Step 1 and every pre-existing test in both
files.

- [ ] **Step 8: Update `docs/reference/cli.md`**

Replace the `## view_neurons.py` section's prose paragraph and flag table (the section currently
starts at the `## view_neurons.py` heading and ends right before `## eval.score_batch`) with:

```markdown
## view_neurons.py

Opens several corrected neurons together in one napari window, each as its own
toggleable Labels layer, composited per z-slice in the shared real-world coordinate
space. Reuses `render_review.neuron_volume` for the per-neuron remap/crop; does not
resolve overlaps between neurons into one canonical label. With `--mesh`, also builds
each neuron's 3D surface (reusing `sam2_utils.meshing.volume_to_mesh`) as its own
toggleable Surface layer in the same window. With `--save-package`, also saves the
built neurons to disk as a VAST-style labelmap package (reopen later with `--package`
instead of rebuilding from `--tree`); combined with `--mesh`, each neuron's mesh is
also cached into the package (one `.ply` per neuron) so `--package ... --mesh` loads
it back instead of recomputing it. See
[../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md](../superpowers/specs/2026-10-04-multi-neuron-2d-viewer-design.md),
[../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md](../superpowers/specs/2026-10-04-multi-neuron-3d-mesh-view-design.md),
[../superpowers/specs/2026-10-04-neuron-package-design.md](../superpowers/specs/2026-10-04-neuron-package-design.md),
and
[../superpowers/specs/2026-10-04-neuron-mesh-package-design.md](../superpowers/specs/2026-10-04-neuron-mesh-package-design.md).

| Flag | Effect |
|------|--------|
| `--tree <dir>` | Bundle folder or output tree root to read neurons from (same per-chain layout either way). Required with `--neurons`, unless `--package` is given instead. |
| `--neurons <names>` | Comma-separated neuron names to open together, e.g. `AVAL,AVAR,RIBL`. Required with `--tree`, unless `--package` is given instead. |
| `--mesh` | Also build and show each neuron's 3D mesh. Slower than the 2D-only default, mesh generation is real compute per neuron. With `--package`, loads a cached mesh instead of recomputing it (fails clearly if the package has none). |
| `--save-package <dir>` | Also save the built `--tree`/`--neurons` selection to this directory as a VAST-style labelmap package. With `--mesh`, also caches each neuron's mesh into the same package. |
| `--package <dir>` | Load from a previously saved package instead of `--tree`/`--neurons`. Add `--mesh` to also load that package's cached meshes. |
```

- [ ] **Step 9: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py docs/reference/cli.md
git commit -m "feat: cache neuron meshes in the saved package"
```
