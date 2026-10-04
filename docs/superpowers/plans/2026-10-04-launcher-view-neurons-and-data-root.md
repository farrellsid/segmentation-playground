# Launcher: view_neurons.py Button and Shared Data Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `launcher.py` a "View neurons" button for `view_neurons.py`, and make its existing
"frames cache" field double as a data-root convenience for the other two path pickers, so a drive
migration (like the current F: -> E: one) needs fewer manual re-points.

**Architecture:** Split `view_neurons.py`'s `main()` into small reusable pieces
(`_build_from_tree`, `_open_viewer`, and a new `run(source, neurons, *, mesh=False)`), the same
shape `render_review.run(source, neurons)` already has. `launcher.py` gets a new
`open_neuron_viewer` entry point (mirroring the existing `open_render_window`), a checkbox, a
button, a relabeled field, and two Browse-dialog fallback changes.

**Tech Stack:** Python, qtpy (Qt), napari (imported lazily, as today).

## Global Constraints

- `view_neurons.run(source, neurons, *, mesh=False)` must produce the exact same viewer as calling
  `main()` with `--tree source --neurons ,-joined(neurons)` and, if `mesh`, `--mesh` — same specs,
  same napari calls, same print line. `main()`'s observable behavior (every existing test in
  `tests/test_view_neurons.py`) must not change.
- The napari-window-building code (`import napari`, `viewer.add_labels`/`add_surface`, the
  `[view] opened ...` print, `napari.run()`) must exist in exactly one place, shared by `run()` and
  both branches of `main()` (the `--tree` branch and the `--package` branch), not duplicated.
- `launcher.py`'s `frames_root` profile key, its `SAM2_FRAMES_ROOT` env var, and
  `machine_checks`' internal check name ("frames cache") are all unchanged. Only the window's
  visible label text for that field changes. An existing `~/.sam2review/profile.json` written
  before this change must still load and behave identically (see
  `test_a_profile_written_before_these_keys_existed_still_loads` in
  `tests/test_launcher_config.py` for the precedent this must not break).
- The new Browse-dialog fallbacks (Raw EM, Bundle/output tree) only apply when the target field
  being browsed is currently empty. A field with a real value already in it is never overridden by
  the data-root fallback.
- `checkpoint_dir`'s Browse dialog is unchanged (no data-root fallback); it is a genuinely
  independent concept (shared model weights, not project data).

---

### Task 1: `view_neurons.py` — extract `run()`, shared by a new direct entry point and `main()`

**Files:**
- Modify: `view_neurons.py` (replace the tail of the file, from `def main() -> None:` onward)
- Test: `tests/test_view_neurons.py`

**Interfaces:**
- Produces: `run(source, neurons: list[str], *, mesh: bool = False) -> None`, the function
  `launcher.py`'s Task 2 calls directly. Also produces two private helpers,
  `_build_from_tree(tree, neurons, *, mesh=False) -> tuple[list[dict], list[dict]]` and
  `_open_viewer(specs, mesh_specs) -> None`, used internally by both `run()` and `main()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_view_neurons.py`, right after the `_FakeViewer` class (currently at lines
82-90, right before the blank line that precedes `test_assigns_distinct_ids_in_selection_order`).
First, replace `_FakeViewer` itself so it records what was called (needed to assert `mesh=True`
actually added a surface layer):

```python
class _FakeViewer:
    """Stands in for napari.Viewer so test_main_*/test_run_* can exercise main()/run()
    end-to-end (argument parsing or direct call, load_package/load_mesh_specs or
    build_layer_specs/build_mesh_specs, the final viewer.add_* calls) without opening
    a real GUI window. Records every add_labels/add_surface call so a test can assert
    on what was actually added, not just that nothing raised."""
    def __init__(self):
        self.add_labels_calls = []
        self.add_surface_calls = []

    def add_labels(self, *a, **kw):
        self.add_labels_calls.append((a, kw))

    def add_surface(self, *a, **kw):
        self.add_surface_calls.append((a, kw))
```

Then add the new tests, right after `test_a_neuron_with_no_masks_is_skipped_not_an_empty_mesh`
(the last test of the `build_mesh_specs` section) and before `test_save_and_load_package_round_trips_names_and_placement`:

```python
def _fake_napari_module(viewer_holder: dict):
    """A fresh fake napari module whose Viewer() call stashes the created _FakeViewer
    into viewer_holder["v"], so a test can inspect it after run()/main() returns."""
    fake = types.ModuleType("napari")

    def _make_viewer(**kw):
        v = _FakeViewer()
        viewer_holder["v"] = v
        return v

    fake.Viewer = _make_viewer
    fake.run = lambda: None
    return fake


def test_run_builds_from_tree_and_opens_the_viewer(monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))

    view_neurons.run("fake-root", ["AIAL"])

    assert len(holder["v"].add_labels_calls) == 1
    assert holder["v"].add_surface_calls == [], "mesh defaults to False, no surface expected"


def test_run_with_mesh_true_also_builds_and_adds_a_mesh_layer(monkeypatch):
    vol = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), spacing),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))

    view_neurons.run("fake-root", ["AIAL"], mesh=True)

    assert len(holder["v"].add_labels_calls) == 1
    assert len(holder["v"].add_surface_calls) == 1, "mesh=True must add a surface layer"


def test_run_raises_when_no_neuron_has_any_masks(monkeypatch):
    empty = np.zeros((0, 0, 0), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "EMPTYN": (empty, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))

    with pytest.raises(SystemExit, match="none of"):
        view_neurons.run("fake-root", ["EMPTYN"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v -k test_run_`
Expected: FAIL with `AttributeError: module 'view_neurons' has no attribute 'run'`

- [ ] **Step 3: Refactor `view_neurons.py`'s tail into `_build_from_tree`, `_open_viewer`, `run`, and a leaner `main`**

Replace everything from `def main() -> None:` to the end of the file (the current tail, starting
right after `load_mesh_specs`'s closing `return specs`) with:

```python
def _build_from_tree(tree, neurons: list[str], *, mesh: bool = False):
    """Build layer specs (and, if requested, mesh specs) from a live tree/bundle.

    Shared by run() and main()'s --tree path, so the "no masks anywhere" check and the
    build order live in exactly one place.
    """
    specs = build_layer_specs(tree, neurons)
    mesh_specs = build_mesh_specs(tree, neurons) if mesh else []
    if not specs and not mesh_specs:
        raise SystemExit(f"[view] none of {neurons} have any masks under {tree}")
    return specs, mesh_specs


def _open_viewer(specs: list[dict], mesh_specs: list[dict]) -> None:
    """Open the napari window with the given layer specs.

    Shared by run() and both branches of main() (--tree and --package), so there is
    exactly one place that builds the window.
    """
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


def run(source, neurons: list[str], *, mesh: bool = False) -> None:
    """Build and open the viewer directly, the same shape render_review.run(source,
    neurons) already has (the function launcher.py's "Render video + mesh" button
    calls). This is what launcher.py's "View neurons" button calls.

    --save-package/--package are CLI-only concerns this entry point does not take;
    see main() below for those.
    """
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    _open_viewer(specs, mesh_specs)


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
                         "with --package, loads a saved mesh instead of recomputing it")
    ap.add_argument("--save-package", default=None, metavar="DIR",
                    help="also save the --tree/--neurons selection as a VAST-style "
                         "labelmap package to this directory; with --mesh, also caches "
                         "each neuron's mesh into the same package")
    ap.add_argument("--package", default=None, metavar="DIR",
                    help="load from a previously saved package instead of --tree/--neurons; "
                         "add --mesh to also load that package's cached meshes")
    args = ap.parse_args()

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
        specs, mesh_specs = _build_from_tree(args.tree, neurons, mesh=args.mesh)
        if args.save_package:
            save_package(specs, args.save_package,
                         built_from={"tree": args.tree, "neurons": neurons},
                         mesh_specs=mesh_specs)
            print(f"[view] saved package to {args.save_package}")

    _open_viewer(specs, mesh_specs)


if __name__ == "__main__":
    main()
```

This is a pure refactor of the existing tail: every line of behavior (the four `SystemExit`
messages, the build order, the save step, the napari calls, the print line) is unchanged, just
split into named pieces. No existing test's expectations should need to change.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: all PASS, including the 3 new `test_run_*` tests and every pre-existing test in the
file (confirms the refactor did not change `main()`'s observable behavior).

- [ ] **Step 5: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py
git commit -m "refactor: extract view_neurons.run(), shared by main() and launcher.py"
```

---

### Task 2: `launcher.py` — a "View neurons" button, and the data-root convenience

**Files:**
- Modify: `launcher.py`
- Test: `tests/test_render_review_gui.py`

**Interfaces:**
- Consumes: `view_neurons.run(source, neurons, *, mesh=False)` (Task 1).
- Produces: `open_neuron_viewer(source: str, neurons, *, mesh: bool = False) -> None`, a new
  module-level function in `launcher.py`, mirroring the existing `open_render_window`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_render_review_gui.py`, inside the existing `TestLauncherIntegration` class
(after `test_no_ticked_neurons_passes_an_empty_list_not_none`, before the file's closing
`if __name__ == "__main__":` block):

```python
    def test_launcher_exposes_a_view_neurons_entry_point(self):
        import launcher
        assert hasattr(launcher, "open_neuron_viewer")

    def test_view_neurons_entry_point_hands_its_source_and_neurons_across(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False: seen.update(
                                source=source, neurons=neurons, mesh=mesh))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"], mesh=True)
        assert seen["source"] == "/some/bundle"
        assert seen["neurons"] == ["AIBL"]
        assert seen["mesh"] is True

    def test_view_neurons_entry_point_no_ticked_neurons_passes_an_empty_list_not_none(
            self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False: seen.update(
                                neurons=neurons))
        launcher.open_neuron_viewer("/some/bundle", None)
        assert seen["neurons"] == []

    def test_view_neurons_entry_point_defaults_mesh_to_false(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False: seen.update(mesh=mesh))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"])
        assert seen["mesh"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_render_review_gui.py -v -k view_neurons`
Expected: FAIL with `AttributeError: module 'launcher' has no attribute 'open_neuron_viewer'`

- [ ] **Step 3: Add `open_neuron_viewer` to `launcher.py`**

Insert right after `open_render_window` (after its closing `render_review.run(source=source,
neurons=list(neurons or []))` line, before `def run() -> None:`):

```python
def open_neuron_viewer(source: str, neurons, *, mesh: bool = False) -> None:
    """Hand the currently picked source, ticked neurons, and mesh choice to
    view_neurons.run.

    Same reasoning as open_render_window: a named function rather than a closure
    inside run(), so it can be tested without Qt.
    """
    import view_neurons
    view_neurons.run(source, list(neurons or []), mesh=mesh)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_render_review_gui.py -v`
Expected: all PASS, including the 4 new tests and every pre-existing test in the file.

- [ ] **Step 5: Wire the "View neurons" button and "3D mesh" checkbox into the launcher window**

In `launcher.py`'s `run()` function, the import line currently reads:

```python
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                                QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                QListWidgetItem, QPushButton, QTextEdit, QVBoxLayout,
                                QWidget)
```

This already imports `QCheckBox` and `QPushButton`, so no import change is needed.

Replace this block (the two buttons after `render_btn`, currently):

```python
    render_btn = QPushButton("Render video + mesh")
    layout.addWidget(render_btn)
    check_btn = QPushButton("Check this machine")
    layout.addWidget(check_btn)
```

with:

```python
    render_btn = QPushButton("Render video + mesh")
    layout.addWidget(render_btn)
    view_row = QHBoxLayout()
    view_btn = QPushButton("View neurons")
    mesh_checkbox = QCheckBox("3D mesh")
    view_row.addWidget(view_btn, 1)
    view_row.addWidget(mesh_checkbox)
    layout.addLayout(view_row)
    check_btn = QPushButton("Check this machine")
    layout.addWidget(check_btn)
```

Then add a `do_view` handler right after `do_render` (after its closing
`open_render_window(path_edit.text().strip(), ticked())` line):

```python
    def do_view(*_):
        open_neuron_viewer(path_edit.text().strip(), ticked(), mesh=mesh_checkbox.isChecked())
```

Then wire it alongside the existing connections. Replace:

```python
    browse.clicked.connect(pick)
    path_edit.editingFinished.connect(refresh)
    launch_btn.clicked.connect(do_launch)
    render_btn.clicked.connect(do_render)
    check_btn.clicked.connect(do_check)
```

with:

```python
    browse.clicked.connect(pick)
    path_edit.editingFinished.connect(refresh)
    launch_btn.clicked.connect(do_launch)
    render_btn.clicked.connect(do_render)
    view_btn.clicked.connect(do_view)
    check_btn.clicked.connect(do_check)
```

- [ ] **Step 6: Relabel the frames-cache field and add the data-root Browse fallback for Raw EM**

The `_path_row` helper currently reads:

```python
    def _path_row(label: str, value: str, caption: str):
        """A label + line edit + Browse button, added to the layout. Returns the edit.

        Four of these now exist; without a helper the window is the same six lines
        four times over, and the fourth one drifts."""
        row = QHBoxLayout()
        edit = QLineEdit(value)
        btn = QPushButton("Browse...")

        def _pick(*_):
            chosen = QFileDialog.getExistingDirectory(win, caption,
                                                      edit.text() or str(Path.home()))
            if chosen:
                edit.setText(chosen)
                # setText() emits textChanged, never editingFinished, which is all
                # the mode gate below is wired to. Without this, Browse-ing a good
                # checkpoint folder left full mode disabled (it can now run), and
                # Browse-ing an unwritable one after full mode was already enabled
                # left it enabled and selected: the exact "selected but not
                # runnable" state the gate exists to prevent. refresh_mode_gate is
                # defined later in this function but resolved at call time, by
                # which point it exists (this closure only runs after a click).
                refresh_mode_gate()

        btn.clicked.connect(_pick)
        row.addWidget(QLabel(label))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit
```

Replace it with (adds one optional parameter, `root_fallback`, used only by the Raw EM row below):

```python
    def _path_row(label: str, value: str, caption: str, *, root_fallback=None):
        """A label + line edit + Browse button, added to the layout. Returns the edit.

        Four of these now exist; without a helper the window is the same six lines
        four times over, and the fourth one drifts.

        root_fallback, when given, is a zero-argument callable returning a directory
        to start the Browse dialog in when this field is empty and the data-root
        field (below) is not. Resolved at click time, the same "defined later,
        resolved later" idiom refresh_mode_gate already relies on in this closure
        (see the comment inside _pick), not evaluated when _path_row is called."""
        row = QHBoxLayout()
        edit = QLineEdit(value)
        btn = QPushButton("Browse...")

        def _pick(*_):
            fallback = (root_fallback() if root_fallback else "") or str(Path.home())
            chosen = QFileDialog.getExistingDirectory(win, caption,
                                                      edit.text() or fallback)
            if chosen:
                edit.setText(chosen)
                # setText() emits textChanged, never editingFinished, which is all
                # the mode gate below is wired to. Without this, Browse-ing a good
                # checkpoint folder left full mode disabled (it can now run), and
                # Browse-ing an unwritable one after full mode was already enabled
                # left it enabled and selected: the exact "selected but not
                # runnable" state the gate exists to prevent. refresh_mode_gate is
                # defined later in this function but resolved at call time, by
                # which point it exists (this closure only runs after a click).
                refresh_mode_gate()

        btn.clicked.connect(_pick)
        row.addWidget(QLabel(label))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit
```

Then replace the three `_path_row` call sites (currently):

```python
    layout.addWidget(QLabel("This machine (needed for recrop and SAM2):"))
    worm_edit = _path_row("Raw EM (tif stack):", profile.get("worm_path", ""),
                          "Pick the raw EM tif stack")
    frames_edit = _path_row("Frames cache:", profile.get("frames_root", ""),
                            "Pick a folder for prepared frames")
    ckpt_edit = _path_row("Checkpoints:", profile.get("checkpoint_dir", ""),
                          "Pick the SAM2 checkpoint folder")
```

with:

```python
    layout.addWidget(QLabel("This machine (needed for recrop and SAM2):"))
    frames_edit = _path_row("Data root (frames cache + scratch):", profile.get("frames_root", ""),
                            "Pick a folder for prepared frames")
    worm_edit = _path_row("Raw EM (tif stack):", profile.get("worm_path", ""),
                          "Pick the raw EM tif stack", root_fallback=lambda: frames_edit.text())
    ckpt_edit = _path_row("Checkpoints:", profile.get("checkpoint_dir", ""),
                          "Pick the SAM2 checkpoint folder")
```

`frames_edit` is reordered before `worm_edit` here (`frames_edit` is still assigned to the same
profile key, `frames_root`, untouched) purely so the Browse-fallback reads top-to-bottom; the
lambda would resolve correctly either order, per the click-time-resolution note above, but putting
the field a fallback depends on before the field that depends on it avoids the question entirely
for a future reader.

The `_checkpoint`/`_frames`/`_raw_em` functions inside `machine_checks` and every other reference
to `"frames_root"`/`"Frames cache"` as an internal name are unchanged; only this one visible label
string changes.

- [ ] **Step 7: Add the Bundle/output-tree Browse fallback**

Replace this (the top-level `pick` function):

```python
    def pick(*_):
        chosen = QFileDialog.getExistingDirectory(win, "Pick a bundle or output tree",
                                                  path_edit.text() or str(Path.home()))
        if chosen:
            path_edit.setText(chosen)
            refresh()
```

with:

```python
    def pick(*_):
        fallback = frames_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(win, "Pick a bundle or output tree",
                                                  path_edit.text() or fallback)
        if chosen:
            path_edit.setText(chosen)
            refresh()
```

- [ ] **Step 8: Run the full test suite**

Run: `py -3 -m pytest tests/test_render_review_gui.py tests/test_launcher_config.py tests/test_view_neurons.py -v`
Expected: all PASS. `tests/test_launcher_config.py` in particular must be unaffected, since none
of its tests touch the window's widget labels or Browse dialogs, only the pure
profile/env/machine-check functions, which this task does not change.

- [ ] **Step 9: Manual smoke check**

This task's Qt widget wiring (the button, the checkbox, the relabeled field, the Browse-dialog
fallback) has no existing automated-test precedent in this file for widget-level behavior (only
the pure functions around the window are tested; see this file's own module docstring). Run:

```bash
py -3 launcher.py
```

Confirm by eye: the field above "Checkpoints:" now reads "Data root (frames cache + scratch):"
(previously "Frames cache:"); a "View neurons" button and a "3D mesh" checkbox appear below
"Render video + mesh"; picking a bundle/tree with neurons ticked and clicking "View neurons" opens
the same napari window `view_neurons.py --tree ... --neurons ...` would from the command line;
ticking "3D mesh" first also shows mesh layers. Report what you saw; this step does not get
checked off until a human (or an equivalent real run) confirms the window actually opened.

- [ ] **Step 10: Commit**

```bash
git add launcher.py tests/test_render_review_gui.py
git commit -m "feat: launcher View neurons button and shared data-root Browse fallback"
```
