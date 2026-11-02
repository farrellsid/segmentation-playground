# view_neurons Cached run() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `launcher.py`'s "View neurons" button reuses a saved package (including mesh
data) on repeat views of the same tree, instead of rebuilding live every click, with a
real staleness check against `_review.csv` plus a manual "Force rebuild" escape hatch.

**Architecture:** Two new keyword-only parameters on `view_neurons.run()`
(`cache`/`force_rebuild`), a new private write-through helper (`_run_cached`), a new
staleness check (`_cache_is_stale`) reading the existing `_review.csv` ledger, and one new
manifest field (`saved_at`) on `save_package`. `launcher.py` gets a "Force rebuild"
checkbox and forwards both new parameters.

**Tech Stack:** Python, this repo's existing `view_neurons.py`/`launcher.py`.

## Global Constraints

- `run()`'s existing signature stays backward compatible: `cache`/`force_rebuild` both
  default to `False`, so `main()`'s `--tree` CLI path and every existing test of `run()`
  is unaffected.
- Cache location is `<source>/.view_package/`, the exact format `save_package`/
  `load_package`/`load_mesh_specs` already read and write. No new on-disk format.
- `_cache_is_stale` only consults `<source>/_review.csv` rows with
  `review_status == "corrected"`; a `_review.csv` with no such row, or no file at all,
  means "no evidence of a change", not "assume stale".
- `force_rebuild=True` skips the cache-read and staleness check entirely and always
  rebuilds, then overwrites the cache.
- Requesting `mesh=True` against a cache that exists but has no `"meshes"` key must
  rebuild (with mesh this time) and overwrite the cache, not silently serve a
  mesh-less view or raise.

---

### Task 1: `view_neurons.py` — `saved_at`, `_cache_is_stale`, `_run_cached`, `run(cache=...)`

**Files:**
- Modify: `view_neurons.py`
- Test: `tests/test_view_neurons.py`

**Interfaces:**
- Produces: `run(source, neurons, *, mesh: bool = False, cache: bool = False,
  force_rebuild: bool = False) -> None` (extended signature). `_cache_is_stale(source,
  neurons: list[str], saved_at: str) -> bool` and `_run_cached(source, neurons: list[str],
  *, mesh: bool, force_rebuild: bool) -> None`, both private, used only by `run()`.
  `save_package(...)`'s written manifest gains a `"saved_at"` key (every caller's return
  value is unaffected; nothing currently reads the manifest's key set exhaustively).

- [ ] **Step 1: Write the failing tests**

Add `import csv` is NOT needed in the test file (only in `view_neurons.py` itself); the
tests below write `_review.csv` directly with `Path.write_text`. Add to
`tests/test_view_neurons.py`, right after `test_run_raises_when_no_neuron_has_any_masks`
(after line 279) and before `test_save_and_load_package_round_trips_names_and_placement`:

```python
def test_cache_is_stale_true_when_a_corrected_row_is_newer(tmp_path):
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        "AIAL,0,corrected,L,,2026-10-09T12:00:00+00:00\n", encoding="utf-8")
    assert view_neurons._cache_is_stale(tmp_path, ["AIAL"],
                                        "2026-10-09T10:00:00+00:00") is True


def test_cache_is_stale_false_when_the_corrected_row_is_older(tmp_path):
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        "AIAL,0,corrected,L,,2026-10-09T08:00:00+00:00\n", encoding="utf-8")
    assert view_neurons._cache_is_stale(tmp_path, ["AIAL"],
                                        "2026-10-09T10:00:00+00:00") is False


def test_cache_is_stale_false_when_no_review_csv_exists(tmp_path):
    assert view_neurons._cache_is_stale(tmp_path, ["AIAL"],
                                        "2026-10-09T10:00:00+00:00") is False


def test_cache_is_stale_ignores_a_different_neurons_correction(tmp_path):
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        "AIAR,0,corrected,L,,2026-10-09T12:00:00+00:00\n", encoding="utf-8")
    assert view_neurons._cache_is_stale(tmp_path, ["AIAL"],
                                        "2026-10-09T10:00:00+00:00") is False


def test_cache_is_stale_ignores_a_non_corrected_status(tmp_path):
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        "AIAL,0,approved,L,,2026-10-09T12:00:00+00:00\n", encoding="utf-8")
    assert view_neurons._cache_is_stale(tmp_path, ["AIAL"],
                                        "2026-10-09T10:00:00+00:00") is False


def test_save_package_records_saved_at(tmp_path):
    specs = _two_specs()
    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    manifest = json.loads((out_dir / "manifest.json").read_text())
    from datetime import datetime
    datetime.fromisoformat(manifest["saved_at"])  # must not raise


def test_run_cache_true_reuses_the_package_on_a_second_call(tmp_path, monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))
    calls = {"n": 0}
    real_build = view_neurons._build_from_tree
    def counting_build(*a, **kw):
        calls["n"] += 1
        return real_build(*a, **kw)
    monkeypatch.setattr(view_neurons, "_build_from_tree", counting_build)

    view_neurons.run(str(tmp_path), ["AIAL"], cache=True)
    view_neurons.run(str(tmp_path), ["AIAL"], cache=True)

    assert calls["n"] == 1, "second call must load from the cache, not rebuild"
    assert (tmp_path / ".view_package" / "manifest.json").exists()


def test_run_cache_true_rebuilds_when_mesh_newly_requested(tmp_path, monkeypatch):
    vol = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), spacing),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))
    calls = {"n": 0}
    real_build = view_neurons._build_from_tree
    def counting_build(*a, **kw):
        calls["n"] += 1
        return real_build(*a, **kw)
    monkeypatch.setattr(view_neurons, "_build_from_tree", counting_build)

    view_neurons.run(str(tmp_path), ["AIAL"], cache=True, mesh=False)
    view_neurons.run(str(tmp_path), ["AIAL"], cache=True, mesh=True)

    assert calls["n"] == 2, "mesh requested but cache lacked it must trigger a rebuild"
    manifest = json.loads((tmp_path / ".view_package" / "manifest.json").read_text())
    assert manifest.get("meshes"), "the rebuilt cache must now include mesh data"


def test_run_cache_true_force_rebuild_ignores_a_fresh_cache(tmp_path, monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))
    calls = {"n": 0}
    real_build = view_neurons._build_from_tree
    def counting_build(*a, **kw):
        calls["n"] += 1
        return real_build(*a, **kw)
    monkeypatch.setattr(view_neurons, "_build_from_tree", counting_build)

    view_neurons.run(str(tmp_path), ["AIAL"], cache=True)
    view_neurons.run(str(tmp_path), ["AIAL"], cache=True, force_rebuild=True)

    assert calls["n"] == 2, "force_rebuild must skip the cache even when fresh"


def test_run_cache_true_rebuilds_when_a_correction_postdates_the_cache(tmp_path, monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))
    holder = {}
    monkeypatch.setitem(sys.modules, "napari", _fake_napari_module(holder))
    calls = {"n": 0}
    real_build = view_neurons._build_from_tree
    def counting_build(*a, **kw):
        calls["n"] += 1
        return real_build(*a, **kw)
    monkeypatch.setattr(view_neurons, "_build_from_tree", counting_build)

    view_neurons.run(str(tmp_path), ["AIAL"], cache=True)
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec="seconds")
    (tmp_path / "_review.csv").write_text(
        "neuron,chain_idx,review_status,reviewer,notes,updated_at\n"
        f"AIAL,0,corrected,L,,{future}\n", encoding="utf-8")

    view_neurons.run(str(tmp_path), ["AIAL"], cache=True)

    assert calls["n"] == 2, "a correction newer than the cache must trigger a rebuild"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_view_neurons.py -v -k "cache_is_stale or saved_at or run_cache"`
Expected: FAIL — the `_cache_is_stale`/`saved_at` tests with `AttributeError`/`KeyError`
(no such function, no such manifest key), the `run_cache_true` tests with `TypeError:
run() got an unexpected keyword argument 'cache'`.

- [ ] **Step 3: Add `csv`/`datetime`/`timezone` imports**

Add to the top-level import block (after `import sys`, before `from pathlib import Path`):

```python
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
```

(This replaces the current `import sys`/`from pathlib import Path` two lines with the
same two lines plus the two new ones, keeping the existing alphabetical-ish grouping.)

- [ ] **Step 4: Add `saved_at` to `save_package`'s manifest**

Replace (currently):

```python
    manifest = {
        "schema_version": 1,
        "built_from": built_from,
        "origin": [z0, y0, x0],
        "spacing": list(spacing),
        "shape": list(shape),
        "neurons": neuron_entries,
    }
```

with:

```python
    manifest = {
        "schema_version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_from": built_from,
        "origin": [z0, y0, x0],
        "spacing": list(spacing),
        "shape": list(shape),
        "neurons": neuron_entries,
    }
```

- [ ] **Step 5: Add `_cache_is_stale`**

Insert right after `_open_viewer` (after its closing `napari.run()` line, before
`def run(...)`):

```python
def _cache_is_stale(source, neurons: list[str], saved_at: str) -> bool:
    """Whether any of `neurons`' chains were corrected (through gui.py) after this
    cache was saved.

    Reads `source`'s _review.csv directly (not via sam2_utils.review_queue, to avoid
    that import for one column read). An absent file or no "corrected" row for these
    neurons means no evidence anything changed, so the cache is trusted, not treated
    as stale by default. Only a "corrected" status counts: "approved"/"rejected" do
    not change mask pixels.
    """
    review_path = Path(source) / "_review.csv"
    if not review_path.exists():
        return False
    saved_dt = datetime.fromisoformat(saved_at)
    with review_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("neuron") in neurons and row.get("review_status") == "corrected"
                    and datetime.fromisoformat(row["updated_at"]) > saved_dt):
                return True
    return False
```

- [ ] **Step 6: Add `_run_cached` and extend `run()`**

Replace (currently):

```python
def run(source, neurons: list[str], *, mesh: bool = False) -> None:
    """Build and open the viewer directly, the same shape render_review.run(source,
    neurons) already has (the function launcher.py's "Render video + mesh" button
    calls). This is what launcher.py's "View neurons" button calls.

    --save-package/--package are CLI-only concerns this entry point does not take;
    see main() below for those.
    """
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    _open_viewer(specs, mesh_specs)
```

with:

```python
def _run_cached(source, neurons: list[str], *, mesh: bool, force_rebuild: bool) -> None:
    """The cache=True path run() delegates to: reuse <source>/.view_package/ when it
    exists, is not stale, and (if mesh is requested) already has mesh data; otherwise
    build live and overwrite the cache so the NEXT call is fast.
    """
    cache_dir = Path(source) / ".view_package"
    manifest_path = cache_dir / "manifest.json"
    if not force_rebuild and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        has_mesh = bool(manifest.get("meshes"))
        fresh = not _cache_is_stale(source, neurons, manifest["saved_at"])
        if fresh and (not mesh or has_mesh):
            specs = load_package(cache_dir)
            mesh_specs = load_mesh_specs(cache_dir) if mesh else []
            _open_viewer(specs, mesh_specs)
            return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    save_package(specs, cache_dir, built_from={"tree": str(source), "neurons": neurons},
                mesh_specs=mesh_specs if mesh else None)
    _open_viewer(specs, mesh_specs)


def run(source, neurons: list[str], *, mesh: bool = False,
       cache: bool = False, force_rebuild: bool = False) -> None:
    """Build and open the viewer directly, the same shape render_review.run(source,
    neurons) already has (the function launcher.py's "Render video + mesh" button
    calls). This is what launcher.py's "View neurons" button calls.

    cache=True (launcher-only; main()'s --tree CLI path never sets it) reuses a saved
    package at <source>/.view_package/ instead of always rebuilding live; see
    _run_cached. force_rebuild=True skips the cache and staleness check entirely.

    --save-package/--package are CLI-only concerns this entry point does not take;
    see main() below for those.
    """
    if cache:
        _run_cached(source, neurons, mesh=mesh, force_rebuild=force_rebuild)
        return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    _open_viewer(specs, mesh_specs)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_view_neurons.py -v`
Expected: all PASS, including every new test from Step 1 and every pre-existing test in
the file (confirms `cache=False`'s default path, every existing `run()`/`save_package`
test, stays byte-identical).

- [ ] **Step 8: Commit**

```bash
git add view_neurons.py tests/test_view_neurons.py
git commit -m "feat: write-through package cache for view_neurons.run(cache=True)"
```

---

### Task 2: `launcher.py` — a "Force rebuild" checkbox

**Files:**
- Modify: `launcher.py`
- Test: `tests/test_render_review_gui.py`

**Interfaces:**
- Consumes: `view_neurons.run(source, neurons, *, mesh=False, cache=False,
  force_rebuild=False)` (Task 1).
- Produces: `open_neuron_viewer(source: str, neurons, *, mesh: bool = False,
  force_rebuild: bool = False) -> None` (extended signature).

- [ ] **Step 1: Write the failing tests**

The 3 existing tests need their stub lambda's parameter list widened from
`lambda source="", neurons=None, mesh=False: ...` to `lambda source="", neurons=None,
mesh=False, cache=False, force_rebuild=False: ...`, since `open_neuron_viewer` will now
always pass `cache=True`, which the old 3-parameter lambda would reject with
`TypeError`. Each test's own `seen.update(...)` body is unchanged. Replace all three in
full:

```python
    def test_view_neurons_entry_point_hands_its_source_and_neurons_across(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(
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
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(neurons=neurons))
        launcher.open_neuron_viewer("/some/bundle", None)
        assert seen["neurons"] == []

    def test_view_neurons_entry_point_defaults_mesh_to_false(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(mesh=mesh))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"])
        assert seen["mesh"] is False
```

Then add, right after `test_view_neurons_entry_point_defaults_mesh_to_false` (after line
137) and before the file's closing blank line / `TestLauncherIntegration` class end:

```python
    def test_view_neurons_entry_point_always_requests_the_cache(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(cache=cache))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"])
        assert seen["cache"] is True

    def test_view_neurons_entry_point_forwards_force_rebuild(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(
                                      force_rebuild=force_rebuild))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"], force_rebuild=True)
        assert seen["force_rebuild"] is True

    def test_view_neurons_entry_point_defaults_force_rebuild_to_false(self, monkeypatch):
        import launcher
        import view_neurons
        seen = {}
        monkeypatch.setattr(view_neurons, "run",
                            lambda source="", neurons=None, mesh=False, cache=False,
                                  force_rebuild=False: seen.update(
                                      force_rebuild=force_rebuild))
        launcher.open_neuron_viewer("/some/bundle", ["AIBL"])
        assert seen["force_rebuild"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_render_review_gui.py -v -k view_neurons`
Expected: the pre-existing 3 tests FAIL first (`TypeError`, since `open_neuron_viewer`
does not yet pass `cache=True`, so the updated lambda's `cache` parameter never gets
exercised the way the new assertions below expect) and the 3 new tests FAIL with
`TypeError: open_neuron_viewer() got an unexpected keyword argument 'force_rebuild'`.

- [ ] **Step 3: Extend `open_neuron_viewer`**

Replace (currently):

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

with:

```python
def open_neuron_viewer(source: str, neurons, *, mesh: bool = False,
                       force_rebuild: bool = False) -> None:
    """Hand the currently picked source, ticked neurons, and mesh choice to
    view_neurons.run.

    Always passes cache=True: the launcher's use case (the same tree, reviewed session
    after session) is exactly what the write-through cache is for, so this is not a
    separate on/off toggle, unlike force_rebuild, which the "Force rebuild" checkbox
    does expose.

    Same reasoning as open_render_window: a named function rather than a closure
    inside run(), so it can be tested without Qt.
    """
    import view_neurons
    view_neurons.run(source, list(neurons or []), mesh=mesh, cache=True,
                     force_rebuild=force_rebuild)
```

- [ ] **Step 4: Add the "Force rebuild" checkbox**

Replace (currently):

```python
    view_row = QHBoxLayout()
    view_btn = QPushButton("View neurons")
    mesh_checkbox = QCheckBox("3D mesh")
    view_row.addWidget(view_btn, 1)
    view_row.addWidget(mesh_checkbox)
    layout.addLayout(view_row)
```

with:

```python
    view_row = QHBoxLayout()
    view_btn = QPushButton("View neurons")
    mesh_checkbox = QCheckBox("3D mesh")
    force_rebuild_checkbox = QCheckBox("Force rebuild")
    view_row.addWidget(view_btn, 1)
    view_row.addWidget(mesh_checkbox)
    view_row.addWidget(force_rebuild_checkbox)
    layout.addLayout(view_row)
```

- [ ] **Step 5: Wire it into `do_view`**

Replace (currently):

```python
    def do_view(*_):
        neurons = ticked()
        if not neurons:
            status.setText("Tick at least one neuron to view")
            return
        try:
            open_neuron_viewer(path_edit.text().strip(), neurons, mesh=mesh_checkbox.isChecked())
        except SystemExit as exc:
            status.setText(str(exc))
```

with:

```python
    def do_view(*_):
        neurons = ticked()
        if not neurons:
            status.setText("Tick at least one neuron to view")
            return
        try:
            open_neuron_viewer(path_edit.text().strip(), neurons,
                               mesh=mesh_checkbox.isChecked(),
                               force_rebuild=force_rebuild_checkbox.isChecked())
        except SystemExit as exc:
            status.setText(str(exc))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_render_review_gui.py tests/test_launcher_config.py tests/test_view_neurons.py -v`
Expected: all PASS.

- [ ] **Step 7: Manual smoke check**

Same reasoning as the earlier "View neurons" button's own Step 9: this file's widget
wiring has no automated test precedent beyond the pure functions. Run `py -3
launcher.py`, pick a real tree with at least one neuron, click "View neurons" twice in a
row (second click should open noticeably faster than the first), tick "Force rebuild"
and click a third time (should be slow again, like the first click), confirm the window
opens correctly each time. Report what you saw.

- [ ] **Step 8: Commit**

```bash
git add launcher.py tests/test_render_review_gui.py
git commit -m "feat: launcher Force rebuild checkbox, always caches View neurons"
```
