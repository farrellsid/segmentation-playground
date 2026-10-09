# view_neurons.run(cache=True): a write-through mesh/package cache for the launcher

Status: design, approved 2026-10-09.

## Why

`launcher.py`'s "View neurons" button always calls `view_neurons.run(source, neurons,
mesh=mesh)`, which always builds live: `build_layer_specs` (fast) and, when `mesh=True`,
`build_mesh_specs` (marching cubes, the slow part, confirmed several-minutes-scale on
real data earlier this session). `view_neurons.py` already has a full save/load package
mechanism (`save_package`/`load_package`/`load_mesh_specs`), but it is CLI-only
(`--save-package`/`--package`); the launcher button never touches it, explicitly scoped
out when that button was first built. This spec wires it in, so viewing the SAME tree a
second time is fast instead of re-running marching cubes every click.

## `run()` gets two new keyword-only parameters

```python
def run(source, neurons: list[str], *, mesh: bool = False,
       cache: bool = False, force_rebuild: bool = False) -> None:
```

Both default to `False`, so every existing caller (`main()`'s `--tree` path, every
existing test) is unaffected; this is purely additive. `launcher.py`'s `open_neuron_viewer`
is the only caller that passes `cache=True`, forwarding a new "Force rebuild" checkbox's
state as `force_rebuild`.

`force_rebuild=True` skips both the cache-read and the staleness check below entirely and
always rebuilds live, then overwrites the cache. It is the manual escape hatch for
anything the automatic check (next section) cannot see, such as a mask changed by a
process other than `gui.py` (a cluster reprop re-run, a hand-edited file).

## Cache location and format

`<source>/.view_package/`, the exact same format `save_package`/`load_package` already
write and read (`manifest.json` + `labels/` + optional `meshes/`). No new format; this
spec only adds a caller that picks the location automatically and decides when to use it.

## `save_package` gains a `saved_at` manifest field

One new key in the manifest dict `save_package` already builds:
`"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")`, the exact same
`_now()`-equivalent format `sam2_utils/review_queue.py`/`batch.py` already use elsewhere
in this codebase, written unconditionally (both 2D-only and mesh-included packages get
one). Every other manifest key is unchanged.

## Staleness check: a real signal, not a guess

`_review.csv` (read via `output_root / "_review.csv"`, the same path
`sam2_utils.review_queue.ReviewQueue` already uses) already carries a real "when was this
chain actually corrected" timestamp: `gui.py` calls `queue.set_status(..., CORRECTED,
...)` in the same action as `_save_and_qc(...)`, so a `corrected` row's `updated_at` is
stamped at the moment masks were last actually edited through the GUI, not at propagation
time (that is `_manifest.csv`'s separate, unrelated `updated_at`, batch-pipeline
bookkeeping, not a human-edit signal).

New function in `view_neurons.py`:

```python
def _cache_is_stale(cache_dir: Path, source, neurons: list[str], saved_at: str) -> bool:
    """Whether any of `neurons`' chains were corrected after this cache was saved.

    Reads `source`'s `_review.csv` directly (not via ReviewQueue, to avoid a
    sam2_utils.review_queue import in this module for one column read); absent file
    or no "corrected" rows for these neurons means no evidence anything changed, so
    the cache is trusted, not treated as stale by default.
    """
    review_path = Path(source) / "_review.csv"
    if not review_path.exists():
        return False
    import csv
    from datetime import datetime
    saved_dt = datetime.fromisoformat(saved_at)
    with review_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("neuron") in neurons and row.get("review_status") == "corrected"
                    and datetime.fromisoformat(row["updated_at"]) > saved_dt):
                return True
    return False
```

## The cached entry point

A new private helper, `_run_cached`, called by `run()` when `cache=True`:

```python
def _run_cached(source, neurons: list[str], *, mesh: bool, force_rebuild: bool) -> None:
    cache_dir = Path(source) / ".view_package"
    manifest_path = cache_dir / "manifest.json"
    if not force_rebuild and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        has_mesh = bool(manifest.get("meshes"))
        fresh = not _cache_is_stale(cache_dir, source, neurons, manifest["saved_at"])
        if fresh and (not mesh or has_mesh):
            specs = load_package(cache_dir)
            mesh_specs = load_mesh_specs(cache_dir) if mesh else []
            _open_viewer(specs, mesh_specs)
            return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    save_package(specs, cache_dir, built_from={"tree": str(source), "neurons": neurons},
                mesh_specs=mesh_specs if mesh else None)
    _open_viewer(specs, mesh_specs)
```

`run()`'s body becomes:

```python
def run(source, neurons: list[str], *, mesh: bool = False,
       cache: bool = False, force_rebuild: bool = False) -> None:
    if cache:
        _run_cached(source, neurons, mesh=mesh, force_rebuild=force_rebuild)
        return
    specs, mesh_specs = _build_from_tree(source, neurons, mesh=mesh)
    _open_viewer(specs, mesh_specs)
```

A manifest saved by an OLDER version of this tool (before `saved_at` existed) is treated
as stale by construction: `datetime.fromisoformat(manifest["saved_at"])` raises `KeyError`
on a missing key, which `_run_cached` lets propagate as a hard failure rather than
silently guessing a value, since a cache this tool never wrote `saved_at` for predates this
feature entirely and should just be rebuilt once by hand (delete `.view_package/`) rather
than have this code grow a migration path for a one-time local cache directory.

## `launcher.py`: one new checkbox

A "Force rebuild" `QCheckBox` next to the existing "3D mesh" one. `do_view` becomes:

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

`open_neuron_viewer` gains a matching `force_rebuild: bool = False` parameter, forwarded
straight to `view_neurons.run(..., cache=True, force_rebuild=force_rebuild)`. `cache=True`
is hardcoded in `open_neuron_viewer`, not exposed as its own checkbox: the launcher's use
case (the same tree, reviewed session after session) is exactly what caching is for, and
exposing a raw on/off toggle for it would just be one more control for "always leave this
on."

## Non-goals

- No staleness detection for anything other than `gui.py`-driven corrections
  (`_review.csv`'s `corrected` rows). A mask changed by a cluster reprop re-run or a hand
  edit is invisible to `_cache_is_stale`; `force_rebuild` is the answer for that, not more
  automatic detection.
- No migration for a pre-`saved_at` cache; delete `.view_package/` by hand once.
- No change to the CLI (`main()`'s `--save-package`/`--package` flags, or `--tree` without
  `--save-package`): `cache`/`force_rebuild` are launcher-only parameters on `run()`,
  `main()` never passes them.
- No per-neuron granularity. The whole package (all neurons currently ticked) is
  considered stale or fresh together, matching how `save_package` already treats a
  package as one unit, not a per-neuron cache.
