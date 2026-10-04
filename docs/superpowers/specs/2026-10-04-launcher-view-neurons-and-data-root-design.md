# launcher.py: a view_neurons.py button, and a shared data-root convenience

Status: design, approved 2026-10-04.

## Why

`launcher.py` was the deferred "integrated launcher" idea from
[[post-segmentation-visualization-and-contact-mapping-goals]]: one place to set paths and launch
every tool instead of separate CLI commands. Checking the actual current file against that idea
found it already covers almost all of it: `output_root`, `frames_root`, `worm_path`, and
`checkpoint_dir` are all there, with machine preflight checks and buttons for "Launch review"
(`gui.py`) and "Render video + mesh" (`render_review.py`). "Manual verify" trees are not a separate
structural concept anywhere in the code either; they are opened through the same "Bundle or output
tree" field as any other tree.

Two real gaps remain. `view_neurons.py` (this session's multi-neuron 2D/3D viewer, now with a
saved package and mesh caching) has no button at all. And `worm_path`/`frames_root` have to be
re-pointed by hand whenever the data drive changes, exactly what just happened (the F: drive
failure, mid-recovery to E: per [[f-drive-seagate-recovery-2026-10]]), because nothing in the code
or the UI reflects that they typically live under one shared root.

## Piece 1: a `view_neurons.py` button

`view_neurons.py`'s `main()` currently does argument parsing, then validation, then building specs,
then opening napari, all in one function driven by `args.*`. This splits out everything after
parsing into a new function with the same shape `render_review.run(source, neurons)` already has
(the function `launcher.py`'s existing "Render video + mesh" button calls):

```python
def run(source, neurons: list[str], *, mesh: bool = False) -> None:
```

`main()` becomes a thin wrapper: parse argv, then call `run(source=args.tree, neurons=[...],
mesh=args.mesh)` for the `--tree`/`--neurons` path, keeping the `--package`/`--save-package`
validation and dispatch in `main()` itself (those are CLI-only concerns this button does not use).
`run()` contains exactly the live-build-and-open logic `main()`'s `else` branch has today
(`build_layer_specs`, optionally `build_mesh_specs`, then the napari window), unchanged in
substance, just made callable with explicit arguments instead of only from argv.

`launcher.py` gets one new checkbox and one new button, added to the same row/area as the existing
"Render video + mesh" button:

- `mesh_checkbox = QCheckBox("3D mesh")`
- `view_btn = QPushButton("View neurons")`, wired to a new `open_neuron_viewer(source, neurons, *,
  mesh)` function (same naming pattern as the existing `open_render_window`), which does `import
  view_neurons; view_neurons.run(source=source, neurons=list(neurons or []), mesh=mesh)`.

No second window. The click handler reads the already-visible `path_edit.text().strip()` and
`ticked()`, exactly what `do_render` already does for the render button, plus the new checkbox's
state. Package save/load (`--save-package`/`--package`) stays command-line only; exposing that in
this UI is a reasonable follow-up, not built here, since it needs its own mode switch and path
field this task does not add.

## Piece 2: a shared data-root convenience, not a new field

`config.py`'s own real defaults show `WORM_PATH`, `OUTPUT_ROOT`, and `FRAMES_ROOT` all sitting
under one machine root (`F:\ZhenLab\Data` today) even though nothing in the code enforces that
relationship; `CHECKPOINT_DIR` defaults separately (relative `checkpoints`, shared model weights,
not project data, correctly left alone). `FRAMES_ROOT` in particular is not really
frames-specific in practice: `pipeline/frames.py`, `batch.py`, and every `experiments/*.py` script
read it as the general scratch root for this whole pipeline, and on this user's own machine its
value has always been identical to the EM data drive's root, not a dedicated "frames" subfolder.

This piece changes the window's label for the existing field (`frames_edit` /
`profile["frames_root"]`) from `"Frames cache:"` to `"Data root (frames cache + scratch):"`. The
profile key, the env var (`SAM2_FRAMES_ROOT`), `machine_checks`' internal check name ("frames
cache"), and every downstream reader of `config.FRAMES_ROOT` are all unchanged; this is a label
change only, so an existing `~/.sam2review/profile.json` from before this change still loads and
behaves identically. No new field, no new profile key: the point is that the "data root" concept
already exists under this name, just mislabeled for what it actually is.

That field's current text becomes the fallback starting directory for two Browse dialogs, used
only when the target field being browsed is currently empty (never overwriting an already-typed
value, the same "empty means unset, full stop" discipline `_checkpoint`/`_frames`/`_raw_em` already
use elsewhere in this file):

- The "Raw EM (tif stack)" Browse dialog (`worm_edit`'s `_pick`, inside the existing `_path_row`
  helper): starting directory becomes `edit.text() or frames_edit.text() or str(Path.home())`
  instead of today's `edit.text() or str(Path.home())`.
- The "Bundle or output tree" Browse dialog (`pick()`, the top-level path picker): starting
  directory becomes `path_edit.text() or frames_edit.text() or str(Path.home())` instead of
  today's `path_edit.text() or str(Path.home())`.

This directly targets the drive-migration pain: re-point the one "Data root" field, and the other
two pickers start from the new drive instead of the home folder, without silently rewriting a path
the user already set for a project that genuinely lives elsewhere.

## Non-goals

- No new profile key or env var. `frames_root`/`SAM2_FRAMES_ROOT` keeps its existing name and
  meaning; this is a label and a Browse-default, not a schema change.
- No change to `checkpoint_dir`, which is a genuinely separate concept (shared across projects,
  not project data) and was never part of the "lives under one root" pattern.
- No automatic re-pointing of an already-set `worm_path` or `output_root` when the data root
  changes. Only empty fields get the new fallback; a field with a real value is left alone.
- No package save/load controls in `launcher.py`. `view_neurons.py`'s `--save-package`/`--package`
  stay command-line only.
- No change to `render_review.py`'s own window or its button; it already works and is untouched by
  either piece.
