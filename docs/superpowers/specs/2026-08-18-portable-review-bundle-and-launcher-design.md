# Portable review: a neuron identity registry, a self-contained bundle, and a launcher

Status: design, approved 2026-08-18.

## Why

A second reviewer (Lucinda) is joining, and the division of labour is settled. This repo's owner
generates the initial masks and re-propagates the chains whose seeds were broken. Lucinda does the
final manual verification of what comes out the other side, which in practice means redrawing
boundaries by hand.

Three facts about her setup drive everything below. She is on a Mac. She has little local compute,
so her main tools are the paint and lasso brushes rather than the model. She still wants SAM2/SAM3
re-propagation available in case she moves to a bigger machine, so the model path cannot simply be
dropped from her build.

Nothing in the repo supports that today. Reviewing means running `gui.py` on this Windows box,
against `F:`, with the flags memorised. Three things block a second machine:

1. Paths live in tracked source. `sam2_utils/config.py` hardcodes `OUTPUT_ROOT` and `FRAMES_ROOT` as
   `F:\...` literals, and only `WORM_PATH` has an environment-variable override. A second machine
   therefore edits a git-tracked file and picks up a merge conflict on every pull.
2. A chain's frame path is absolute and baked in. `state.json` records `frames_dir` as an absolute
   path at generation time. It will not resolve on another machine, and `gui.py`'s fallback
   (`_ensure_local_frames`, gui.py:238) then regenerates the frames from the raw EM tif store. Those
   frames run about 255 MB each at full resolution, so that store will not be sitting on a laptop.
3. Neuron identity is not data. A mask's identity is the name of the directory holding it, and
   `experiments/dense_overlay.py:91` assigns numeric ids as
   `{n: i + 1 for i, n in enumerate(neurons)}`. A neuron's id is whichever position it landed in for
   that particular render, so rendering a different subset moves the numbers. That cannot support
   picking specific neurons to display, and it is worse for a future Blender or VAST export, where a
   segment number has to mean the same thing every time.

Two findings from the code make the fix smaller than it first looks. Both were checked directly
rather than assumed.

The model already loads lazily: `ReviewContext.ensure_predictors` has exactly three callers
(`rerun_image_phase`, `resume_propagation`, and the recrop path), so opening a chain, scrubbing,
painting, lassoing and saving never reach it. The review path is also torch-free, since `gui.py`'s
module-level imports pull no torch and neither does `import pipeline` (verified by importing it and
inspecting `sys.modules`); `napari` is imported inside `ReviewGUI.__init__` and `sam2_utils.setup`
inside `ensure_predictors`.

A redraw-only install with no torch at all is therefore viable, which removes the worst part of
setting SAM2 up on macOS.

## Scope

This spec covers the identity registry, the mask metadata sidecar, the review bundle and its return
trip, the launcher, and the small set of `gui.py` changes those need.

It does not cover the Blender exporter, the VAST exporter, or the per-frame labelmap exporter. Those
get a later spec. They are the reason the registry exists and they will consume it, but designing
them needs research into both target formats first. Verifying the re-propagation driver is also out
of scope and stays a separate task.

## 1. Neuron identity registry

`data/neuron_registry.csv`, tracked in git, append-only:

```
neuron_id,cell_name,first_seen,notes
1,ADAL,2026-08-18,
2,ADAR,2026-08-18,
...
```

The first build enumerates the distinct `cell_name` values in `data/chains.json` (5206 chain records
today) sorted by name, so the initial assignment is reproducible. After that the order stops
mattering and the ids are frozen. A neuron that shows up later appends with the next free id, and
nothing is ever renumbered, since a renumber would silently invalidate every export already
produced.

Access goes through `sam2_utils/registry.py`, kept import-light in the same style as `config.py` (no
torch, no cv2, no network):

```python
load_registry() -> dict[str, int]      # cell_name -> neuron_id
neuron_id(cell_name: str) -> int
neuron_name(neuron_id: int) -> str
```

Ids are 1-based, leaving 0 reserved for background so the values drop straight into a `uint16`
labelmap later. `uint16` caps at 65535, comfortably above the neuron count.

A guard test pins the current mapping and fails if any existing row's id changes. Everything
downstream assumes these numbers are permanent, so that test is the one carrying the most weight
here.

`experiments/dense_overlay.py` switches from its enumerate-order ids to registry lookups.

## 2. Per-chain metadata sidecar

Each chain directory gains a `meta.json` beside its `state.json`:

```json
{
  "schema_version": 1,
  "neuron_id": 42,
  "cell_name": "AIAL",
  "chain_idx": 0,
  "mask_space": "_pcrop",
  "crop_window": {"origin_tif": [4096, 3712], "size_tif": [2048, 1792], "scale": 2},
  "z_range": [1402, 1490],
  "mask_scale": 2,
  "provenance": {
    "source_tree": "target_perslice_only_guard_sam3_merged",
    "backend": "sam3",
    "reprop_variant": "mask_seed",
    "exported": "2026-08-18T12:00:00Z"
  }
}
```

`mask_scale` is the mask's own downscale relative to full-resolution EM, which is not the same
number for every chain: a legacy `_sam` chain records the pipeline's `save_downscale` (8), while a
tier-2 `_pcrop` chain records the crop's own `chain_crop_scale` (2 in the example above, coarser
when `chain_crop_max_px` forced the read down). An exporter that assumed a single global mask scale
would misplace every tier-2 chain, so the field is per chain by design.

The schema is stable and self-contained so a consumer can read it without importing `pipeline`. That
matters because the Blender and VAST exporters must not couple themselves to pipeline internals; if
they do, every internal refactor breaks the export path.

Most of these values already exist inside `state.json`. `meta.json` is a versioned projection of
them rather than a competing source of truth, and `state.json` stays authoritative for anything the
pipeline itself reads back.

The pipeline writes it for new chains, and `backfill_meta.py` covers the trees that already exist,
since the corrected AIA/AIY trees predate this.

## 3. The review bundle

A bundle is a directory that can be zipped, sent, and opened on a machine that has none of this
project's data.

```
<bundle>/
  bundle.json                    schema version, chain index, registry snapshot
  neurons.csv                    the registry rows this bundle uses
  <neuron>/
    chain_00/
      meta.json
      state.json                 frames_dir rewritten RELATIVE
      qc.csv
      masks/mask_<z:04d>.png
      frames/00000.jpg ...       the redraw canvas
```

`export_bundle.py` builds it, as a root-level driver matching the existing convention of `batch.py`
and `gui.py`. The pure functions live in `sam2_utils/bundle.py`: build the index, rewrite paths,
validate a bundle, compute review progress.

On the frame canvas, each chain ships the frames it was actually propagated on. Tier-2 `_pcrop`
chains are already read at `chain_crop_scale=2` and capped at `chain_crop_max_px=1536`, which is
half resolution and works as a redraw canvas unchanged. Legacy `_sam` chains only have the scale-8
full frame, too coarse to draw a membrane on, so the exporter prebakes a scale-2 crop around the
mask extent for those. That puts every chain on a comparable footing to review. Half resolution is
the agreed target, and the prior hand-segmentation practice in this lab used quarter resolution, so
this sits at or above the established bar. The exporter needs the raw EM store once, at export time,
on this machine.

The path fix is what makes any of this work: `state.json`'s `frames_dir` is rewritten to the
relative `frames/`, and `gui.py` learns to resolve a relative `frames_dir` against the chain
directory. That stops the raw-tif fallback from firing on a machine with no raw tifs.

For the return trip, `import_bundle.py` merges a returned bundle back into the master tree. It moves
only what the reviewer can change: `masks/`, `qc.csv`, and the reviewer's rows in `_review.csv` and
`_labels.csv`. It refuses to run when the bundle's `schema_version` or the chain's identity does not
match the target, and it reports which chains changed instead of merging silently. Since her
corrections are the actual deliverable, the import half carries as much weight as the export half.

## 4. Launcher

`launcher.py` at the repo root, built on `qtpy`, which napari already depends on. That adds no new
dependency and runs on macOS.

One window, in the order a reviewer actually thinks:

1. Pick a bundle or an output tree, with recents remembered.
2. See the neurons and chains it holds, each with review progress read from `_review.csv`.
3. Choose which neurons to load, by tick box. This is the same subset mechanism that later serves
   presentation highlighting.
4. Choose a mode: redraw only (no model, no torch) or SAM2/SAM3 reprop enabled.
5. Launch, which constructs the `ReviewContext` and `ReviewGUI` directly rather than shelling out.

Settings persist to `~/.sam2review/profile.json`, outside the repo, so nothing a reviewer configures
can collide with git.

One supporting change goes with it: `sam2_utils/config.py` gains `SAM2_OUTPUT_ROOT` and
`SAM2_FRAMES_ROOT` environment overrides, mirroring the pattern `WORM_PATH` already uses for
`SAM2_WORM_PATH`. The launcher sets them from the profile. This is what ends config-file editing,
and it is worth doing even for people who never open the launcher.

The reprop toggle degrades honestly. When torch is missing or no usable device is present, the
option is disabled and says why, instead of letting someone pick it and hit an exception several
minutes into a session.

## 5. Changes to gui.py

The GUI Lucinda opens should show her the tools she can actually use and nothing else. Counting the
controls `_build_widgets` creates today, 10 of 28 are model or compute machinery: the prompt-label
combo, box draw, reset prompts, re-run image phase, resume propagation, and the five-control recrop
cluster (grow spin, recrop, pick region, confirm, cancel). Seven of the 16 keybindings are the same
story (`p`, `n`, `b`, `r`, `g`, `c`, `f`). On a Mac with no GPU every one of those is dead weight,
and worse than dead weight, because a reviewer cannot tell which controls are meant for her.

So the GUI gains a review mode, and the refactor needed to support it gets scoped by that
requirement rather than done speculatively.

`_build_widgets` is one linear block of about 110 lines today, which cannot express two widget sets.
It splits into four panel builders, navigation, drawing, model, and verdict, assembled by a single
method that consults the mode. `_bind_keys` splits the same way, so the model keys are simply not
registered in review mode instead of being registered and then refusing. Two modes ship:
`review` (navigation, drawing, verdict) and `full` (everything, the current behaviour and the
default, so nothing changes for existing use).

That is the whole structural change. This spec still does not split `ReviewGUI` as a class. The
panel split is driven by a feature that needs it; a wholesale decomposition is not, and it would put
a working tool at risk to serve a need nobody has articulated yet. If Lucinda later starts writing
features against the class, that is the point to revisit, with her real needs known instead of
guessed.

Alongside the mode, four smaller changes:

1. Resolve a relative `frames_dir` against the chain directory (section 3).
2. Add a neuron subset filter, so the launcher can open a session scoped to chosen neurons.
3. In `full` mode, model actions still degrade gracefully when torch is absent, reporting that the
   model is unavailable instead of raising. Review mode never reaches this path, but `full` mode on
   a machine without torch still needs to fail politely.
4. Extract only what the launcher and bundle need into importable functions, namely chain indexing,
   review progress, and path resolution. These have to be testable without opening a napari window,
   which is the reason to move them.

A torch-free `requirements-review.txt` and a short macOS setup document ship alongside.

The launcher's mode selector (section 4) sets this directly: redraw only maps to `review`, reprop
enabled maps to `full`.

## Testing

All tests torch-free and runnable in CI, matching the existing suite:

- `test_registry.py`: ids are stable, the pinned mapping is unchanged, lookups round-trip, an
  unknown name fails loudly.
- `test_bundle.py`: export then import round-trips on a synthetic tree; a relative `frames_dir`
  resolves; a mismatched schema version is refused; only reviewer-owned files move on import.
- `test_meta_schema.py`: `meta.json` validates, and is readable without importing `pipeline`.
- `test_launcher_config.py`: the launcher assembles a correct `ReviewContext` from a profile without
  constructing a window.
- `test_gui_modes.py`: the panel builders return the expected control sets per mode, `review` omits
  all 10 model controls, `full` matches the current surface exactly, and no model keybinding is
  registered in `review`. Written against the builder functions rather than a live viewer, so it
  runs without napari, in the same spirit as the existing `test_gui_box.py`.

The `full`-mode control set is worth pinning rather than eyeballing: it is what guarantees this
change is invisible to existing use.

The existing `test_import_direction.py` boundary still applies: `sam2_utils/registry.py` and
`sam2_utils/bundle.py` are library code and must not import `eval`.

## Risks and open points

- The first registry build is irreversible in practice. Once ids ship in an export they are fixed,
  so build it once, review the output, then commit.
- Bundle size is unmeasured. `F:` was unmounted while this was designed, so there is no real
  per-chain figure yet. Measure it on the first real export before promising a delivery mechanism.
  If it comes out too large for email or Drive, the fallback is fewer neurons per bundle rather than
  lower resolution.
- Exporting a legacy `_sam` chain reads raw EM to build its scale-2 canvas. Given `F:`'s documented
  flakiness, the exporter should be resumable and should reuse `load_frame_sam`'s existing retry
  rather than adding its own.
- None of this changes mask resolution. Masks still save at their native downscale, so a crisper
  canvas helps her eye and her drawing without making the saved mask finer.
