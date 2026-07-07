# GUI run-picker: browse and open runs by their metadata

Status: design, not yet implemented. Companion to the resolution-experiments spec
(2026-07-06); this is the "organize the outputs" follow-up.

## Problem

`output_masks/` now holds many runs at once: the default tier-2 target run, the four
resolution-experiment variants, older test trees, and (soon) more. Opening one to review
means passing its path by hand with `--output-root`, and remembering which folder was which
config. Two concrete pains:

1. No way to see, at a glance, what each run is (which preset, worm, scale, how many chains
   done vs flagged) without opening it or reading files by hand.
2. The GUI builds a fixed default config (`scale=8, save_downscale=8`) and assumes the target
   worm. That is wrong for the non-canonical runs: a `save_downscale=1` full-res run or a
   `_pcrop` tier-2 run still *displays* correctly (the overlay scale is derived from pixel
   dimensions), but any re-segmentation action re-runs at scale 8, and the EM source is not
   chosen from the run's dataset. Reviewing a downloaded cluster run also needs `--hires-em`
   passed by hand because its `frames_dir` is a dead node-local path.

The metadata to fix all of this already exists: `batch.write_run_meta` writes `_run_meta.json`
into every run (preset, dataset, `scale`/`save_downscale`/`image_size`, tier2 flags, model,
neuron list, date, git commit, and the actual built `image_size`). Nothing reads it back yet.

## Goal

A run-picker that scans a runs-root, reads each run's `_run_meta.json`, lists the runs with
their key fields, and opens the chosen one with the GUI configured from that run's own metadata
(so re-segmentation, EM source, and hires fallback are all correct without extra flags).

## What marks a run

A directory containing `_manifest.csv`. `_run_meta.json` next to it carries the rich metadata;
runs without it (older or hand-made trees) still open, they just show name plus manifest-derived
counts and fall back to canonical assumptions with a warning.

## Design

New pure module `sam2_utils/runs.py` (no napari, no torch, unit-testable):

- `RunInfo` dataclass: `path`, `preset`, `dataset`, `scale`, `save_downscale`, `image_size`,
  `tier2` (bool/summary), `model_size`, `n_neurons`, `written_at`, `git_commit`, plus
  `n_done`/`n_flagged`/`n_failed` read from `_manifest.csv`. Missing `_run_meta.json` leaves the
  meta fields `None` and marks `has_meta=False`.
- `scan_runs(runs_root, *, recursive=True) -> list[RunInfo]`: walk for dirs with `_manifest.csv`,
  build a `RunInfo` each, sort newest-first. Skips the shard-internal `chunk_*` dirs.
- `run_config(info) -> dict`: the `PipelineConfig` kwargs implied by the run's meta (scale,
  save_downscale, image_size, model_size), for the GUI to build a matching config.

`gui.py` changes:

- New `--runs-root DIR` flag. When passed (or when neither `--output-root` nor `--runs-root`
  is given and the cwd default is a runs-root), open a picker instead of a single tree.
- Picker: a startup dialog / dock listing `scan_runs` results in a small table (run name,
  dataset, scale / image_size, tier2, model, done/flagged/failed, date). Selecting a row opens
  that run as the `output_root`, reusing the existing open flow.
- On open, build the GUI's `PipelineConfig` from `run_config(info)` (not the hardcoded scale-8
  default) so re-segmentation honors the run's scale, and resolve the EM source from `dataset`
  (target worm vs a GT worm). Auto-enable hires-EM when the run's `frames_dir` does not exist
  locally (the downloaded-cluster case), removing the need to pass `--hires-em` by hand.
- Runs without `_run_meta.json`: keep today's behavior (canonical config), plus a one-line
  warning in the info panel that scale/dataset were assumed.

## Approaches considered

- **A (chosen):** `--runs-root` + a metadata picker + config-from-meta on open. Modest, and it
  also fixes the re-segmentation-scale and hires-EM papercuts in the same change.
- **B:** convention only, plus a `scripts/list_runs.py` that prints the table; the human still
  passes `--output-root`. Cheaper, no GUI change, but does not fix the config-from-meta issue.
- **C:** a full run manager (tags, filters, side-by-side diff of two runs). Overkill now; the
  four-way comparison is served well enough by opening runs one at a time.

## Components and isolation

- `sam2_utils/runs.py` is pure data (scan + parse), so it is tested without a display or GPU and
  is reusable by `eval` or a CLI later.
- The picker is the only napari-facing addition to `gui.py`; the open path is unchanged once an
  `output_root` is chosen.

## Testing

- `runs.py`: fixtures of fake run dirs (with and without `_run_meta.json`, with a `chunk_*` to
  confirm it is skipped) assert `scan_runs` finds them, parses fields, and derives counts; and
  `run_config` returns the right kwargs. CPU-only, torch-free.

## Non-goals (YAGNI)

Tagging, filtering, search, and side-by-side comparison. Editing runs from the picker. A
database or index file; the filesystem scan is the index.

## Open questions

- Picker widget: a plain list is enough for a handful of runs; a sortable table is nicer if the
  count grows. Start with a simple list showing the one-line summary.
- EM-source resolution for non-target datasets: the target worm reads `WORM_PATH`; a GT-worm run
  would need its own EM path. For now only the target worm is reviewed, so resolve target and
  leave a clear error for other datasets until that path is needed.
