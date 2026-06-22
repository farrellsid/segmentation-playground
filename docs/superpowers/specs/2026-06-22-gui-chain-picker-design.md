# GUI chain picker: two selectors and a flag-only / everything mode

Date: 2026-06-22
Status: approved, ready to implement

## Problem

The review GUI opens chains through one `ComboBox` (`gui.ReviewGUI._picker`) fed by
`ReviewQueue.pending()`, which lists only chains the batch flagged and the reviewer
has not yet dispositioned. During manual proofreading the reviewer needs to open any
chain on disk, flagged or not, and a single flat `neuron/chain_NN` list does not scale
when a neuron has many chains. The reviewer wants to pick a neuron first, then a chain.

## Goals

- Split the one picker into two cascading selectors: pick neuron, then pick chain.
- Make every chain that exists on disk pickable, not just the flagged ones.
- Add a mode toggle, flag-only versus everything, that is the single source of truth
  for the dropdowns and for the next/prev CHAIN navigation.

## Non-goals

- No change to how a chain is loaded, corrected, or saved once open.
- No change to the batch, the manifest, or the triage detail.
- No live polling: the manual refresh button stays.

## Decisions

These were settled during brainstorming:

- The everything list comes from on-disk chain directories, the exact set
  `open_chain` can load, so the picker never offers a chain that fails to open.
- The GUI launches in flag-only mode, preserving today's behavior.
- The mode governs both the dropdowns and the next/prev CHAIN cycling (and so the
  chain auto-opened on launch, which in flag-only mode is unchanged).
- The chain selector shows a status badge next to each index.

## Design

### Data layer: extend `ReviewQueue`

Three additions in `sam2_utils/review_queue.py`, siblings to `flagged_chains()`,
keeping the module torch-free and napari-free:

- `all_chains() -> List[tuple]`: scan `output_root` for `<neuron>/chain_NN`
  directories and return `(neuron, chain_idx)` sorted by `(neuron, chain_idx)`.
  This is the openable universe, a superset of `flagged_chains()`: a chain does not
  have to be flagged, or even present in the manifest, to be opened, only saved to
  disk. Top-level files such as `_manifest.csv` are skipped (not directories), and a
  missing `output_root` returns an empty list.
- `manifest_status(neuron, chain_idx) -> Optional[str]`: the batch execution status
  (`flagged`, `done`, `failed`) for a chain, or `None` when it has no manifest row.
- `chain_status(neuron, chain_idx) -> str`: the badge string. Precedence: the human's
  review disposition if the chain has one (`in_review`, `approved`, `rejected`,
  `corrected`), else the manifest status, else `unreviewed`. The review disposition
  wins because it is the later, human word on the chain.

### GUI layer: three widgets replace the one picker

In `gui.ReviewGUI._build_widgets`, the single `_picker` plus its open button become:

- `_mode_combo`: choices `["flagged only", "everything"]`, default `"flagged only"`,
  driving an internal `_show_all` flag.
- `_neuron_combo`: the neurons present in the current mode's chain set.
- `_chain_combo`: the chains for the selected neuron, built as `(label, idx)` tuples
  so `.value` is the integer chain index directly, with no string parsing. Labels
  read `chain_03 [flagged]`.

The `open selected chain` button reads `_neuron_combo.value` and `_chain_combo.value`.

Cascade and guards:

- Changing the mode repopulates both selectors.
- Changing the neuron repopulates the chain selector.
- A `_populating` guard prevents redundant re-entry while choices are being set
  programmatically.

### Mode as the single source of truth

One helper, `_mode_chains()`, returns `pending(include_in_review=True)` in flag-only
mode or `queue.all_chains()` in everything mode. It feeds both the selector population
and `_step_chain` (the next/prev CHAIN cycling), so the two never disagree. After a
chain opens, the selectors sync to it and re-read its badge, which has just become
`in_review` through the existing `queue.claim` call in `open_chain`.

`launch()` is unchanged: the default mode is flag-only, so it still auto-opens the
first pending chain, or an empty viewer when the queue is empty.

## Testing

Add cases to `tests/test_review_queue.py`, CPU-only and torch-free:

- `all_chains` enumerates `<neuron>/chain_NN` directories, sorted, ignoring
  top-level CSV files, and returns an empty list for a missing root.
- `chain_status` precedence: a dispositioned chain reports its review status; an
  undispositioned flagged chain reports `flagged`; a chain with neither reports
  `unreviewed`.

GUI wiring stays untested, matching the rest of `gui.py` (napari is not in the test
environment).

## Docs to update in the same change

- The GUI reference or `docs/reference/cli.md`: the new mode and the two selectors.
- `docs/CHANGELOG.md`: the picker change.
