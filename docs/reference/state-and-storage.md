# State and storage

The filesystem is the database. A run writes two separate trees: `output/` for results and
`frames_root/` for the SAM2 JPEG frames. State serializes per chain, so a run resumes after a crash
without recomputing finished chains.

## ChainState

`ChainState` (in `pipeline.py`) is the serializable per-chain record. It carries:

- `neuron`, `chain_idx`, `status` (one of `pending`, `running`, `done`, `flagged`, `failed`)
- `anchor_frame_idx`
- `prompts`: the points, labels, and box used to seed
- `image_mask_ref`, `qc_summary`
- `triage_frames`: the frames that need human review
- `crop_window`: set for tier-2 chains, so the `_pcrop` space can be rebuilt
- per-phase timing

It serializes to `state.json`. A chain can be paused, resumed after a crash, or reopened in the GUI
from this file alone.

## The output tree

```
output/
  _manifest.csv               # every chain and its execution status; drives the batch and resume
  _triage.csv                 # queued (intervene) frames across all chains; feeds the GUI
  _review.csv                 # the GUI-owned review-status ledger, separate from the manifest
  _timing.csv                 # per-chain phase seconds + peak VRAM; for a tier-2 chain t_total covers both passes and t_pass1 is the first (_sam) pass
  _labels.csv                 # one row per labelled frame (the GUI's training-data exhaust)
  <neuron>/
    chain_00/
      state.json              # the ChainState above
      meta.json               # portable identity + geometry, readable without importing pipeline
      qc.csv                  # per-frame QC metrics, plus a `queue` column
      masks/mask_<z:04d>.png  # 0/255 uint8, canonical _sam space
    chain_01/ ...
```

The batch owns the execution status in `_manifest.csv`. The GUI owns the review status in
`_review.csv`. They never write the same column. This is the cheap form of partition ownership: two
writers, two files.

## The frames tree

```
frames_root/
  frames_cache_s<scale>/
    z<file_z>.jpg             # shared decode cache: each EM frame downscaled once, ever
  chain_views/
    <neuron>_chain<idx>_s<scale>/
      00000.jpg ...           # 0-indexed links into the cache, per chain
```

The decode cache means overlapping chains pay the large imread-and-resize once across the dataset,
not once per chain. The per-chain views are links (hard-link on Windows, since the cache and views
share a volume).

## The review bundle

A bundle is a portable copy of part of an output tree, for a reviewer on another machine:

```
<bundle>/
  bundle.json                    schema version, chain index, source tree
  neurons.csv                    the registry rows this bundle uses
  data/chains.json               the exported neurons' chain records, complete and in source order
  data/nodes.csv                 the exported neurons' CATMAID node rows, raw columns
  <neuron>/chain_00/
    meta.json
    state.json                   frames_dir is RELATIVE here, unlike the master tree
    qc.csv
    masks/mask_<z:04d>.png
    frames/00000.jpg ...
```

`export_bundle.py` writes one, `import_bundle.py` merges a returned one back. Per chain, only
`masks/` and `qc.csv` travel on the return trip (`bundle.REVIEWER_OWNED`): a bundle's `state.json`
records a relative `frames_dir` and would break the master tree if copied over it.

Two tree-root ledgers come back as well (`bundle.REVIEWER_LEDGERS`), merged ROW-WISE rather than
copied. Inside a bundle `output_root` IS the bundle, so the GUI writes a reviewer's chain
dispositions to `<bundle>/_review.csv` and her per-frame verdicts to `<bundle>/_labels.csv`, and
four of the ten keys review mode exposes write to nothing else. `_review.csv` holds one row per
`(neuron, chain_idx)`, so a bundle chain's row replaces the master's and every other row is left
alone. `_labels.csv` de-duplicates on `(neuron, chain_idx, z)`, the key `LabelStore` itself upserts
on, and a collision on that key is resolved by the later `ts`. Copying either file wholesale would
delete the master's rows for chains that were never in the bundle.

The relative `frames_dir` is the point of the format. A master tree bakes an absolute path, which
does not resolve elsewhere, and `gui.py` then falls back to regenerating frames from the raw EM
store. `gui.resolve_frames_dir` joins a relative value against the chain directory instead. That
function, `bundle.validate_bundle` and `export_bundle`'s copy-source check all go through the one
implementation in `sam2_utils/bundle.py`, because three private copies of the rule is how they
drifted apart before.

`data/` is what makes a bundle openable at all. `data/chains.json` and `data/aggregate_data_pv.csv`
are gitignored, so a reviewer who clones the repo has neither, and `ReviewContext` needs both to
resolve a chain and to build `annotate_df`. The bundle therefore carries its own slice, filtered by
NEURON: every exported neuron keeps all of its chains, in the source file's order, because
`chain_idx` is a position in that list and dropping one chain would shift every later index.
`nodes.csv` ships the raw columns, with no precomputed `x_tif`/`y_tif`, so the coordinate affine
stays in code and a later correction to it still reaches an already-shipped bundle.
`ReviewContext` prefers these files when they exist and falls back to the `config` paths when they
do not, so a normal master-tree session is unchanged.

## Resume behavior

Resume is automatic. A chain already `done` or `flagged` is skipped on re-run. An interrupted chain
is left `running` and retried on the next launch. `clean` wipes prior outputs first and is
scope-aware: a full reset when no neurons are named, otherwise only the named neurons.

## A gotcha worth knowing

`_manifest.csv` is append-mode: rows are written per chain as they run, not rewritten. If a QC or gate
threshold changes between runs, early and late chains silently mix two configs. Clear or re-score the
manifest after any threshold change. A quick sanity check is to confirm the minimum area fraction
among area-failures is above the maximum among passes.
