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
  _timing.csv                 # per-chain phase seconds and peak VRAM
  _labels.csv                 # one row per labelled frame (the GUI's training-data exhaust)
  <neuron>/
    chain_00/
      state.json              # the ChainState above
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

## Resume behavior

Resume is automatic. A chain already `done` or `flagged` is skipped on re-run. An interrupted chain
is left `running` and retried on the next launch. `clean` wipes prior outputs first and is
scope-aware: a full reset when no neurons are named, otherwise only the named neurons.

## A gotcha worth knowing

`_manifest.csv` is append-mode: rows are written per chain as they run, not rewritten. If a QC or gate
threshold changes between runs, early and late chains silently mix two configs. Clear or re-score the
manifest after any threshold change. A quick sanity check is to confirm the minimum area fraction
among area-failures is above the maximum among passes.
