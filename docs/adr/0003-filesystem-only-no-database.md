# 0003. Filesystem only, no database

Status: Accepted

## Context

The pipeline runs locally on one Windows and GPU box, operated by one or two lab members. It produces
per-chain masks, per-chain state, QC tables, and a few cross-chain indexes. A database or a server
would add setup, a daemon to keep running, and a dependency to learn, for a workload that is a batch
job on local files.

## Decision

Use the filesystem as the database. Masks are PNGs on disk. Per-chain state is `state.json`.
Cross-chain indexes are CSVs at the root of the output tree: `_manifest.csv` for execution status,
`_triage.csv` for queued frames, `_review.csv` for the GUI's review status, `_timing.csv`, and
`_labels.csv`. No server, no database, no web app.

## Consequences

- Setup is `pip install` plus pointing a few paths at the data. There is nothing to run in the
  background.
- The manifest CSV is the work queue and the resume log at once. Multi-GPU sharding later is mostly a
  matter of workers claiming rows.
- Two writers never share a column. The batch owns execution status in `_manifest.csv`; the GUI owns
  review status in `_review.csv`. This is the cheap form of partition ownership.
- The manifest is append-mode: rows are written per chain, not rewritten. A threshold change between
  runs can silently mix two configs, so clear or re-score after any threshold change. See
  [state-and-storage.md](../reference/state-and-storage.md).
- Concurrent writers (a batch and a GUI at once) would need a file lock. That is deferred; today the
  GUI refreshes the queue on demand and one reviewer works at a time.
