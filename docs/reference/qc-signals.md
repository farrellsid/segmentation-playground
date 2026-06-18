# QC signals and flagging

QC runs after a chain's masks are saved. It scores every frame, writes `qc.csv`, and sets the chain
status to `done` or `flagged`. The metric code is in `sam2_utils/qc.py`; the thresholds are `qc_*`
knobs on `PipelineConfig` (tune them there, not in `qc.py`).

## Per-frame signals

| Signal | What it measures | Flags when |
|--------|------------------|-----------|
| `area_ratio` | The mask area this frame relative to the neighbouring frame | Outside `qc_area_ratio_bounds` |
| `temporal_iou` | Overlap of this frame's mask with the neighbouring frame | Below `qc_temporal_iou_min` |
| `skeleton_contained` | Whether this chain's skeleton node falls inside the mask | Explicit False only (tri-state) |
| `pred_iou` | SAM2's own mask-decoder IoU estimate for the frame | Below `qc_pred_iou_min` |

`skeleton_contained` is tri-state: True, False, or NaN when there is no node at that z. Only an
explicit False flags, so frames in a z-gap abstain rather than false-flag. It must use this chain's
nodes, not the whole neuron's. See [coordinate-spaces.md](coordinate-spaces.md).

## Flag versus queue

A frame is flagged when any one signal trips. A frame is queued for a human (intervene) when at least
`qc_triage_min_signals` fire at once (default 2). The chain status keys on the queue count, so the
per-frame queue and the chain verdict never disagree.

The single-signal flags stay in `qc.csv` as diagnostics and as label data, but only the queued
(multi-signal) frames reach a human through `_triage.csv`. In practice the skeleton-containment
signal dominates raw flags, while the multi-signal queue is the smaller, corroborated set.

## Thresholds

The knobs are `qc_area_ratio_bounds`, `qc_temporal_iou_min`, `qc_pred_iou_min`,
`qc_skeleton_dilation_px`, and `qc_triage_min_signals`. See
[configuration.md](configuration.md) for the full list.

Two thresholds are deliberately left at borrowed defaults until ground-truth labels exist to
calibrate them: the `pred_iou` floor and the anchor-gate area ceiling. If you change any threshold
mid-campaign, clear or re-score the manifest (it is append-mode).

## A known limitation

The flag rule is the weakest part of the system. It scores the saved stack after the fact and is a
noisy proxy for real error. Replacing it with a learned error detector trained on the GUI's collected
labels is the planned next step. See [../explanation/roadmap.md](../explanation/roadmap.md).
