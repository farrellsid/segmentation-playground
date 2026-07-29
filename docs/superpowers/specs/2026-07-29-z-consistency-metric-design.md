# Phase 0.a: z-to-z consistency metric

Status: design, approved 2026-07-29 (autonomous session, user away; written directly to spec rather
than through interactive brainstorming, per the roadmap's own framing of this item as "cheap" and
already well-specified: "Add a z-to-z IoU / drift term before trusting any per-slice-vs-propagation
ranking"). Scope: the metric only, no lever, no mask change.

## Why this, why now

`eval/merge_metric.py` scores each frame of a chain independently: foreign-node containment,
dropout, and (with `--no-membrane` off) the membrane-aware bleed/underfill scalars. Nothing in that
ruler looks across z. This makes it structurally blind to the one property that separates the two
segmentation strategies the project is actively choosing between: per-slice re-seeds every frame
independently and can never accumulate drift, so it trivially "wins" any per-frame ruler by
construction, while propagation buys temporal consistency (a neuron's shape and position change
smoothly frame to frame) at the cost of occasional drift or dropout over a long chain. The current
ruler cannot see the thing propagation is good at, which matters concretely because the pipeline's
output feeds 3D meshes, and a per-slice-jagged reconstruction is a real, visible defect the current
metric has no way to report ([[perslice-jitter-strip]] rendered exactly this jaggedness, informally,
for the presentation deck). The working preference has now shifted toward propagation for this
reason (per the roadmap's Phase 1.5 note), which makes this gap more load-bearing, not less.

## Scope

In:

- A new per-transition record (not per-frame, per-frame-PAIR) computed from a chain's raw masks:
  IoU and centroid drift between consecutive z's own masks.
- Wired into `eval/merge_metric.py`'s existing per-chain scoring pass, alongside (not replacing) the
  existing per-frame records.
- New summary fields reported next to the existing ones (`mean_z2z_iou`, `mean_centroid_drift_px`,
  and a low-consistency rate), so a per-slice-vs-propagation comparison can read both the existing
  bleed ruler and this new consistency ruler side by side.

Out:

- Any change to which mask a run keeps, or any new lever. This is measurement only, matching every
  other Phase-0/2 addition to this file.
- Retro-scoring every existing run tree. Land the metric, verified on synthetic data plus one real
  chain; retro-scoring specific trees for a specific comparison is a follow-on use of this tool, not
  part of building it.
- Gap-spanning "smoothed" comparisons (e.g. comparing z to z+2 when z+1 is missing) beyond simply
  recording the gap size next to the score. Interpreting a gap correctly is a human's job when they
  read the number; the tool's job is to not hide that a gap happened.

## The metric

Two pure functions in `eval/merge_metric.py`, next to `score_chain` (same file, since both consume
`pipeline.chain_masks_in_sam`'s `{z: (mask, x0, y0)}` return shape and belong to the same "Phase 0
ruler" concern; not a new module, this is small and single-purpose):

```python
def z_transitions(masks: dict[int, tuple[np.ndarray, int, int]]) -> list[dict]
```

For each pair of masks adjacent in sorted z order (not necessarily adjacent integers, a chain can
have gaps): place both masks into a shared coordinate frame using their own `(x0, y0)` offsets. Two
masks can have different local shapes (a tier-2 crop's window can move or resize between frames), so
IoU cannot compare the two boolean arrays directly, it needs a shared canvas:
`x_min = min(x0_a, x0_b)`, `y_min = min(y0_a, y0_b)`, `x_max = max(x0_a + w_a, x0_b + w_b)`,
`y_max = max(y0_a + h_a, y0_b + h_b)`; allocate a `(y_max - y_min, x_max - x_min)` boolean canvas per
mask, paste each mask in at its own offset minus `(x_min, y_min)`, then `intersection = (canvas_a &
canvas_b).sum()`, `union = (canvas_a | canvas_b).sum()`, `iou = intersection / union` (0 when both
masks are empty, though that case is already routed to the dropout path below and never reaches this
division). Centroid is simpler and does not need the shared canvas: each mask's own centroid in its
local coordinates plus its own `(x0, y0)` offset gives its position in the shared frame directly, no
canvas needed for that half. Record, per transition:

- `z_from`, `z_to`, `gap` (`z_to - z_from`, 1 for a true adjacent pair, more if a slice was skipped).
- `iou`: intersection-over-union of the two masks in the shared frame. `None` if either mask is empty
  (dropout is already the existing `empty`/`dropout_rate` signal's job; a transition touching an
  empty mask is not a consistency reading, it is a dropout reading, do not conflate the two).
- `centroid_drift_px`: Euclidean distance between the two masks' centroids, in the shared frame's
  pixel units (the run's `_sam` grid, same units the rest of `merge_metric.py` already uses).

```python
def summarize_z_consistency(transitions: list[dict]) -> dict
```

Aggregates one chain's (or, called again, one run's concatenated) transition records: `n_transitions`,
`mean_z2z_iou`, `mean_centroid_drift_px`, `frac_gap1_transitions` (what fraction of transitions were
true adjacent pairs, gap == 1, since a run riddled with gaps makes the other numbers less trustworthy
and a reader needs to know that up front), and `frac_low_iou` (fraction of gap==1 transitions with
`iou` below a threshold, default 0.5, an explicit low-consistency rate mirroring the existing
`mild_bleed_rate` headline pattern). `None`-valued (dropout) transitions are excluded from the IoU/
drift aggregates but counted separately as `n_dropout_transitions`, so a chain that scores well on
IoU only because most of its transitions were skipped due to dropout does not look falsely good.

## Integration

`score_run` (the existing whole-tree scorer) gains an additional pass alongside its existing per-chain
loop: for each chain, call `pipeline.chain_masks_in_sam(chain_dir)` again and pass the result to
`z_transitions`, collect transitions across all chains, and fold `summarize_z_consistency`'s output
into the same summary dict `summarize()` already returns, as new keys alongside `foreign_frame_rate`,
`mild_bleed_rate`, etc. This is a second call to `chain_masks_in_sam` per chain, `score_chain` already
loads the same masks internally and its signature is not touched by this spec (two existing tests,
`tests/test_merge_metric.py:63,167`, call `score_chain` directly with a chain-directory path, so
changing it to accept a pre-loaded masks dict instead would break them for a saving that does not
matter here: `chain_masks_in_sam` reads small per-frame mask PNGs, not the large EM frames the
membrane pass separately caches, so the extra read is cheap and not worth threading a shared dict
through an already-tested function's signature). `format_summary`'s printed line gains a third
`|`-separated segment when consistency data is present, matching the existing pattern where the
membrane segment only appears when that data exists.

The per-transition records themselves are NOT joined onto the existing per-frame `_merge_metric.csv`
(different grain: one row per frame vs one row per frame-pair, joining them would either duplicate
frame rows or force an awkward null-heavy schema). They get their own file,
`<root>/_z_consistency.csv`, written by `score_run` alongside the existing per-frame CSV, one row per
transition across every chain in the tree (with `neuron` and `chain_idx` columns added, matching the
existing per-frame CSV's convention, so the two files can be joined by a reader who wants to).

No new CLI flags needed for the base case (this pass is cheap, pure array math on already-loaded
masks, no new EM reads, so it runs unconditionally as part of the existing `merge_metric` scoring,
same way Phase-0 foreign-node scoring always runs). Add one flag, `--low-iou-threshold` (default
0.5), matching the existing pattern of exposing detector thresholds (`--tau`, `--tol`) as CLI
overrides rather than hardcoding them.

## Testing (CPU, torch-free)

`tests/test_z_consistency.py`, synthetic `{z: (mask, x0, y0)}` dicts, no frames, no torch:

- Two identical masks at consecutive z: `iou == 1.0`, `centroid_drift_px == 0.0`, `gap == 1`.
- Two masks shifted by a known pixel offset: `centroid_drift_px` matches the offset's magnitude
  (within floating-point tolerance); `iou` is less than 1 and matches a hand-computed value for a
  simple shifted-rectangle case.
- The same physical region expressed through two different local window shapes and offsets (the
  case the shared-canvas paste logic exists for, per the docstring's tier-2-crop-window rationale
  above): `iou == 1.0`, `centroid_drift_px == 0.0`. Also cover the offset-only case, identical local
  masks at two different offsets, where `centroid_drift_px` is exactly the offset's own magnitude.
- Two completely disjoint masks: `iou == 0.0`.
- A transition where one mask is empty: `iou is None`, `centroid_drift_px is None`, and it is
  excluded from `summarize_z_consistency`'s IoU/drift means but counted in `n_dropout_transitions`.
- A chain with a z-gap (e.g. masks at z=10, 11, 13, skipping 12): the transition 11->13 has `gap == 2`,
  and `frac_gap1_transitions` in the summary correctly reports 1 of 2 transitions as gap==1 (using a
  3-mask chain: 10->11 gap 1, 11->13 gap 2).
- `summarize_z_consistency` on an empty transitions list: returns a summary with `n_transitions == 0`
  and no exception (matching `summarize`'s existing empty-input handling elsewhere in the file).
- `frac_low_iou` at the default 0.5 threshold: a mix of high- and low-IoU gap==1 transitions produces
  the correct fraction, and a dropout transition (`iou is None`) is excluded from the denominator.

## Docs to update on landing

- `docs/reference/cli.md`: this is where the merge-metric's summary fields are already documented
  (confirmed: `foreign_frame_rate`, `mild_bleed_rate`, etc. are described around line 153-156, not in
  `qc-signals.md`, which covers a different, older QC signal). Add the new
  `--low-iou-threshold` flag and the new summary fields (`mean_z2z_iou`, `mean_centroid_drift_px`,
  `frac_low_iou`, `frac_gap1_transitions`, `n_dropout_transitions`) there, next to the existing ones.
- `docs/explanation/roadmap.md`: mark item 12 / Phase 0.a landed, and note it is now available to
  re-score any existing run tree for a per-slice-vs-propagation comparison (a follow-on use, not
  built here).
- `docs/CHANGELOG.md` on landing.
- No ADR: this is an additive measurement tool, not a load-bearing tradeoff a newcomer would question,
  matching the pattern of ADR 0015/0016 (those got ADRs because they picked a specific detection
  *design*, e.g. border-to-border vs any-membrane-inside; a z-to-z IoU is the obvious, undebatable
  choice for "how consistent are two adjacent masks," nothing here needs defending later).

## Risks and accepted limitations

- IoU and centroid drift are blind to *why* two adjacent masks differ: real anatomical change (a
  neurite branching, tapering, or turning a corner) looks identical to drift or a tracking error in
  this metric. This is a known, accepted limitation of any purely geometric consistency signal; it is
  a comparator across runs/methods on the same chains, not an absolute defect detector, the same
  caveat every other `membrane.py`/`merge_metric.py` detector already carries.
- A chain with many z-gaps (heavy `blowup_guard` skipping, or a config with sparse coverage) will have
  a low `frac_gap1_transitions`, making the IoU/drift means less trustworthy for that chain. The
  summary reports this fraction explicitly so a reader is not misled by a mean computed over few,
  possibly-unrepresentative pairs, but the tool does not itself refuse to report a number in that case.
- This does not, by itself, resolve the per-slice-vs-propagation debate. It adds the missing axis; a
  full comparison still needs both this metric and the existing bleed ruler read together, per the
  roadmap's framing ("before trusting any ranking," not "instead of the existing ranking").
