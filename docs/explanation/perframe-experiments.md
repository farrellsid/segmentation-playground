# Per-frame segmentation experiments

A running log of `run_perframe.py` runs, including sweeps over the Approach-1 knob grid. Each run
writes its full output under `results/perframe/<run>/`: `config.json` (the exact knobs, git commit,
and command line), `scores.csv` (one row per frame), and `montages/` (one EM / labelled-instance /
membrane-overlay figure per frame). That directory is gitignored, since it is regenerable from the
config; this table is the committed record of what each run tried and how it scored. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md

Each table row summarises one run as the mean of its per-frame scores (`own_coverage`,
`total_foreign`, `mean_boundary_on_membrane`, `overlap_fraction`; see `eval/perframe_score.py` for
what each one measures). `own_coverage`, `total_foreign`, `mean_boundary_on_membrane`,
`spanning_rate`, and `mean_underfill` are scored on the RESOLVED (post-`--resolver`) masks, so
`--resolver` moves them; `overlap_fraction` is deliberately the exception, a PRE-resolution
diagnostic (the raw masks' pairwise pixel fight before `--resolver` arbitrates it), since scoring
it on the already-disjoint resolved masks would read near-zero regardless of how contested the raw
step was. `notes` is for anything the numbers do not capture, filled in by hand.

| run | approach | negatives | selection | resolver | frames | own_coverage | total_foreign | mean_boundary_on_membrane | overlap_fraction | notes |
|-----|----------|-----------|-----------|----------|--------|---------------|----------------|----------------------------|-------------------|-------|
| results/perframe/sweep_smoke/neg_on-sel_pred_iou-res_argmax | prompt | on | pred_iou | argmax | 1400 | 0.994 | 10997.00 | 0.716 | 13.2419 | pre-fix, superseded: scored the raw pre-resolution union masks (a `--resolver` change would not have moved these numbers), see CHANGELOG 2026-07-21 |
| results/perframe/smoke_fixed | prompt | on | metric | argmax | 1400 | 0.994 | 10981.00 | 0.721 | 13.2427 | comparison-protocol worked example below; beats the pred_iou row above on total_foreign and mean_boundary_on_membrane at equal coverage, so these were the "best knobs" used in the protocol. pre-fix, superseded (same caveat as the row above), see CHANGELOG 2026-07-21 |
| results/perframe/tune_dryrun | amg | - | metric | argmax | 1400 | 0.974 | 0.00 | 0.677 | 0.0000 | single tune-grid point (points_per_side 16, below the 32/64 default grid, chosen for local speed); only 38 of the frame's ~160 nodes matched an AMG mask. pre-fix, superseded: predates both the AMG fairness fix (unmatched cells now count as uncovered) and the unified scoring contract, see "the metric is incomplete" below and CHANGELOG 2026-07-21 |

## Comparison protocol

The design's three-way comparison: Approach 1 at its best sweep knobs, Approach 2 at its default
AMG parameters, and Approach 2 after `--tune`, all run on the same 5-to-10-frame sample, then read
off each run's `own_coverage`, `total_foreign`, `mean_boundary_on_membrane`, and `overlap_fraction`
(from `scores.csv` or a table row above) and checked against the montages. The three commands:

```bash
# Approach 1, best knobs from the sweep. Currently negatives on, selection metric, resolver
# argmax, the CLI defaults, per the two rows above (metric selection beats pred_iou selection
# at equal coverage; the full 12-combo sweep, run with --sweep, settles this properly).
py -3 run_perframe.py --approach prompt --frames <5-10 z's> --negatives on --selection metric \
    --resolver argmax --scale 8 --model-size tiny --out results/perframe/compare_prompt_best

# Approach 2, default AMG params.
py -3 run_perframe.py --approach amg --frames <same z's> --match metric --resolver argmax \
    --scale 8 --model-size tiny --out results/perframe/compare_amg_default

# Approach 2, tuned (12-combo default grid over the same frames, maximising the composite
# objective; --tune-grid narrows the grid if the default is too slow, see "compute cost" below).
py -3 run_perframe.py --approach amg --tune --frames <same z's> --match metric \
    --resolver argmax --scale 8 --model-size tiny --out results/perframe/compare_amg_tuned
```

**Status.** The full 5-to-10-frame, three-way run above has not completed on this box; see "compute
cost" below for why. What follows is a worked example on the single frame (z1400, about 160 cells,
a busy section) that already had real local runs, showing exactly how to read the three-way
comparison once the full sample is in. The real multi-frame run is a CCDB step, using the same
`run_perframe.py` commands on a GPU node.

**Reading the worked example.** Approach 1 at its best local knobs (the `smoke_fixed` row) covers
99.4% of nodes, at a cost of 10,981 foreign-node hits and a pairwise overlap 13.2 times the frame's
own area. Both are signs that the generous, negatives-on prompt path leans hard on the F3 resolver to
sort the fight for pixels out afterward. The single AMG tuning trial we do have (`tune_dryrun`, a
narrowed grid point, not the full search) shows the opposite profile: zero foreign hits and zero
overlap. That is mostly because only 38 of the frame's roughly 160 nodes matched an AMG mask at all,
so the other cells never entered the score. This is exactly why a coverage or bleed number can't be
trusted alone: a small, clean-looking AMG match can be small only because most of the frame went
unmatched, while a generous prompt run can look bleed-heavy only because F3 is doing real, mostly
successful cleanup work after the fact. The montages settle it. Open
`results/perframe/smoke_fixed/montages/1400.png` and `results/perframe/tune_dryrun/montages/` and
look at how much of the frame is actually labelled, not just how the labelled part scores.

These specific numbers are pre-fix (see the table notes above): re-run today, `smoke_fixed` would
score its resolved masks rather than the raw pre-resolution unions, and `tune_dryrun` would count
its 122 unmatched nodes as uncovered rather than dropping them from the mean. The qualitative point
of the worked example (numbers alone are not enough, check the montage) still holds either way, but
the exact figures above should not be quoted as current behaviour.

**The metric is incomplete, in one remaining way.** The design calls this out up front:
`score_frame` rewards a mask for containing its own node and staying off foreign nodes, not for
sitting on the true membrane. A mask that swallows its node and a chunk of surrounding tissue scores
identically to one that traces the cell precisely, as long as neither reaches a neighbour's node.
`mean_boundary_on_membrane` and `spanning_rate` push back on this somewhat, since they do read the raw
EM, but they are still comparative signals, not ground truth. So the montages, not the numbers, are
the deciding evidence for any claim about which approach segments better.

A second gap, found while building the first version of this comparison, has since been fixed: a
node with no matching AMG mask used to be silently absent from `cell_masks`, so it never entered
`own_coverage`'s mean instead of counting as a dropout. Both approaches now guarantee a key for
every node-bearing cell (an empty mask when nothing was matched or nothing survived overlap
resolution), and both score `own_coverage`/`total_foreign`/`mean_boundary_on_membrane`/
`spanning_rate`/`mean_underfill` on their RESOLVED masks, so the two approaches' numbers mean the
same thing and a `--resolver` sweep actually moves them. `overlap_fraction` stays a pre-resolution
diagnostic on purpose (see the note above the table); read it as "how much the raw step fought over
pixels before `--resolver` sorted it out," not as a property of the final segmentation. Still read
`n_cells` next to `own_coverage` for an AMG run, and check the montage for how much of the frame is
unlabelled.

**The `--match area` caveat.** Approach 2's `--match area` picks the smallest AMG mask containing a
node, with no cross-node exclusion. If one under-segmented AMG blob is the smallest mask containing
two different cells' nodes, both cells match that same blob, and F3 only splits it by nearest seed
afterward, a merged blob wearing two names. `own_coverage` reads as if both cells segmented cleanly,
when really one blob covered both. Prefer `--match metric` (the F2 composite match) whenever the
match itself needs to be trusted, and treat any `area`-matched `own_coverage` near 1.0 on a crowded
frame with suspicion until the montage confirms it.

**Compute cost, confirmed locally.** Approach 2's default AMG parameters (`points_per_side=64`,
`crop_n_layers=1`) did not finish a single frame within a two-minute wall-clock budget on this box,
even on one of the lighter target-worm sections (about 100 nodes, well under the 160-node z1400
frame used above); a narrower manual grid point (`points_per_side=16`) did complete one AMG call in
under two minutes, which is why `tune_dryrun` used it instead of the real default grid. A `--tune`
run multiplies this by the grid size (12 trials by default, plus one more for the winning params'
montage pass), so the full three-way, multi-frame comparison this section describes is realistically
a CCDB job, not a local one. This matches the design's own risk note (per-frame AMG over many frames
is compute-heavy) and is the reason the full comparison stays a documented protocol here rather than
a finished result.

## Results (2026-07-21): local A1-vs-A2 and A1-measures, scale 8, tiny model

First real per-frame runs. Figures under `docs/figures/perframe/`:
`approach1-vs-approach2-comparison.png` (5 frames) and `approach1-measures-grid.png` (z=1326), with
their scores CSVs. All are scale 8, tiny model, no per-cell crop.

A1 vs A2 (5 frames, 94 to 188 cells). Approach 1 (prompt, negatives on, metric, argmax) gives
near-complete coverage (own_coverage 0.98 to 1.00) but bleeds (total_foreign 10 to 51, pre-resolution
overlap 5 to 17). Approach 2 (auto-mask, light params points_per_side=16, no crops, match metric)
finds only 8 to 17% of cells (own_coverage 0.08 to 0.17); the masks it does find are clean, but it
misses most cells. A2's low coverage is partly the light local params: a full AMG (64 points plus
crops) on CCDB, plus the tuner, is needed for a fair A2 verdict.

A1 measures (z=1326). Negatives on halves the pre-resolution overlap (1.1 to 0.4) at almost no
coverage cost. Metric selection edges pred_iou on boundary-on-membrane, and both beat generous
(blockier, spillier, boundary 0.59 to 0.61). Resolver is a tradeoff: watershed snaps boundaries onto
membranes (boundary-on-membrane 0.87 to 0.90) but over-floods across the coarse scale-8 membrane map,
raising foreign bleed four to five fold (9 to 38-45), while argmax keeps foreign low with geometric
boundaries. Solid A1 default: negatives-on, metric selection, argmax. Open levers: per-cell cropping
for A1 (the per-chain resolution win, not yet in the runner), a sharper trained membrane map to make
watershed safe, and grow-to-membrane fill (Phase-2 item 2c) for underfill and A2 coverage rescue. The
metric is incomplete (rewards node-containment, not true boundaries), so read these with the montages.
