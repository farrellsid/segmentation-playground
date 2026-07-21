# Per-frame segmentation experiments

A running log of `run_perframe.py` runs, including sweeps over the Approach-1 knob grid. Each run
writes its full output under `results/perframe/<run>/`: `config.json` (the exact knobs, git commit,
and command line), `scores.csv` (one row per frame), and `montages/` (one EM / labelled-instance /
membrane-overlay figure per frame). That directory is gitignored, since it is regenerable from the
config; this table is the committed record of what each run tried and how it scored. Design:
docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md

Each table row summarises one run as the mean of its per-frame scores (`own_coverage`,
`total_foreign`, `mean_boundary_on_membrane`, `overlap_fraction`; see `eval/perframe_score.py` for
what each one measures). `notes` is for anything the numbers do not capture, filled in by hand.

| run | approach | negatives | selection | resolver | frames | own_coverage | total_foreign | mean_boundary_on_membrane | overlap_fraction | notes |
|-----|----------|-----------|-----------|----------|--------|---------------|----------------|----------------------------|-------------------|-------|
| results/perframe/sweep_smoke/neg_on-sel_pred_iou-res_argmax | prompt | on | pred_iou | argmax | 1400 | 0.994 | 10997.00 | 0.716 | 13.2419 | |
| results/perframe/smoke_fixed | prompt | on | metric | argmax | 1400 | 0.994 | 10981.00 | 0.721 | 13.2427 | comparison-protocol worked example below; beats the pred_iou row above on total_foreign and mean_boundary_on_membrane at equal coverage, so these are the "best knobs" used in the protocol |
| results/perframe/tune_dryrun | amg | - | metric | argmax | 1400 | 0.974 | 0.00 | 0.677 | 0.0000 | single tune-grid point (points_per_side 16, below the 32/64 default grid, chosen for local speed); only 38 of the frame's ~160 nodes matched an AMG mask, so these numbers cover that matched subset only, not full-frame recall, see "the metric is incomplete" below |

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
py -3 run_perframe.py --tune --frames <same z's> --match metric --resolver argmax \
    --scale 8 --model-size tiny --out results/perframe/compare_amg_tuned
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

**The metric is incomplete, in two ways.** First, the one the design calls out up front:
`score_frame` rewards a mask for containing its own node and staying off foreign nodes, not for
sitting on the true membrane. A mask that swallows its node and a chunk of surrounding tissue scores
identically to one that traces the cell precisely, as long as neither reaches a neighbour's node.
`mean_boundary_on_membrane` and `spanning_rate` push back on this somewhat, since they do read the raw
EM, but they are still comparative signals, not ground truth. So the montages, not the numbers, are
the deciding evidence for any claim about which approach segments better. Second, and more subtly,
found while building this comparison: Approach 2's `own_coverage` is computed only over the cells that
matched an AMG mask (`n_cells` in `scores.csv` is the size of `cell_masks`, the matched set, not the
frame's full node count). A node with no matching AMG mask never enters the score at all. It is not
counted as a dropout; it is simply absent. So a tuning trial that matches few nodes but matches them
cleanly can look better than one that matches most of the frame with some mess. Always read `n_cells`
next to `own_coverage` for an AMG run, and check the montage for how much of the frame is unlabelled.

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
