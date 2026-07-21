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
