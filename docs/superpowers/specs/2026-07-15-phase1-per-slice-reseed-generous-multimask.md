# Phase 1: per-slice re-seeding + generous-capped multimask

Design for the two Phase 1 "cheap structural fixes" from the redesigned roadmap (§5 Phase 1). Both
land behind config flags so default pipeline behavior is byte-identical when off, and both are
measured against the Phase 0 merge-metric (`eval.merge_metric`).

## Problems addressed

- **Identity-switch / drift bleed (item 1).** SAM2 video propagation carries one anchor's mask across
  ~300 slices via memory; far from the anchor it drifts onto a neighbouring cell and stays there
  (measured: AVAL/ch16 shows 111 foreign-node-hit frames in fullres). The merge-metric confirms severe
  bleed is the dominant remaining error (foreign_frame_rate ~0.32 even in the best run).
- **Nested-membrane / soma nucleus-only mask (item 3).** A point prompt inside a nucleus segments the
  nucleus, not the whole soma. We want the mask to INCLUDE the nucleus and reach the outer membrane,
  not be limited to the nucleus and not blow up to the whole frame.

## Item 1: per-slice re-seeding

**Behavior.** A new orchestration path gated by `PipelineConfig.per_slice_reseed` (default False).
When on, a chain is segmented slice-by-slice with NO video propagation: each slice is treated as its
own anchor, re-grounded on its own skeleton node, so memory can never carry the wrong cell across
slices.

**Per-slice loop.** For each slice z in the chain's z-range, reuse the existing anchor procedure
(`anchor_crop_predict` / `image_predict` / `box_from_mask`, `pipeline/predict.py`):
1. Get the centreline (x, y) at z. Skeleton nodes are ~one-per-slice (median 1.0 nodes/slice; 91% of
   multi-node chains are node-per-slice), so the common case reads the node directly. For gap slices
   (~9%), linearly interpolate the centreline between the two nearest same-chain nodes.
2. Centre a skeleton-following crop on that point at `crop_scale` (reuse `crop.chain_crop_window` /
   `CropWindow.around_node`). Because the crop is node-centred, there is NO coarse locate pass.
3. Build prompts (node positive + neighbour-node negatives via `build_prompts`), run image-mode in the
   crop, `box_from_mask`, re-predict with the box.
4. Select the mask with the generous-capped rule (item 3), save it for slice z.

**What it reuses vs changes.** It reuses the anchor procedure and the crop machinery unchanged, so it
keeps the proven crop resolution win (including `crop_scale=1`, the full-res second pass). The change
is orchestration only: `pipeline/orchestrator.run_chain` gets a branch that loops the anchor procedure
per slice instead of calling `propagate()` once. Masks are saved per slice exactly as today (same
filenames, same `_sam`/`_pcrop` space via `crop_window`), so downstream readers (`chain_masks_in_sam`,
QC, `eval.merge_metric`) are unaffected.

**Explicitly not done.** No video propagation for these chains. No cross-slice memory. This trades
temporal continuity (a slice with a genuinely ambiguous cross-section has no neighbour context) for
drift-immunity; the trade is measured by the merge-metric A/B, not assumed.

## Item 3: generous-capped multimask

**Behavior.** A new selection mode gated by `PipelineConfig.multimask_generous` (default False),
consulted inside `_select_anchor_mask` (`pipeline/predict.py`) when `multimask_anchor` is on. Among the
three SAM2 candidates that contain the positive node and are single-CC, prefer the one with the LARGER
area, but hard-reject any candidate whose area fraction exceeds the max-area cap. SAM2's largest
candidate is frequently a whole-frame blob, so "largest" alone is wrong; the cap is what makes
"generous" safe. Concretely, the ranking key changes from preferring smaller/higher-score toward
preferring larger area WITHIN `[gate_min_area_frac, gate_max_area_frac]`, with candidates above
`gate_max_area_frac` dropped.

**Resolution-aware leeway.** The area bounds and the containment radius carry leeway that scales with
effective resolution, extending the existing `space_ratio = scale / eff_crop_scale` rescale already
applied to `contain_radius` and `box_margin` in `orchestrator.run_chain`. At a finer crop a soma fills
more of a tight crop, so the cap needs headroom the coarse setting does not; the leeway is a function
of `crop_scale`, not a hard constant.

**Interaction with item 1.** Because every slice is an anchor under item 1, the generous-capped rule
applies per slice automatically. It is also usable independently on the current propagation path (it
only touches anchor selection).

## Config flags (all default OFF; existing behavior preserved)

- `per_slice_reseed: bool = False` (item 1).
- `multimask_generous: bool = False` (item 3; only consulted when `multimask_anchor` is on).
- Reuse existing knobs: `crop_scale` / `chain_crop_scale`, `gate_min_area_frac`, `gate_max_area_frac`,
  `k_max_neg`, `seed_negatives`. Add a leeway knob only if the existing gate fractions prove
  insufficient in the smoke (decide from data, do not pre-add).

## Testing (hard requirements)

- **Local smoke always uses a downscaled image.** Never run full-res locally (RTX 3050, 6GB; full-res
  masks already OOM'd the GUI). Local smoke uses a coarse setting (small `model_size`, high `scale` /
  `crop_scale`, a few short chains) purely to confirm the code path runs and produces sane masks.
- **Full-res quality runs only on CCDB / Narval** via the existing preset + Slurm-array workflow.
- **Scoring is the merge-metric.** After each run, score with `eval.merge_metric` and A/B against the
  current best baseline (`tier2_s1forced_neg`): foreign_frame_rate for bleed, dropout_rate for omission.
- Unit tests are CPU-only and torch-free (per-slice loop wiring, centreline interpolation for gap
  slices, and the generous-capped selection key, all testable on synthetic masks / node tables).

## Success criteria

- Item 1: per-slice re-seeding lowers foreign_frame_rate vs the `tier2_s1forced_neg` baseline without
  raising dropout_rate, on the merge-metric over the EXP_NEURONS subset. Specifically it should cut the
  drift cases (e.g. AVAL/ch16 foreign-hit frames well below fullres's 111).
- Item 3: `multimask_generous` raises soma coverage (the soma mask reaches the outer membrane, includes
  the nucleus) without a net increase in neurite foreign_frame_rate.

## Out of scope

- Mutex-watershed / multicut non-overlap (old item 2) is deferred to depend on the Phase 2 membrane
  map; it needs a boundary signal to be worth building. Roadmap §5 updated accordingly.
- No negative point inside the nucleus: that would exclude the nucleus, the opposite of the goal.
- No skeleton-graph-aware interpolation beyond simple linear centreline fill for gap slices; branches
  and complex gaps are handled per chain (one chain is one arm) and revisited only if the smoke shows
  gap artifacts.

## Risks / open questions

- Per-slice compute is heavier (an image encode per slice). Each encode is a small crop, and CCDB
  absorbs the cost; confirm wall-clock on the smoke before a full run.
- Generous-capped selection could still nudge neurite bleed up; the merge-metric A/B is the gate, and
  the cap + resolution leeway are the tuning levers.
- Centreline interpolation quality on the ~9% gap slices is unverified; the smoke must include at least
  one chain with a gap.
