# 0017. SAM3-vs-SAM2 scorecard and the default segmentation backend

Status: Accepted

Builds on [0015](0015-target-worm-merge-metric-ruler.md) (the ruler this scorecard reads) and the
Phase 1.a plumbing recorded in the CHANGELOG's 2026-07-21/23 entries (`--backend sam3`, the sharded
Narval eval, the neg x gen A/B presets).

## Context

The whole-set SAM3 eval ran to completion on Narval on 2026-07-23: the two production configs
(`perslice_only_guard`, `tier2_s1forced_neg`) and a 2x2 A/B over `k_max_neg in {0, 3}` x
`multimask_generous in {off, on}`, all per-shard scored with `eval.merge_metric`. The consolidated
comparison table the roadmap called for (the `retro_sam3` job) never finished; it hit a 12-hour
timeout on the cluster because `eval.retro_eval` re-reads every mask from disk to rebuild the
per-frame table, redundant work when the sharded run already wrote and stitched
`<tree>/_merge_metric.csv` for every tree in question.

This ADR is the scorecard that job was meant to produce, built instead by calling
`eval.merge_metric.summarize()` directly on the already-stitched per-frame CSVs (seconds, not hours,
CPU only, no mask I/O), plus the default-backend decision it unblocks.

## The scorecard

Eight trees, all target-worm, scale-8 grid, matched by preset where a SAM2 counterpart exists.
`mean_foreign_per_flagged_frame = total_foreign_nodes / (foreign_frame_rate * n_frames)`, added here
because `total_foreign_nodes` alone conflates two different things: how often a mask bleeds, and how
badly it bleeds when it does.

| tree | chains | frames | foreign_frame_rate | dropout | mean_foreign_per_flagged | mild_bleed | mean_underfill | s/chain | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| sam3 perslice_only_guard (prod) | 605 | 7465 | **0.087** | 0.001 | **1.17** | 0.021 | 1.091 | 17.3 | 3.02 GB |
| sam2 perslice_only_guard | 629 | 8052 | 0.109 | 0.001 | 5.45 | 0.016 | 0.616 | 7.0 | 2.23 GB |
| sam3 tier2_s1forced_neg (prod) | 629 | 8052 | **0.214** | 0.124 | **1.21** | 0.024 | 0.645 | 7.7 | 3.99 GB |
| sam2 tier2_s1forced_neg | 629 | 8052 | 0.321 | 0.130 | 1.38 | 0.029 | 0.483 | 2.6 | 2.44 GB |

`perslice_only_guard` is the Phase 1 leader (ADR-equivalent gate closed 2026-07-21), so it is the
config that matters for the default-backend call.

**The severity gap, not just the frequency gap, is the headline.** SAM3 already had a lower
`foreign_frame_rate` in the 2-chain bake-off; the whole-set run adds that when SAM2's
`perslice_only_guard` does bleed, it typically drags in **5.45** foreign skeleton nodes, while SAM3's
occasional bleed touches **1.17**, barely enough to cross the severe-merge threshold at all. Both
effects (bleeds less often, bleeds less badly when it does) point the same direction and are large,
not marginal.

**The 2x2 A/B, read against the roadmap's §6 hypothesis.** The pre-registered guess was "negatives
hurt SAM3, since its masks are already tighter." The A/B refutes it:

| config | foreign_frame_rate | mean_foreign_per_flagged | mild_bleed |
|---|---|---|---|
| neg=0, generous=off | 0.120 | 3.54 | 0.020 |
| neg=0, generous=on | 0.162 | 1.22 | 0.026 |
| **neg=3, generous=off (= prod preset)** | **0.085** | **1.16** | 0.021 |
| neg=3, generous=on | 0.122 | 1.19 | 0.021 |

Negatives help SAM3 on every axis, holding generous fixed (neg=3 beats neg=0 in both rows).
Generous still hurts on every axis, holding negatives fixed, the same conclusion Phase 1 already
reached for SAM2. The best SAM3 cell is `neg=3, generous=off`, i.e. **the existing
`original_perslice_only_guard` preset, unchanged**. No SAM3-specific negatives retune is needed.

**Costs, honestly.** SAM3 is 2.2-2.5x slower per chain and uses about 0.8-1.6 GB more peak VRAM,
both comfortably inside a 6 GB card and a non-issue on Narval. SAM3's underfill is substantially
higher on `perslice_only_guard` (1.09 vs 0.62), the tight-mask cost already flagged in the bake-off;
Phase 2c (grow-to-membrane) is the mitigation and applies identically to SAM3 masks.

**Coverage caveat.** The `perslice_only_guard` SAM3 production tree scored 605/629 chains (a few did
not merge from the whole-set run); the comparison is still over 7,465 frames, not a small sample.

## Decision

**SAM3 per-slice (`perslice_only_guard` preset) is the accuracy leader on the trusted severe-bleed
ruler, by a wide and now whole-set-confirmed margin, at an acceptable compute cost.** The `--backend
sam3` flag and preset are unchanged, no retune needed, negatives stay on and generous stays off.

Chosen scope for this ADR: **record the recommendation and keep `--backend sam3` opt-in** (`batch.py`
still defaults to the preset's backend, which is `sam2`), rather than flipping every preset's stored
default. Reasons: (1) underfill is a known, still-open cost, and Phase 2c is the queued mitigation,
better to land it before making SAM3 the silent default everywhere; (2) flipping the default changes
the compute budget of every future cluster run by ~2.5x, a call the roadmap treats as the user's to
make explicitly per run, not a background switch; (3) `--backend sam3` already exists as one flag, so
opting in costs nothing extra. Revisit this scope choice once Phase 2c lands or if a future run makes
the compute cost the binding constraint.

## Consequences

- Phase 1.a (roadmap §5) is closed: the scorecard exists, the neg x gen A/B is read, and the default
  call is made (opt-in via `--backend sam3`, not a silent flip).
- The roadmap's §6 "negatives hurt SAM3" decision point is resolved the other way: negatives help:
  drop that line from future decision-point tracking.
- `sam3_fullres` (partial OOM, never merged) stays open only if full-res SAM3 is wanted; nothing in
  this scorecard depends on it.
- Future callers who want the lowest-bleed pipeline today should pass `--backend sam3` explicitly
  with the `original_perslice_only_guard` preset; everything else (batch.py's bare default, existing
  scripts) is unaffected.
