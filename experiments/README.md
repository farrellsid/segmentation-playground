# experiments

**Status: kept for reproducibility.** One-off A/B harnesses and sweeps. Not imported by the library
or any driver. Captured `*.log` files and `ab_figs/` are disposable.

One-off measurement scripts from the M4.5 A/B round. **None of these are part of the durable
library or any driver**, nothing in `pipeline.py` / `batch.py` / `gui.py` / `run_aval.py` imports
them. They are kept so each finding can be reproduced, and as primary source material for write-ups.
The decisions they produced are already folded into the codebase and documented in
[`../docs/CHANGELOG.md`](../docs/CHANGELOG.md) **old §8** (the canonical A/B results log).

## Each harness → its finding

| Harness | PIPELINE_HISTORY anchor | What it measured / decided |
|---|---|---|
| `ab_fallback.py` | old §8.1 | Anchor-gated tier-2 → `_sam` fallback. Decision: `chain_crop_fallback=True`, `chain_crop_min_image_score=0.7`. Gate-only criterion rejected (over-zoom passes geometry); `image_score` is the discriminating pre-propagation signal. |
| `ab_seed.py` | old §8.3 | Seed ablation (mask vs box vs points vs negatives). Decision: default seed stays **`box_pos`** (won); `seed_negatives`/`seed_mask` stay default-off. Mask-seed-as-AUTO and negatives-as-default both rejected. |
| `ab_underfill.py` | old §8.4 | `box_margin_frac` (%-of-bbox pad) for under-filled anchors. Validated (RIML c25: queue 4→0) but **default-OFF** as a targeted retry lever; frac-as-universal-default and mask-for-underfill both rejected. |
| `ab_tier2_wide.py` | old §8.5 → §8.8 | Wider tier-2 A/B (15 chains: improved 3, regressed 0, net −10). Led to the lab-approved decision to flip tier-2 default-on for flagged chains, **landed** as the `batch.py` auto second-pass (`tier2_on_flagged`, old §8.8). (Supersedes the earlier removed `ab_tier2.py`.) |
| `sweep_dilation.py` | (qc skeleton-dilation sweep) | Sweeps `qc_skeleton_dilation_px` for the `skeleton_contained` signal. *Finding location in PIPELINE_HISTORY to confirm, flagged in the reorg needs-decision list.* |

## Running a harness

These scripts do `from pipeline import ...` / `from sam2_utils import ...`, which need the **repo
root** on `sys.path`. After the reorg move into `experiments/`, run them as a module from the repo
root (not `py -3 experiments/ab_seed.py`, which would put `experiments/` on the path instead):

```
py -3 -m experiments.ab_seed       # from the repo root
```

They write to `config.OUTPUT_ROOT`'s parent (e.g. `…/ab_seed/`), so output location is unaffected by
the move. They require torch + a GPU + the CATMAID-derived data, like the real pipeline.

## Disposable artifacts (flagged, NOT deleted this pass)

- **`*.log`**, captured stdout from the runs above (~1.2 MB total). Disposable; the findings live in
  PIPELINE_HISTORY old §8. Currently git-tracked. `experiments/*.log` is git-ignored going forward, so
  *new* logs won't be tracked; the existing tracked ones await an explicit delete go-ahead.
- **`ab_figs/`**, A/B comparison PNGs (baseline vs tier-2 overlays, ~6.2 MB). Disposable; awaiting a
  delete go-ahead.

See [`../docs/explanation/design-notes.md`](../docs/explanation/design-notes.md) §8 (item 32) for the full keep/archive/delete
needs-decision checklist.
