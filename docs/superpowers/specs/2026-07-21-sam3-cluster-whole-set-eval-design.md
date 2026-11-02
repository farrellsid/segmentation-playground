# SAM3 whole-set cluster evaluation, design (Phase 2)

Date: 2026-07-21
Status: proposed (awaiting review)
Follows: `docs/superpowers/specs/2026-07-21-sam3-pvs-bakeoff-design.md` (the 2-chain bake-off, done),
`docs/explanation/sam3-bakeoff-findings.md` (the result this scales up).

## Context

The 2-chain bake-off showed SAM 3 sharply cuts foreign-node bleed on the target worm, with SAM 3
per-slice the leading cell (lowest bleed, zero dropout on both chains). That is a strong first
signal on n=2 chains of one neuron. This phase confirms it at scale: run SAM 3 over the whole
target-worm neuron set on Compute Canada (Narval), score with the same merge metric, and compare to
the SAM 2 baselines we already have from the Phase-1 CCDB A/B.

## Goal

Run two SAM 3 configurations over all target-worm neurons (the same set the Phase-1 A/B covered) via
`batch.py` on the Narval cluster, and score them with `eval.merge_metric`:

- SAM 3 **per-slice**, matched to the Phase-1 winning SAM 2 config (`perslice_only` + blow-up guard,
  no generous), compared against that SAM 2 baseline tree.
- SAM 3 **propagation**, matched to the SAM 2 propagation baseline (crop tier-2, first-slice-forced,
  negatives on), compared against that SAM 2 baseline tree.

SAM 2 is NOT re-run: its baseline trees already exist (downloaded under
`F:\...\output_masks\original_*`), so this phase adds only the two SAM 3 trees and the comparison.

## Non-goals

No SAM 2 re-run, no PCS/text path, no finetuning, no change to SAM 2 default behavior. SAM 2 stays
the default backend everywhere; a run without `--backend sam3` is byte-identical to today.

## Design

### 1. Backend switch (the only pipeline code change)

- Add two fields to `PipelineConfig`: `backend: str = "sam2"` and `sam3_checkpoint: Optional[str] =
  None` (the local HuggingFace checkpoint dir; on Narval this is the uploaded path).
- Add `--backend {sam2,sam3}` and `--sam3-checkpoint PATH` to `batch.py`.
- Factor the four `setup.build_predictor` calls (in `_build_session` and `_build_gt_session`) into one
  helper `build_predictors(cfg) -> (image_predictor, video_predictor)`:
  - `cfg.backend == "sam2"`: today's `setup.build_predictor(kind=...)` path, unchanged.
  - `cfg.backend == "sam3"`: `Sam3ImagePredictor(cfg.sam3_checkpoint)` and
    `Sam3VideoPredictor(cfg.sam3_checkpoint)` (the adapters from the bake-off, already reviewed).
- Both session builders call the helper. Nothing else in the pipeline changes: the adapters present
  the SAM 2 predictor interface, so `run_chain`, per-slice, propagation, `save_masks`, the manifest,
  and resume all work unchanged. This is the whole point of the adapter design.

### 2. Which configs, and keeping SAM 3 output separate

The per-slice-vs-propagation choice is already a preset/config concern, so `--backend` is orthogonal:
reuse the exact presets that produced the SAM 2 baselines and add `--backend sam3` plus an
`--output-root` redirect so SAM 3 writes its own trees and never overwrites the SAM 2 baseline:

```
batch.py --preset <sam2-perslice-baseline-preset> --backend sam3 \
         --sam3-checkpoint <narval-ckpt> --output-root <...>/perslice_only_guard_sam3
batch.py --preset <sam2-propagation-baseline-preset> --backend sam3 \
         --sam3-checkpoint <narval-ckpt> --output-root <...>/tier2_s1forced_neg_sam3
```

The exact preset names that produced the two SAM 2 baseline trees are pinned during planning against
`sam2_utils/presets.py` (the Phase-1 close-out shipped `original_perslice_only_guard` and siblings; the
propagation baseline preset is confirmed there too, or added as a thin preset if missing). No new SAM 3
presets are needed; `--backend` + `--output-root` are enough.

### 3. Cluster wiring

`cluster/make_chunks.py` is unchanged (same neuron set, same chunking). `cluster/run_array.sh` gains a
way to pass the backend, checkpoint, preset, and output-root through to `batch.py` (via environment
variables the SBATCH script forwards, matching how it already parameterizes the preset). The array
still chunks by neuron so one model load amortizes over a chunk.

### 4. Narval deployment (a runbook, human-executed)

Narval login is Duo-MFA gated, so this cannot be run headless. The deliverable is the code plus a
`docs/how-to/run-sam3-on-narval.md` runbook covering:

1. Upload the checkpoint (`F:\sam3\huggingface`, about 3.4 GB) to a Narval project/scratch path.
2. Stand up `transformers` >= 5.13 in a FRESH virtual environment on Narval (not the existing SAM 2
   env), because the SAM 3 stack pulls newer `transformers`; a fresh venv avoids disturbing the
   working SAM 2 cluster env. Record the exact module loads and pip versions that work.
3. Submit the array for each of the two SAM 3 configs (`sbatch` with the backend/checkpoint/output-root
   env vars), sized `--array=0-<N-1>%<concurrency>` from `make_chunks.py`.
4. Merge shards (`merge_shards.py`), download the two SAM 3 trees.
5. Score locally with `eval.merge_metric` (or `eval/retro_eval.py`) and table the numbers next to the
   SAM 2 baselines.

Narval GPUs are large, so the local 6 GB constraint and CPU-offload settings do not bind there.

## Risks and open questions

1. **Narval dependency env (top risk).** `transformers` 5.13 + its deps in a fresh venv on Narval,
   with a compatible torch build for the cluster's CUDA. Mitigation: fresh venv, pin versions, smoke
   one chunk before the full array. The exact working env is discovered during deployment and recorded
   in the runbook.
2. **Compute and allocation.** SAM 3 is about 3 to 4x slower per cell than SAM 2 (bake-off timings).
   Over the whole neuron set times two configs, that is a real allocation ask. Mitigation: a
   single-chunk smoke first to measure per-chain walltime on Narval hardware, then size the array and
   request accordingly. Log the estimate before submitting the full run.
3. **Mask-format parity for scoring.** SAM 3 masks must save in the same on-disk shape SAM 2 writes so
   `merge_metric` scores both identically. The adapters already return pipeline-shaped masks and
   `save_masks` is backend-agnostic, but the plan includes a local single-chain `--backend sam3`
   `batch.py` run that saves and scores, to confirm parity before anything goes to the cluster.
4. **pred_iou is NaN for SAM 3 propagation** (the SAM2-specific `_track_step` hook does not fire).
   This is inert for the mask-only merge metric, as in the bake-off, but any QC that reads pred_iou on
   the cluster run should be treated as disabled for SAM 3.
5. **checkpoint upload** of 3.4 GB over the network; do it once to a persistent project path.

## Testing

- CPU-only, torch-free test that `build_predictors` routes on `cfg.backend`: monkeypatch
  `setup.build_predictor` and the two adapter classes to sentinels and assert the sam2 path calls the
  former and the sam3 path constructs the adapters with `cfg.sam3_checkpoint`. No GPU, no model load.
- CLI parse test for `--backend` / `--sam3-checkpoint` and the default (`sam2`).
- A local single-chain `--backend sam3 batch.py` run (GPU, manual) that writes and scores masks,
  confirming mask-format parity (risk 3) before the cluster runbook is executed.

## Success criteria

- `batch.py --backend sam3` runs a target-worm chain end to end locally and saves masks scorable by
  `merge_metric` identically to SAM 2.
- The runbook takes a reader from a fresh Narval shell to two scored SAM 3 trees.
- Final deliverable: SAM 3 per-slice and propagation merge-metric numbers over the whole target-worm
  set, tabled beside the SAM 2 baselines, with a clear verdict on whether the 2-chain SAM 3 win holds
  at scale. If it does, productionizing (making `--backend sam3` a first-class run mode, and a decision
  on default) is the follow-on.
