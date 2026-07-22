# How to run the SAM3 whole-set comparison on Narval

This is the SAM3-specific delta on top of [run-on-narval.md](run-on-narval.md). Read that
file first for the general Narval setup (account, SSH + Duo, cloning the repo, `SAM2_WORM_PATH`,
the `data/` csvs, the array/merge mechanics). This document covers only what changes when a run
adds `--backend sam3`: a separate checkpoint, a separate environment, and two output trees that
must never touch the SAM2 baselines.

Background: `batch.py --backend sam3 --sam3-checkpoint <dir>` routes the pipeline through the
HuggingFace SAM3 adapters in `sam2_utils/sam3_backend.py` instead of SAM2; `cluster/run_array.sh`
forwards `PRESET`, `SAM_BACKEND`, `SAM3_CKPT`, and `OUT_ROOT` from the `sbatch --export` line
through to `batch.py`. Design: `docs/superpowers/specs/2026-07-21-sam3-cluster-whole-set-eval-design.md`.
Motivation and the 2-chain result this scales up: `docs/explanation/sam3-bakeoff-findings.md`.

## What this run produces

Two SAM3 trees, matched to the two SAM2 baselines from the Phase 1 close-out so the comparison
is a clean model swap, nothing else changes:

| Config | SAM2 baseline preset | SAM3 run |
|---|---|---|
| Per-slice | `original_perslice_only_guard` | same preset, `--backend sam3` |
| Propagation | `original_tier2_s1forced_neg` | same preset, `--backend sam3` |

SAM2 is not re-run. Its baseline trees already exist; this only adds the two SAM3 trees.

## 1. Upload the checkpoint

The local checkpoint is `F:\sam3\huggingface` (HuggingFace format, about 3.4 GB: `config.json`,
`model.safetensors`, `sam3.pt`, the tokenizer/processor files). Push it once to a persistent
project path, not scratch (scratch can be purged; you do not want to re-upload 3.4 GB before a
retry):

```bash
scp -r "F:\sam3\huggingface" narval:projects/def-mzhen/<user>/sam3_checkpoint
```

If `scp` is slow or drops on a network this size, use Globus instead (Alliance's recommended
route for multi-GB transfers). Either way, the reader ends up with one path to pass as
`SAM3_CKPT`, for example `~/projects/def-mzhen/<user>/sam3_checkpoint`.

## 2. Environment: a fresh venv first, a container as the fallback

This is the part most likely to need iteration, so budget time for it and treat the exact
versions that end up working as a discovery to record, not something to assume in advance.

**Do not reuse `~/sam2env`.** SAM3 needs `transformers>=5.13`, newer than anything the existing
SAM2 environment was built against; a second, independent venv keeps the working SAM2 cluster
environment untouched if the SAM3 install goes wrong.

**Do not use conda/mamba.** Even though `transformers`' own docs sometimes suggest it, the
Alliance-supported route on Narval is the module system plus a venv (or, if that fails, the
container below). The same reasoning as the general runbook's opencv gotcha applies here: a pip
wheel that only depends on a missing module fails at import time, not at install time, and
conda does not sidestep that either.

```bash
cd ~/projects/def-mzhen/<user>/segmentation-playground

# Same non-torch dependencies batch.py needs regardless of backend (numpy, scipy, pandas,
# matplotlib, Pillow, psutil, cv2) come from these modules, not the venv, exactly as in
# the SAM2 setup. Start from the same module line as run-on-narval.md; if transformers
# later refuses to import against python/3.11, run `module avail python` and `module avail
# cuda` to find a newer pair and rebuild the venv against those instead.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a

virtualenv --no-download ~/sam3env && source ~/sam3env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch            # the Alliance wheelhouse build: prebuilt and already
                                         # CUDA-matched to the modules above, no torch.org URL needed
pip install transformers>=5.13          # not in the wheelhouse, pulled from PyPI, login node only
                                         # (compute nodes are offline, same rule as the SAM2 install)
```

`run_array.sh` activates `${VENV:-$HOME/sam2env}`, so every SAM3 `sbatch` below passes
`VENV=$HOME/sam3env` in its `--export` list to select this environment. A SAM2 run leaves `VENV`
unset and activates `~/sam2env` exactly as before.

**Record what worked.** Before submitting the smoke chunk, capture the exact combination that
imported and ran cleanly, `module list` plus `pip freeze | grep -Ei
'torch|transformers|tokenizers|safetensors|huggingface'`, so a future re-provision (a new
account, a purged venv) does not have to rediscover it by trial and error. A one-line note in
this file's git history or in `.git/sdd/` is enough.

### Fallback: an Apptainer container

If the venv route fails on a torch/CUDA mismatch (the Alliance wheelhouse's `torch` build does
not line up with the loaded `cuda`/`cudnn` modules for `transformers>=5.13`, or `transformers`
imports but errors the first time it touches the GPU), build a container image from the
known-good local stack instead of chasing module combinations. Locally, this project already
runs `torch 2.12.0+cu130` with `transformers 5.13.1` successfully, so that pairing is the known
target:

```bash
# on the login node (it has outbound internet; compute nodes do not) or any machine with
# Apptainer/Docker, then copy the .sif to Narval if built elsewhere
apptainer build sam3.sif docker://pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime
apptainer exec sam3.sif pip install --user 'transformers>=5.13' pandas scikit-image \
    opencv-python h5py tqdm requests psutil
```

Swap the base image tag for whatever combination actually matches the verified local pairing at
the time of building, and adjust the pip list against `requirements.txt`. Store `sam3.sif` on
project storage next to the checkpoint (it is large; build it once). In `run_array.sh` (or a
copy of it for this path), replace the `module load ... && source ~/sam3env/bin/activate` lines
with:

```bash
apptainer exec --nv --bind "$REPO":"$REPO" sam3.sif python batch.py ...
```

`--nv` maps the host NVIDIA driver into the container, without it `torch.cuda.is_available()`
is false inside the container the same way it is on a login node.

## 3. Smoke ONE chunk first

Before sizing or submitting the full array, confirm the environment loads, SAM3 loads the
checkpoint, and one chunk of chains runs end to end, and measure how long it actually takes:

```bash
python cluster/make_chunks.py      # writes cluster/neuron_chunks.txt, prints the chunk count N
```

```bash
sbatch --array=0-0 \
  --export=ALL,VENV=$HOME/sam3env,PRESET=original_perslice_only_guard,SAM_BACKEND=sam3,\
SAM3_CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint,\
OUT_ROOT=/scratch/$USER/target_perslice_only_guard_sam3_smoke \
  cluster/run_array.sh
```

`--array=0-0` on the `sbatch` command line overrides the `#SBATCH --array` line inside the
script, so no file edit is needed for the smoke test. Give the smoke run its own `_smoke`
`OUT_ROOT`, separate from the real run's, so there is nothing to reconcile afterward.

Check `cluster/logs/<jobname>-<jobid>_0.out` for the model load and the per-chain progress
lines, and the shard's `_timing.csv` for the per-chain seconds once it finishes. Multiply that
per-chain time by the chunk size and the number of chunks to size `--time` for the real array,
and remember SAM3 is roughly 3 to 4x slower per cell than SAM2 (see Caveats below), so do not
reuse the SAM2 array's `--time` value unchanged.

## 4. Submit the two full SAM3 whole-set runs

Once the smoke chunk confirms the environment and gives a walltime estimate, size the array from
`make_chunks.py`'s printed count and submit both configs. These are the two confirmed commands
(`cluster/run_array.sh`'s own usage comment carries the same pair):

```bash
sbatch --array=0-<N-1>%<concurrency> \
  --export=ALL,VENV=$HOME/sam3env,PRESET=original_perslice_only_guard,SAM_BACKEND=sam3,\
SAM3_CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint,\
OUT_ROOT=/scratch/$USER/target_perslice_only_guard_sam3 \
  cluster/run_array.sh

sbatch --array=0-<N-1>%<concurrency> \
  --export=ALL,VENV=$HOME/sam3env,PRESET=original_tier2_s1forced_neg,SAM_BACKEND=sam3,\
SAM3_CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint,\
OUT_ROOT=/scratch/$USER/target_tier2_s1forced_neg_sam3 \
  cluster/run_array.sh
```

Both runs read the same uploaded checkpoint and differ only in `PRESET` and `OUT_ROOT`. `PRESET`
picks the matching SAM2 baseline config (`original_perslice_only_guard` for per-slice,
`original_tier2_s1forced_neg` for propagation) so the only thing that changes between a SAM3
tree and its SAM2 counterpart is the backend. Always set `SAM3_CKPT` when `SAM_BACKEND=sam3`:
without it the adapter falls back to its local dev default path, which does not exist on Narval
and fails at model load.

**Merging and downloading (same shape as the SAM2 flow).** `OUT_ROOT` is the SHARD ROOT:
`run_array.sh` points its `SHARD_ROOT` at `OUT_ROOT`, so each array task writes its own
`$OUT_ROOT/chunk_<i>/` shard, exactly like the default SAM2 run and with no shared-tree
contention (no two tasks touch the same file or manifest). After the array finishes, merge and
download each SAM3 variant with the existing tools:

- **Merge.** `cluster/merge_shards.py` is already parameterized, so call it directly (rather than
  `run_merge.sh`, which hardcodes the SAM2 paths):

```bash
python cluster/merge_shards.py --shard-root /scratch/$USER/target_perslice_only_guard_sam3 \
    --out /scratch/$USER/target_perslice_only_guard_sam3_merged
python cluster/merge_shards.py --shard-root /scratch/$USER/target_tier2_s1forced_neg_sam3 \
    --out /scratch/$USER/target_tier2_s1forced_neg_sam3_merged
```

- **Download.** `cluster/stage_download.sh` auto-globs `/scratch/$USER/*_merged`, so it picks up
  both `*_sam3_merged` trees with no edit and tars each (dereferencing the shard symlinks) into
  `/scratch/$USER/downloads/`. Pull those tarballs with Globus or scp.

## 5. Score locally

Pull the two SAM3 trees down (see above) next to your local copies of the two SAM2 baseline
trees, then score all four in one call so the comparison table is generated consistently:

```bash
py -3 -m eval.merge_metric \
  --root <local>/original_perslice_only_guard \
  --root <local>/target_perslice_only_guard_sam3_merged \
  --root <local>/original_tier2_s1forced_neg \
  --root <local>/target_tier2_s1forced_neg_sam3_merged
```

This prints one summary line per tree: `foreign_frame_rate`, `dropout_rate`,
`mild_bleed_rate`, `boundary_on_membrane`, and `underfill`. `--root` can repeat as many times as
needed; `eval.retro_eval` (`py -3 -m eval.retro_eval --root <tree> [--root ...] [--membrane]`)
gives the same merge-metric numbers plus compute (`_timing.csv`) and legacy QC (`_manifest.csv`)
columns in one table, useful if you also want the per-chain wall-clock comparison, subject to
the manifest caveat above. Compare the SAM3 numbers against the SAM2 baseline numbers already in
`docs/CHANGELOG.md`'s 2026-07-20 Phase 1 close-out entry, and against the 2-chain bake-off
result in `docs/explanation/sam3-bakeoff-findings.md`, to see whether the SAM3 win holds at the
full target-worm scale.

## Caveats (read before trusting a number)

- **SAM3 is roughly 3 to 4x slower per cell than SAM2** (the bake-off's own timings). This is
  the main allocation driver: size `--time` and the array's total GPU-hours off the smoke
  chunk's measured walltime, not off the SAM2 array's existing settings.
- **`pred_iou` is NaN for SAM3 propagation** (the SAM2-specific `_track_step` hook that
  populates it never fires on the SAM3 adapter). This is inert for `eval.merge_metric`, which is
  mask-only and never reads `pred_iou`, but treat any other QC or analysis that reads `pred_iou`
  as disabled for SAM3 runs.
- **SAM3 must write to its own `OUT_ROOT` and never overwrite a SAM2 baseline tree.** Both
  `OUT_ROOT` values above are new paths; double-check before submitting that neither equals nor
  nests inside an existing SAM2 baseline tree. There is no code-level guard against this, only
  the path you choose.
