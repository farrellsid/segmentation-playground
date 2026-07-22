#!/bin/bash
# run_array.sh: Slurm array job for the target-worm batch on Narval.
#
# Each array task processes one chunk of neurons (one line of neuron_chunks.txt)
# with batch.py, writing to its own output shard so parallel tasks never share the
# top-level manifest/triage CSVs. Merge the shards afterwards with merge_shards.py
# (submit it as an afterok dependency; see the submit note at the bottom).
#
# Before submitting:
#   1. py -3 cluster/make_chunks.py            # writes cluster/neuron_chunks.txt
#   2. set --array below to 0-<N-1>%<concurrency> using the N it prints
#   3. confirm the module names in an salloc session (they vary by cluster/StdEnv)
#
# Submit:
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   sbatch cluster/run_array.sh
#
# Optional SAM3 A/B (unset means an unchanged SAM2 run, same as before these existed).
# PRESET selects the config; the two SAM3 whole-set runs are:
#   sbatch --export=ALL,PRESET=original_perslice_only_guard,SAM_BACKEND=sam3,SAM3_CKPT=/path/to/ckpt,OUT_ROOT=/scratch/$USER/target_perslice_only_guard_sam3 cluster/run_array.sh
#   sbatch --export=ALL,PRESET=original_tier2_s1forced_neg,SAM_BACKEND=sam3,SAM3_CKPT=/path/to/ckpt,OUT_ROOT=/scratch/$USER/target_tier2_s1forced_neg_sam3 cluster/run_array.sh

#SBATCH --job-name=sam2-target
#SBATCH --account=def-mzhen        # bare account; Slurm auto-routes to _gpu via --gres
#SBATCH --array=0-18%8            # 19 chunks (default chunk-size 7), <= 8 at once; match make_chunks output
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8         # within Narval's per-GPU ratio (<= 12 cores/GPU)
#SBATCH --mem=48G                 # <= ~124G/GPU on Narval
#SBATCH --time=12:00:00           # generous first-pass guess; tighten from _timing.csv
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail

# --- paths (edit for your account) -------------------------------------------
REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/fsid/SAM2_test_NR_raw
# Each array task writes its own chunk_<id> shard here, merged afterwards. The SAM3
# runs set OUT_ROOT to point this at a SAM3-specific root, so shards stay per-task
# (no shared-tree contention) and merge_shards.py / stage_download.sh work unchanged.
SHARD_ROOT=${OUT_ROOT:-/scratch/$USER/target_shards}
CHUNKS=$REPO/cluster/neuron_chunks.txt

# --- environment -------------------------------------------------------------
# opencv/scipy-stack/ipykernel supply cv2, numpy/scipy/pandas/matplotlib/Pillow, and
# psutil (they live in modules, not the venv), so load them BEFORE activating the venv.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
source $HOME/sam2env/bin/activate

cd "$REPO"
mkdir -p cluster/logs "$SHARD_ROOT"

# --- pick this task's chunk of neurons ---------------------------------------
# array id is 0-based; sed lines are 1-based, so read line (id + 1).
NEURONS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNKS")
if [ -z "$NEURONS" ]; then
    echo "[run_array] no neurons on line $((SLURM_ARRAY_TASK_ID + 1)) of $CHUNKS" >&2
    exit 1
fi
echo "[run_array] task $SLURM_ARRAY_TASK_ID neurons: $NEURONS"

# --- optional backend pass-through --------------------------------------------
# Unset, these reproduce the SAM2 baseline run exactly: PRESET defaults to original,
# SAM_BACKEND defaults to sam2 (explicit, the same effective value the preset already
# sets), and SAM3_CKPT stays empty so no extra flag is appended. OUT_ROOT (the shard
# root) is handled above where SHARD_ROOT is set, so SAM3 shards stay per-task like SAM2.
# The two SAM3 whole-set runs set PRESET to the matching baseline preset
# (original_perslice_only_guard for per-slice, original_tier2_s1forced_neg for propagation).
PRESET=${PRESET:-original}
SAM_BACKEND=${SAM_BACKEND:-sam2}
SAM3_CKPT=${SAM3_CKPT:-}

BACKEND_ARGS=(--backend "$SAM_BACKEND")
if [ -n "$SAM3_CKPT" ]; then
    BACKEND_ARGS+=(--sam3-checkpoint "$SAM3_CKPT")
fi

# --- run ---------------------------------------------------------------------
# Frame cache goes to node-local $SLURM_TMPDIR (fast, private, auto-cleaned) so
# concurrent tasks do not contend on the network filesystem regenerating JPEGs.
# Each task writes its own shard: $SHARD_ROOT/chunk_<id> (SHARD_ROOT = OUT_ROOT when set).
python batch.py \
    --preset "$PRESET" \
    --neurons $NEURONS \
    --output-root "$SHARD_ROOT/chunk_${SLURM_ARRAY_TASK_ID}" \
    --frames-root "$SLURM_TMPDIR/frames" \
    "${BACKEND_ARGS[@]}"

echo "[run_array] task $SLURM_ARRAY_TASK_ID done"

# After the whole array finishes, merge the shards into one tree + ledger:
#   sbatch --dependency=afterok:<arrayjobid> cluster/run_merge.sh
# (run_merge.sh just activates the env and calls cluster/merge_shards.py)
