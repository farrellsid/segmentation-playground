#!/bin/bash
# run_eval_array.sh: Slurm ARRAY job to score ONE merged tree in parallel on CPUs.
#
# The merge-metric scorer (eval.merge_metric) runs a whole tree on one core; there is
# no reason to. This splits the work by neuron the same way run_exp.sh splits
# segmentation: each array task scores its own neuron subset (one line of a chunks
# file) and writes its own shard CSV, <tree>/_merge_metric.shard_<taskid>.csv, so no
# two tasks touch the same file. After the array finishes, stitch the shards into the
# canonical <tree>/_merge_metric.csv with eval.concat_merge_shards (see the tail note).
#
# This is a CPU job (no --gres): scoring reads saved masks + the raw EM for the
# membrane pass, no GPU. The membrane pass (MEMBRANE=1, the default) is the slow part
# (it reads the EM per z and runs the ridge filter); MEMBRANE=0 gives the fast
# Phase-0-only node/dropout numbers with no EM reads.
#
# Env (via --export=ALL,...):
#   TREE      (required) the merged tree to score, e.g. /scratch/$USER/<tree>_merged
#   CHUNKS    neuron chunks file; default cluster/exp_neuron_chunks.txt (the 16
#             EXP_NEURONS, one per line -> array 0-15, one neuron per task)
#   MEMBRANE  1 (default) runs the Phase-2 membrane pass; 0 skips it (--no-membrane)
#   VENV      venv to activate; default $HOME/sam2env (no torch needed, but its deps
#             cover pandas/skimage; SAM3 runs can leave this at the SAM2 env)
#   RADIUS    node-containment radius in _sam px; default is the scorer's own default
#   SCALE     override the _sam grid scale (only if TREE has no _run_meta.json)
#
# Submit (16-neuron tree, one neuron per task):
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   sbatch --array=0-15%8 --job-name=eval_<tree> \
#     --export=ALL,TREE=/scratch/$USER/<tree>_merged cluster/run_eval_array.sh
#   # then stitch (cheap, run on a login node or as an afterok CPU step):
#   py -3 -m eval.concat_merge_shards --tree /scratch/$USER/<tree>_merged

#SBATCH --account=def-mzhen         # CPU allocation (no --gres, so no _gpu routing)
#SBATCH --array=0-15%8              # 16 chunks (1 neuron each); match the chunks file
#SBATCH --cpus-per-task=2           # membrane ridge filter is single-tree; a couple cores is plenty
#SBATCH --mem=16G                   # scale-8 masks + one EM frame at a time; small
#SBATCH --time=03:00:00             # generous per-neuron guess; tighten from the first run
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail

: "${TREE:?set TREE=/scratch/\$USER/<tree>_merged via --export=ALL,TREE=...}"

# --- paths (edit for your account) -------------------------------------------
REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
# The membrane pass reads the raw EM through pipeline.load_frame_sam, same source the
# segmentation run used, so the worm path must be set exactly as in run_array.sh.
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/fsid/SAM2_test_NR_raw
CHUNKS=${CHUNKS:-$REPO/cluster/exp_neuron_chunks.txt}

# --- environment (same module set as the segmentation runs; CPU only) --------
# cv2/numpy/scipy/pandas/Pillow + skimage come from the modules; the venv adds the rest.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
VENV=${VENV:-$HOME/sam2env}
source "$VENV/bin/activate"

cd "$REPO"
mkdir -p cluster/logs

# --- pick this task's neuron subset ------------------------------------------
NEURONS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNKS")
if [ -z "$NEURONS" ]; then
    echo "[run_eval] no neurons on line $((SLURM_ARRAY_TASK_ID + 1)) of $CHUNKS" >&2
    exit 1
fi
echo "[run_eval] tree=$TREE task=$SLURM_ARRAY_TASK_ID neurons: $NEURONS git=$(git rev-parse --short HEAD)"

# --- optional flags -----------------------------------------------------------
MEMBRANE=${MEMBRANE:-1}
EXTRA_ARGS=()
if [ "$MEMBRANE" = "0" ]; then
    EXTRA_ARGS+=(--no-membrane)
fi
if [ -n "${RADIUS:-}" ]; then
    EXTRA_ARGS+=(--radius "$RADIUS")
fi
if [ -n "${SCALE:-}" ]; then
    EXTRA_ARGS+=(--scale "$SCALE")
fi

# --- score this task's subset into its own shard CSV -------------------------
SHARD_CSV="$TREE/_merge_metric.shard_${SLURM_ARRAY_TASK_ID}.csv"
python -m eval.merge_metric \
    --root "$TREE" \
    --neurons "$NEURONS" \
    --out-csv "$SHARD_CSV" \
    "${EXTRA_ARGS[@]}"

echo "[run_eval] tree=$TREE task=$SLURM_ARRAY_TASK_ID done -> $SHARD_CSV"

# After the whole array finishes, stitch the shards into <tree>/_merge_metric.csv:
#   py -3 -m eval.concat_merge_shards --tree "$TREE"
