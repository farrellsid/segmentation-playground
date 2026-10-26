#!/bin/bash
# run_reprop_corrected_seed.sh: Slurm ARRAY job, one task per BATCH of manually-
# corrected chains, re-propagating from the corrected anchor mask in two variants
# (mask-seed, box-seed). Each task builds ONE video predictor (SAM2 by default, SAM3
# via SAM_BACKEND=sam3, see below) and reuses it across its whole batch of chains.
#
# Was one task per chain until the 2026-10-25 round (286 chains): real per-task
# timing showed ~1-3 min of actual propagation work each, but every task also paid
# a full SAM2 checkpoint load, unmeasured and repeated 286 times over. Batching
# amortizes that load across BATCH_SIZE chains per task instead of paying it once
# per chain, and also means far fewer separate scheduling requests competing for
# GPU time on a shared, often-busy partition.
#
# Round trip this is one leg of (docs/how-to/run-on-narval.md has the fuller version):
#   1. Generate per-slice masks on Narval (batch.py --preset original_perslice_only_guard),
#      already-documented, no new script.
#   2. Pull the merged tree to local (rsync -L).
#   3. Correct anchor frames locally in gui.py (experiments/make_review_tree.py sets up
#      the working copy first).
#   4. Find which chains actually changed: py -3 experiments/find_corrected_chains.py
#      --source <original tree> --working <working tree> --neurons <list> --out-csv
#      cluster/corrected_chains.csv, THIS SCRIPT reads that CSV.
#   5. Push the working tree (just the corrected chains are enough, state.json +
#      masks/) back to Narval project storage.
#   6. Submit this array job, one task per BATCH_SIZE rows of corrected_chains.csv.
#   7. Pull the two output trees back (rsync -L), same as step 2.
#
# Submit:
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   N=$(($(wc -l < cluster/corrected_chains.csv) - 1))   # rows minus the header
#   BATCH_SIZE=15
#   N_TASKS=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))      # ceil(N / BATCH_SIZE)
#   sbatch --array=0-$((N_TASKS - 1))%8 \
#       --export=ALL,WORKING_TREE=/scratch/$USER/manual_verify_AIYL_AIYR,\
#OUT_MASK=/scratch/$USER/reprop_maskseed,OUT_BOX=/scratch/$USER/reprop_boxseed,\
#OUT_FRAMES=/scratch/$USER/reprop_frames,BATCH_SIZE=15 \
#       cluster/run_reprop_corrected_seed.sh
#
# BATCH_SIZE defaults to 15 if not set. OUT_FRAMES is optional: set it to persist
# each chain's prepared frames to scratch instead of letting them vanish with
# $SLURM_TMPDIR when the task ends (see the SAM2_FRAMES_ROOT comment below for why
# they'd otherwise be lost). Without it, masks only, no frames survive the task.
#
# For SAM3 (matching the ORIGINAL masks this script reseeds from, which are already
# SAM3-generated, see docs/how-to/run-sam3-on-narval.md), add to the --export list:
#   VENV=$HOME/sam3env,SAM_BACKEND=sam3,SAM3_CKPT=$HOME/projects/def-mzhen/<user>/sam3_checkpoint
# Unset, this script is SAM2-only, exactly its pre-2026-10-25 behaviour.
#
# --time below is sized for a full BATCH_SIZE=15 batch (one predictor load + up to
# 15 chains' worth of propagation); shrink it if BATCH_SIZE is set smaller.

#SBATCH --account=def-mzhen        # bare account; Slurm auto-routes to _gpu via --gres
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00            # one predictor load + a batch of chains; raise if needed
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail

: "${WORKING_TREE:?set WORKING_TREE (the corrected tree on Narval storage) via --export}"
: "${OUT_MASK:?set OUT_MASK (mask-seed variant output tree) via --export}"
: "${OUT_BOX:?set OUT_BOX (box-seed variant output tree) via --export}"
MANIFEST="${MANIFEST:-cluster/corrected_chains.csv}"
# optional: persist prepared frames to scratch instead of losing them with
# $SLURM_TMPDIR. Unset means the old behaviour, masks only.
FRAMES_ARGS=()
if [ -n "${OUT_FRAMES:-}" ]; then
    FRAMES_ARGS=(--out-frames "$OUT_FRAMES")
fi
# both (default, unchanged) or mask. Mask-seed beat box-seed on overfill on all four
# AIA/AIY sides while matching its underfill fix (2026-08-14), so a run that only wants
# the winner halves its GPU time with VARIANT=mask. OUT_BOX stays required either way;
# nothing is written there when the box variant is skipped.
VARIANT="${VARIANT:-both}"
case "$VARIANT" in
    both) VARIANT_ARGS=() ;;
    mask) VARIANT_ARGS=(--skip-box-variant) ;;
    *)    echo "[reprop-array] VARIANT must be 'both' or 'mask', got '$VARIANT'" >&2; exit 2 ;;
esac

# --- paths (edit for your account, matches run-on-narval.md) -----------------
REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/fsid/SAM2_test_NR_raw

# Frame cache and per-chain views go to node-local $SLURM_TMPDIR (fast, private,
# auto-cleaned), the same place run_array.sh and run_exp.sh put them via their
# --frames-root flag. This script drives propagate_from_corrected_seed.py, which has
# no such flag and reads sam2_utils.config.FRAMES_ROOT, so the env override is how it
# gets told. Leaving it unset is not harmless: the default is the Windows path
# F:\ZhenLab\Data, which on Linux is a RELATIVE directory name that gets created
# under the repo, and the legacy _sam frame prep then builds its view out of symlinks
# whose relative targets resolve against the wrong directory. That is what killed all
# 15 legacy chains of the 2026-08-21 arrays; tier-2 chains survived only because they
# write real JPEGs instead of linking.
export SAM2_FRAMES_ROOT=${SLURM_TMPDIR:-/tmp}/frames

# --- environment (same module set as every other job here) -------------------
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
# VENV/SAM_BACKEND/SAM3_CKPT: same optional pass-through run_array.sh already uses for
# its SAM3 A/B runs. Unset reproduces the old SAM2-only behaviour exactly (VENV defaults
# to sam2env, SAM_BACKEND to sam2, no --sam3-checkpoint flag appended). The ORIGINAL
# masks this script reseeds from are already SAM3 (target_perslice_only_guard_sam3_merged);
# this script itself only grew SAM3 support in the 2026-10-25 round.
VENV=${VENV:-$HOME/sam2env}
source "$VENV/bin/activate"

cd "$REPO"
mkdir -p cluster/logs

SAM_BACKEND=${SAM_BACKEND:-sam2}
SAM3_CKPT=${SAM3_CKPT:-}
BACKEND_ARGS=(--backend "$SAM_BACKEND")
if [ -n "$SAM3_CKPT" ]; then
    BACKEND_ARGS+=(--sam3-checkpoint "$SAM3_CKPT")
fi

# --- this task's row range (find_corrected_chains.py's --out-csv, header:
# neuron,chain_idx,anchor_z; row 0 of the array owns data rows [0, BATCH_SIZE),
# propagate_from_corrected_seed.py's --row-start/--row-count read the SAME
# 0-based-after-header indexing pandas.read_csv().iloc[] uses) ------------------
BATCH_SIZE="${BATCH_SIZE:-15}"
ROW_START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
echo "[reprop-array] task=$SLURM_ARRAY_TASK_ID rows=[$ROW_START, $((ROW_START + BATCH_SIZE))) git=$(git rev-parse --short HEAD)"

python experiments/propagate_from_corrected_seed.py \
    --working "$WORKING_TREE" \
    --manifest "$MANIFEST" --row-start "$ROW_START" --row-count "$BATCH_SIZE" \
    --out-mask "$OUT_MASK" --out-box "$OUT_BOX" "${VARIANT_ARGS[@]}" "${FRAMES_ARGS[@]}" \
    "${BACKEND_ARGS[@]}"

echo "[reprop-array] task=$SLURM_ARRAY_TASK_ID done"
