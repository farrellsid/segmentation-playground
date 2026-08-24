#!/bin/bash
# run_reprop_corrected_seed.sh: Slurm ARRAY job, one task per manually-corrected chain,
# re-propagating from the corrected anchor mask in two variants (mask-seed, box-seed).
#
# NOT YET SMOKE-TESTED ON NARVAL (written and lint-checked locally, following
# run_exp.sh's structure exactly; verify with one array task before trusting the full
# run, same discipline every other job in this directory expects).
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
#   6. Submit this array job, one task per row of corrected_chains.csv.
#   7. Pull the two output trees back (rsync -L), same as step 2.
#
# Submit:
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   N=$(($(wc -l < cluster/corrected_chains.csv) - 1))   # rows minus the header
#   sbatch --array=0-$((N - 1))%8 \
#       --export=ALL,WORKING_TREE=/scratch/$USER/manual_verify_AIYL_AIYR,\
#OUT_MASK=/scratch/$USER/reprop_maskseed,OUT_BOX=/scratch/$USER/reprop_boxseed \
#       cluster/run_reprop_corrected_seed.sh
#
# Each task propagates ONE chain (both variants), much lighter than a full per-slice
# generation array task, hence the smaller --mem/--cpus-per-task/--time below relative
# to run_exp.sh; tighten further once a real task's _timing.csv-equivalent (this script
# does not write one yet) shows the real wall-clock.

#SBATCH --account=def-mzhen        # bare account; Slurm auto-routes to _gpu via --gres
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00            # both variants, one chain; raise if a real run needs more
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail

: "${WORKING_TREE:?set WORKING_TREE (the corrected tree on Narval storage) via --export}"
: "${OUT_MASK:?set OUT_MASK (mask-seed variant output tree) via --export}"
: "${OUT_BOX:?set OUT_BOX (box-seed variant output tree) via --export}"
MANIFEST="${MANIFEST:-cluster/corrected_chains.csv}"
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
source $HOME/sam2env/bin/activate

cd "$REPO"
mkdir -p cluster/logs

# --- pick this task's chain from the manifest (find_corrected_chains.py's --out-csv,
# header: neuron,chain_idx,anchor_z; row 0 of the array = the FIRST data row, line 2
# of the file, one header line offset from the array index) --------------------------
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
if [ -z "$ROW" ]; then
    echo "[reprop-array] no row $((SLURM_ARRAY_TASK_ID + 2)) in $MANIFEST" >&2
    exit 1
fi
NEURON=$(echo "$ROW" | cut -d, -f1)
CHAIN_IDX=$(echo "$ROW" | cut -d, -f2)
echo "[reprop-array] task=$SLURM_ARRAY_TASK_ID neuron=$NEURON chain=$CHAIN_IDX git=$(git rev-parse --short HEAD)"

python experiments/propagate_from_corrected_seed.py \
    --working "$WORKING_TREE" \
    --neuron "$NEURON" --chain "$CHAIN_IDX" \
    --out-mask "$OUT_MASK" --out-box "$OUT_BOX" "${VARIANT_ARGS[@]}"

echo "[reprop-array] task=$SLURM_ARRAY_TASK_ID neuron=$NEURON chain=$CHAIN_IDX done"
