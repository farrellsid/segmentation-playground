#!/bin/bash
# run_exp.sh: Slurm ARRAY job for one resolution-experiment variant on Narval.
#
# The resolution experiments (original_fullres / original_tier2forced / original_bigimg)
# run a fixed neuron subset (presets.EXP_NEURONS) split into small chunks so they run in
# parallel, isolate AVAL's many chains to one task, and survive a single task failing.
# Chunks come from cluster/exp_neuron_chunks.txt (1 neuron/line, 16 lines -> array 0-15, so the
# heavier per-slice presets isolate each neuron and AVAL cannot wall a whole 2-neuron chunk);
# regenerate with:
#   py -3 cluster/make_chunks.py --chunk-size 1 --neurons <EXP_NEURONS...> --out cluster/exp_neuron_chunks.txt
#
# Submit ONCE PER VARIANT, passing the preset via --export (same script, three arrays =
# 3 x 8 = 24 GPU tasks; add the full-run baseline you already have for the 4th comparison):
#   cd ~/projects/def-mzhen/fsid/segmentation-playground
#   sbatch --job-name=exp_fullres     --export=ALL,EXP_PRESET=original_fullres     cluster/run_exp.sh
#   sbatch --job-name=exp_tier2forced --export=ALL,EXP_PRESET=original_tier2forced cluster/run_exp.sh
#   sbatch --job-name=exp_bigimg      --export=ALL,EXP_PRESET=original_bigimg      cluster/run_exp.sh
# Then merge each variant's shards into one tree (submit as an afterok dependency):
#   sbatch --dependency=afterok:<fullres_jobid>     --export=ALL,EXP_PRESET=original_fullres     cluster/run_merge_exp.sh
#   sbatch --dependency=afterok:<tier2forced_jobid> --export=ALL,EXP_PRESET=original_tier2forced cluster/run_merge_exp.sh
#   sbatch --dependency=afterok:<bigimg_jobid>      --export=ALL,EXP_PRESET=original_bigimg      cluster/run_merge_exp.sh
#
# Shards -> /scratch/$USER/<preset>_shards/chunk_i; merged tree -> /scratch/$USER/<preset>_merged
# (all distinct from the full run's target_shards/target_merged). Pull merged trees with
# Globus or tar (docs/how-to/run-on-narval.md). NOTE: original_bigimg (image_size=2048) is
# the risky one: it may OOM on a 40GB A100 or fail the post-build image_size assertion, so an
# empty chunk shard is expected-failure, check its log + _run_meta.json.

#SBATCH --account=def-mzhen        # bare account; Slurm auto-routes to _gpu via --gres
#SBATCH --array=0-15%8             # 16 chunks (1 neuron each); match exp_neuron_chunks.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8          # within Narval's per-GPU ratio (<= 12 cores/GPU)
#SBATCH --mem=64G                  # higher than the full run: full-res frames + bigimg are heavier
#SBATCH --time=12:00:00            # per chunk; AVAL's chunk is the long pole
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail

: "${EXP_PRESET:?set EXP_PRESET (original_fullres | original_tier2forced | original_bigimg) via --export=ALL,EXP_PRESET=...}"

# --- paths (edit for your account) -------------------------------------------
REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/fsid/SAM2_test_NR_raw
SHARD_ROOT=/scratch/$USER/${EXP_PRESET}_shards
CHUNKS=$REPO/cluster/exp_neuron_chunks.txt

# --- environment (same module set as the full run) ---------------------------
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
source $HOME/sam2env/bin/activate

cd "$REPO"
mkdir -p cluster/logs "$SHARD_ROOT"

# --- pick this task's chunk of neurons ---------------------------------------
NEURONS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNKS")
if [ -z "$NEURONS" ]; then
    echo "[run_exp] no neurons on line $((SLURM_ARRAY_TASK_ID + 1)) of $CHUNKS" >&2
    exit 1
fi
echo "[run_exp] preset=$EXP_PRESET task=$SLURM_ARRAY_TASK_ID neurons: $NEURONS git=$(git rev-parse --short HEAD)"

# --no chunking of the neuron subset here beyond the array line: --neurons overrides the
# preset's baked EXP_NEURONS so this task runs only its 2 neurons, into its own shard.
python batch.py \
    --preset "$EXP_PRESET" \
    --neurons $NEURONS \
    --output-root "$SHARD_ROOT/chunk_${SLURM_ARRAY_TASK_ID}" \
    --frames-root "$SLURM_TMPDIR/frames"

echo "[run_exp] preset=$EXP_PRESET task=$SLURM_ARRAY_TASK_ID done"
