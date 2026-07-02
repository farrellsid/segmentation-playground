#!/bin/bash
# run_merge.sh: merge the array's output shards into one tree + ledger.
#
# Submit as a dependency so it runs only after the whole array succeeds:
#   sbatch --dependency=afterok:<arrayjobid> cluster/run_merge.sh
#
# CPU-only and quick (symlinks + CSV concat + one triage rebuild), so no GPU.

#SBATCH --job-name=sam2-merge
#SBATCH --account=def-mzhen
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail

REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
SHARD_ROOT=/scratch/$USER/target_shards
MERGED=/scratch/$USER/target_merged

# merge imports batch (which pulls pipeline/torch), so load the same modules as the
# array job. CPU-only: the cuda module just sets paths, no GPU needed here.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
source $HOME/sam2env/bin/activate

cd "$REPO"
python cluster/merge_shards.py --shard-root "$SHARD_ROOT" --out "$MERGED"
echo "[run_merge] merged tree at $MERGED"
