#!/bin/bash
# run_py.sh: run one CPU python step (merge_shards / concat_merge_shards / retro_eval)
# inside the cluster module set and venv, as a dependency job so it never runs on the
# login node. The command to run is passed as arguments:
#   sbatch --dependency=afterok:<id> cluster/run_py.sh python cluster/merge_shards.py --shard-root X --out Y
#   sbatch --dependency=afterok:<id> cluster/run_py.sh python -m eval.concat_merge_shards --tree Z
# Defaults suit the light steps (symlink + CSV concat). Override --mem/--time on the
# sbatch line for the heavier ones (retro_eval with --membrane).

#SBATCH --account=def-mzhen        # bare account; no --gres, so it routes to CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail

REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
export SAM2_WORM_PATH=$HOME/projects/def-mzhen/fsid/SAM2_test_NR_raw

# Same modules as the array jobs: merge_shards and retro_eval import batch, which pulls
# pipeline and torch, so the module set has to match. CPU only, cuda just sets paths.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
VENV=${VENV:-$HOME/sam2env}
source "$VENV/bin/activate"

cd "$REPO"
mkdir -p cluster/logs

echo "[run_py] $*"
"$@"
echo "[run_py] done"
