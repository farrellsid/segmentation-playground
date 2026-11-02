#!/bin/bash
# run_merge_exp.sh: merge one experiment variant's array shards into a single tree.
#
# Submit as an afterok dependency of that variant's run_exp.sh array, passing the SAME
# EXP_PRESET so it finds the right shards:
#   sbatch --dependency=afterok:<jobid> --export=ALL,EXP_PRESET=original_fullres cluster/run_merge_exp.sh
#
# Reuses cluster/merge_shards.py (generic): symlinks each chunk's neuron dirs into one
# tree, concatenates the manifests/timing, rebuilds the triage queue, and copies a shard's
# _run_meta.json up so the merged tree carries its own provenance. CPU-only and quick.

#SBATCH --job-name=exp-merge
#SBATCH --account=def-mzhen        # bare account; auto-routes to _cpu (no --gres here)
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail

: "${EXP_PRESET:?set EXP_PRESET (original_fullres | original_tier2forced | original_bigimg) via --export=ALL,EXP_PRESET=...}"

REPO=$HOME/projects/def-mzhen/fsid/segmentation-playground
SHARD_ROOT=/scratch/$USER/${EXP_PRESET}_shards
MERGED=/scratch/$USER/${EXP_PRESET}_merged

# merge imports batch (pulls pipeline/torch), so load the same modules. CPU-only.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 cudnn/9.2.1.18 \
    opencv/4.13.0 scipy-stack/2026a ipykernel/2026a
source $HOME/sam2env/bin/activate

cd "$REPO"
python cluster/merge_shards.py --shard-root "$SHARD_ROOT" --out "$MERGED"
echo "[run_merge_exp] $EXP_PRESET merged -> $MERGED"
