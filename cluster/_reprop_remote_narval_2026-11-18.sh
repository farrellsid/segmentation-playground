#!/bin/bash
# Run on NARVAL (Nibi outage fallback, incident 1779) via: Get-Content cluster\_reprop_remote_narval_2026-11-18.sh -Raw | ssh narval bash -s
# (Duo prompt first, then it runs unattended.)
#
# 2026-11-18 batch: the five missing neurons (ALNL, RMDDL, URYDL, URYVR, SABVR), seeds re-verified in the GUI. 65 corrected chains,
# corrected chains (anchor mask differs from the pristine target_perslice_only_guard_sam3_merged
# source), 12 array tasks of 15. No frames tarball: it is only needed for Lucinda bundles.
#
# Needs, in this order, before it is run:
#   1. cluster/corrected_chains_missing5_2026-11-18.csv committed and pushed (this script git pulls it)
#   2. /scratch/$USER/reprop_missing5_2026-11-18.tar.gz uploaded
#        scp "D:\Zhen Lab\SAM2 Segmentation\segmentation-playground\cluster\reprop_missing5_2026-11-18.tar.gz" narval:/scratch/fsid/
#
# Backend: SAM2 (matching the most recently completed round, the Oct/Nov ADA/AVK/etc
# batch -- its own state.json shows "backend": "sam2", not SAM3, despite
# propagate_from_corrected_seed.py's general docstring preference for SAM3. Staying
# consistent with that real precedent also sidesteps needing to confirm ~/sam3env exists
# on Nibi, which was never checked (same open item the 2026-11-05 script flagged).
set -euo pipefail

TAG=missing5_2026-11-18
BATCH_SIZE=15

cd ~/projects/def-mzhen/fsid/segmentation-playground
git pull

MANIFEST=cluster/corrected_chains_${TAG}.csv
[ -f "$MANIFEST" ] || { echo "ABORT: $MANIFEST not on this clone; push it first, then re-run"; exit 1; }
N=$(( $(wc -l < "$MANIFEST") - 1 ))
N_TASKS=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))

STAGE_DIR=/scratch/$USER/manual_verify_${TAG}_batch
ARCHIVE=/scratch/$USER/reprop_${TAG}.tar.gz
[ -s "$ARCHIVE" ] || { echo "ABORT: $ARCHIVE missing or empty; upload it first"; exit 1; }
rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR"
tar -xzf "$ARCHIVE" -C "$STAGE_DIR"

STAGED=$(find "$STAGE_DIR" -mindepth 2 -maxdepth 2 -type d -name 'chain_*' | wc -l)
STATES=$(find "$STAGE_DIR" -name state.json | wc -l)
ZERO=$(find "$STAGE_DIR" -type f -size 0 | wc -l)
echo "manifest rows=$N staged chains=$STAGED state.json=$STATES zero-byte=$ZERO tasks=$N_TASKS"
if [ "$STAGED" -ne "$N" ] || [ "$STATES" -ne "$N" ] || [ "$ZERO" -ne 0 ]; then
    echo "ABORT: staged tree does not match the manifest (or has empty files); nothing submitted"
    exit 1
fi

# Narval: the script's own #SBATCH --account=def-mzhen / --gres=gpu:1 apply as is (A100).
SBATCH_OUT=$(sbatch --array=0-$((N_TASKS - 1))%8 \
    --export=ALL,WORKING_TREE=$STAGE_DIR,OUT_MASK=/scratch/$USER/reprop_maskseed_${TAG},OUT_BOX=/scratch/$USER/reprop_boxseed_${TAG},MANIFEST=$MANIFEST,VARIANT=mask,BATCH_SIZE=$BATCH_SIZE \
    cluster/run_reprop_corrected_seed.sh)
echo "$SBATCH_OUT"
JOBID=${SBATCH_OUT##* }
echo "reprop array job id: $JOBID"

PACK_OUT=$(sbatch --dependency=afterany:$JOBID \
    --job-name=tar-reprop-${TAG} --account=def-mzhen --cpus-per-task=1 --mem=4G --time=02:00:00 \
    --output=/scratch/$USER/tar-reprop-${TAG}-%j.out \
    --wrap="tar -czhf /scratch/$USER/reprop_maskseed_${TAG}.tar.gz -C /scratch/$USER reprop_maskseed_${TAG} && ls -lh /scratch/$USER/reprop_maskseed_${TAG}.tar.gz")
echo "$PACK_OUT"
echo "done"
