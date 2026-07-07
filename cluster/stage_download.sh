#!/bin/bash
# stage_download.sh: package the finished experiment outputs into single, ready-to-pull
# tarballs so the morning download is one fast transfer instead of many small files.
#
# Submit AFTER the four merges, as a dependency so it fires the moment they finish
# (regardless of individual success, afterany, so a failed bigimg does not block it):
#   sbatch --dependency=afterany:<merge1>:<merge2>:<merge3>:<merge4> cluster/stage_download.sh
#
# Each <preset>_merged tree is a symlink forest into the shards; `tar -h` dereferences
# those links so the archive holds REAL files (the same fix as `rsync -L`). One .tar.gz
# per variant lands in /scratch/$USER/downloads/, which is trivial to pull with Globus or
# scp (single file, no follow-symlinks concern). CPU-only and quick.

#SBATCH --job-name=exp-stage-dl
#SBATCH --account=def-mzhen        # bare account; routes to _cpu (no --gres)
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail

OUT=/scratch/$USER/downloads
mkdir -p "$OUT" cluster/logs

# Auto-discover every merged tree under scratch (so new variants are picked up without
# editing this script), and skip any that is already tarred (idempotent: re-running only
# packages newly-merged variants).
for d in /scratch/$USER/*_merged; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    tb="$OUT/${name}.tar.gz"
    if [ -f "$tb" ]; then
        echo "[stage] skip (already tarred): $tb"
        continue
    fi
    echo "[stage] tarring $d -> $tb"
    # -h dereferences the shard symlinks so the archive holds real files.
    tar -czhf "$tb" -C "$(dirname "$d")" "$name"
done

echo "[stage] done. Ready-to-pull tarballs:"
ls -lh "$OUT"
