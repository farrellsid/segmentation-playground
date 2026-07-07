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

for name in original_fullres original_wholeimg_s4 original_tier2forced original_bigimg; do
    d=/scratch/$USER/${name}_merged
    if [ ! -d "$d" ]; then
        echo "[stage] SKIP (missing, its merge may have had no shards): $d"
        continue
    fi
    echo "[stage] tarring $d -> $OUT/${name}_merged.tar.gz"
    # -h dereferences the shard symlinks so the archive holds real files.
    tar -czhf "$OUT/${name}_merged.tar.gz" -C "$(dirname "$d")" "${name}_merged"
done

echo "[stage] done. Ready-to-pull tarballs:"
ls -lh "$OUT"
