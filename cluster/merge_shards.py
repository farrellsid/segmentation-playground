"""merge_shards.py: stitch per-task output shards into one tree + ledger.

Each array task wrote to its own shard ``<shard_root>/chunk_<i>/`` with a whole
``<neuron>/chain_<idx>/`` subtree plus its own ``_manifest.csv`` / ``_timing.csv`` /
``_triage.csv``. Neurons are disjoint across shards (make_chunks guarantees it), so
merging is a clean union with no dedup:

  1. symlink every shard's ``<neuron>/`` dir into a single merged tree,
  2. concatenate the shard manifests and timing CSVs,
  3. rebuild the global ``_triage.csv`` with ``batch.build_triage_queue`` (it reads
     each chain's on-disk ``qc.csv`` through the symlinks).

Symlinks keep the shards intact so the merge is re-runnable. To pull the merged
tree off the cluster as real files, use ``rsync -L`` (follow symlinks).

Usage (from the repo root)
--------------------------
    py -3 cluster/merge_shards.py \
        --shard-root /scratch/$USER/target_shards \
        --out /scratch/$USER/target_merged
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Run standalone (python cluster/merge_shards.py) or as a module: put the repo root
# on the path so `import batch` / `sam2_utils` resolve either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import batch


def find_shards(shard_root: Path) -> List[Path]:
    """The ``chunk_*`` dirs under ``shard_root``, sorted by chunk index."""
    shards = [p for p in shard_root.glob("chunk_*") if p.is_dir()]
    return sorted(shards, key=lambda p: int(p.name.split("_")[1]))


def link_neuron_dirs(shards: List[Path], out_root: Path) -> int:
    """Symlink each shard's per-neuron dirs into ``out_root``. Returns the count.

    A neuron dir is anything that is not one of the top-level CSVs / the mp4 dir.
    Because chunks are disjoint, a target name should never already exist; if it
    does (a re-run, or overlapping chunks), the existing link is replaced.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    skip = {"_manifest.csv", "_triage.csv", "_timing.csv", "mp4"}
    n = 0
    for shard in shards:
        for entry in shard.iterdir():
            if entry.name in skip or not entry.is_dir():
                continue
            link = out_root / entry.name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(entry.resolve(), link)
            n += 1
    return n


def concat_csv(shards: List[Path], name: str, out_path: Path) -> int:
    """Concatenate a per-shard CSV across shards into ``out_path``. Returns row count."""
    frames = [pd.read_csv(shard / name) for shard in shards if (shard / name).exists()]
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_path, index=False)
    return len(df)


def merge(shard_root: Path, out_root: Path) -> None:
    shards = find_shards(shard_root)
    if not shards:
        raise SystemExit(f"[merge] no chunk_* shards under {shard_root}")
    print(f"[merge] {len(shards)} shards -> {out_root}")

    n_linked = link_neuron_dirs(shards, out_root)
    n_manifest = concat_csv(shards, "_manifest.csv", out_root / "_manifest.csv")
    n_timing = concat_csv(shards, "_timing.csv", out_root / "_timing.csv")
    print(f"[merge] linked {n_linked} neuron dirs; "
          f"manifest rows={n_manifest}, timing rows={n_timing}")

    # Carry a shard's run provenance up to the merged root. Each shard wrote its own
    # _run_meta.json (identical preset/knobs/git across chunks, only the per-chunk
    # `neurons` differs), so copying the first shard's copy documents the merged tree
    # for a post-hoc, unobserved cluster run. link_neuron_dirs skips it (not a dir).
    src_meta = next((s / "_run_meta.json" for s in shards
                     if (s / "_run_meta.json").exists()), None)
    if src_meta is not None:
        (out_root / "_run_meta.json").write_text(src_meta.read_text())
        print(f"[merge] copied run provenance from {src_meta}")

    # Rebuild the global triage queue from the merged manifest. build_triage_queue
    # reads each chain's qc.csv off disk (through the symlinks), so it does not care
    # that the neuron dirs are links rather than real dirs.
    manifest = pd.read_csv(out_root / "_manifest.csv")
    batch.build_triage_queue(out_root, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", type=Path, required=True,
                    help="dir containing the chunk_* shards")
    ap.add_argument("--out", type=Path, required=True,
                    help="merged output tree to create")
    args = ap.parse_args()
    merge(args.shard_root, args.out)


if __name__ == "__main__":
    main()
