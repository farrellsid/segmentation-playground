"""concat_merge_shards.py: stitch a sharded merge-metric run into one CSV.

The sharded eval array (cluster/run_eval_array.sh) splits one merged tree across
CPUs: each task runs eval.merge_metric on its own neuron subset and writes
``<tree>/_merge_metric.shard_<i>.csv``. This helper concatenates those shard CSVs
back into the canonical ``<tree>/_merge_metric.csv`` (the same per-frame file a
single-CPU ``score_run`` would have written) and prints the whole-tree summary.

Neuron subsets are disjoint across shards (split_neurons guarantees it), so the
concat is a plain row union with no dedup, exactly like cluster/merge_shards.py's
manifest/timing concat.

Usage (from the repo root)
--------------------------
    py -3 -m eval.concat_merge_shards --tree /scratch/$USER/<tree>_merged
    py -3 -m eval.concat_merge_shards --tree <tree> --out <tree>/_merge_metric.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

# Run as a module (py -3 -m eval.concat_merge_shards) or standalone: put the repo
# root on the path so ``eval`` resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from eval import merge_metric as mm

SHARD_GLOB = "_merge_metric.shard_*.csv"


def split_neurons(neurons: Sequence[str], n_shards: int) -> List[List[str]]:
    """Split ``neurons`` into ``n_shards`` balanced, contiguous, disjoint chunks.

    The first ``len(neurons) % n_shards`` chunks get one extra neuron, so sizes
    differ by at most one and every neuron lands in exactly one chunk. With
    ``n_shards == len(neurons)`` this is one neuron per shard (what the 16-neuron
    eval array uses); empty trailing chunks appear only when ``n_shards`` exceeds
    the neuron count, which the array should never request."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    neurons = list(neurons)
    k, r = divmod(len(neurons), n_shards)
    out: List[List[str]] = []
    i = 0
    for s in range(n_shards):
        size = k + (1 if s < r else 0)
        out.append(neurons[i:i + size])
        i += size
    return out


def concat_shard_frames(shard_csvs: Sequence[Path]) -> pd.DataFrame:
    """Read and row-concatenate shard per-frame CSVs into one DataFrame.

    Missing paths are skipped so a single failed shard does not sink the concat
    (its neurons are simply absent from the stitched table, which is visible in
    the printed n_chains). Raises if nothing readable is found."""
    frames = [pd.read_csv(p) for p in shard_csvs if Path(p).exists()]
    if not frames:
        raise SystemExit(f"[concat] no readable shard CSVs among {list(shard_csvs)}")
    return pd.concat(frames, ignore_index=True)


def _as_bool(v):
    """CSV round-trips python bools to the strings 'True'/'False'; map both back."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def summarize_concat(df: pd.DataFrame) -> dict:
    """Whole-tree summary from a stitched per-frame table, coercing the CSV's
    string bools back to real dtypes so mm.summarize sees the same types it does
    in-memory."""
    df = df.copy()
    for col in ("own_contained", "empty"):
        df[col] = df[col].map(_as_bool).astype(bool)
    df["n_foreign"] = pd.to_numeric(df["n_foreign"], errors="coerce").fillna(0).astype(int)
    if "spanning_merge" in df.columns:
        # Keep NaN as NaN (frames with no membrane map); map only the real values.
        df["spanning_merge"] = df["spanning_merge"].map(
            lambda v: _as_bool(v) if pd.notna(v) and v != "" else float("nan"))
        for col in ("boundary_on_membrane", "underfill_fraction", "bled_fraction"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return mm.summarize(df)


def concat_tree(tree: Path, out_csv: Path | None = None,
                shard_glob: str = SHARD_GLOB) -> tuple[Path, dict]:
    """Stitch a tree's shard CSVs into one file; return (out_path, summary)."""
    tree = Path(tree)
    shard_csvs = sorted(tree.glob(shard_glob),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not shard_csvs:
        raise SystemExit(f"[concat] no {shard_glob} shards under {tree}")
    df = concat_shard_frames(shard_csvs)
    dest = Path(out_csv) if out_csv is not None else tree / "_merge_metric.csv"
    df.to_csv(dest, index=False)
    summary = summarize_concat(df)
    print(f"[concat] {len(shard_csvs)} shards -> {dest} ({len(df)} frames)")
    print(mm.format_summary(tree.name, summary))
    return dest, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True,
                    help="the merged tree whose _merge_metric.shard_*.csv to stitch")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <tree>/_merge_metric.csv)")
    ap.add_argument("--shard-glob", default=SHARD_GLOB,
                    help=f"glob for the shard CSVs (default: {SHARD_GLOB})")
    args = ap.parse_args(argv)
    concat_tree(args.tree, args.out, args.shard_glob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
