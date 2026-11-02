"""concat_merge_shards.py: stitch a sharded merge-metric run into one CSV.

The sharded eval array (cluster/run_eval_array.sh) splits one merged tree across
CPUs: each task runs eval.merge_metric on its own neuron subset and writes
``<tree>/_merge_metric.shard_<i>.csv``. This helper concatenates those shard CSVs
back into the canonical ``<tree>/_merge_metric.csv`` (the same per-frame file a
single-CPU ``score_run`` would have written) and prints the whole-tree summary.

Each task also writes its own shard-scoped z-consistency CSV
(``<tree>/_z_consistency.shard_<i>.csv``, same shard-per-task naming as the per-frame
CSV). When any exist, this helper stitches those too, into a matching
``<tree>/_z_consistency.csv``, and folds the z-consistency summary into the same
printed/returned summary. A tree with no z-consistency shards (scored before this
feature existed, or a tree whose chains produced zero transitions) is handled
gracefully: the z segment is simply omitted, matching how score_run already
degrades when a tree has no z data.

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
Z_SHARD_GLOB = "_z_consistency.shard_*.csv"


def _shard_sort_key(p: Path):
    """Sort shard files by their trailing suffix: numerically when the cluster's
    integer shard index (``shard_<i>.csv``) is there, alphabetically otherwise (a
    neuron-named shard, e.g. ``membrane_shard_SDQR.csv``). Ordering is cosmetic,
    concatenation is a plain row union either way, so this only needs to not crash
    on a non-numeric suffix, not produce a specific order across the two kinds."""
    suffix = p.stem.split("_")[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


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


def concat_shard_z_transitions(shard_csvs: Sequence[Path]) -> pd.DataFrame:
    """Read and row-concatenate shard z-consistency CSVs into one DataFrame.

    Missing paths are skipped, same as concat_shard_frames. Unlike
    concat_shard_frames, an empty result here is not an error: a tree scored before
    this feature existed, or one whose chains produced zero z-transitions, may
    legitimately have no z-consistency shards, so this returns an empty DataFrame
    rather than raising."""
    frames = [pd.read_csv(p) for p in shard_csvs if Path(p).exists()]
    if not frames:
        return pd.DataFrame()
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


def summarize_concat_z(df: pd.DataFrame, *, low_iou_threshold: float = 0.5) -> dict:
    """Whole-tree z-consistency summary from a stitched z-transition table.

    A CSV round-trip turns a Python None (iou/centroid_drift_px of a dropout
    transition) into an empty string or NaN depending on the column's inferred
    dtype; coerce both back to real None before handing records to
    mm.summarize_z_consistency, so it sees the same values score_run would have
    computed in memory."""
    df = df.copy()
    for col in ("iou", "centroid_drift_px"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "gap" in df.columns:
        df["gap"] = pd.to_numeric(df["gap"], errors="coerce").astype(int)
    records = df.to_dict("records")
    for r in records:
        for k in ("iou", "centroid_drift_px"):
            v = r.get(k)
            r[k] = None if pd.isna(v) else float(v)
    return mm.summarize_z_consistency(records, low_iou_threshold=low_iou_threshold)


def concat_tree(tree: Path, out_csv: Path | None = None,
                shard_glob: str = SHARD_GLOB) -> tuple[Path, dict]:
    """Stitch a tree's shard CSVs into one file; return (out_path, summary).

    Also stitches the tree's z-consistency shards (Z_SHARD_GLOB), if any, into a
    matching _z_consistency.csv (path derived from dest the same way score_run
    derives it from out_csv, so --out overrides apply to both files consistently),
    and folds summarize_concat_z's output into the returned/printed summary. A tree
    with no z-consistency shards skips this step entirely: the summary and printed
    line simply omit the z segment, same as a tree scored with no z data."""
    tree = Path(tree)
    shard_csvs = sorted(tree.glob(shard_glob), key=_shard_sort_key)
    if not shard_csvs:
        raise SystemExit(f"[concat] no {shard_glob} shards under {tree}")
    df = concat_shard_frames(shard_csvs)
    dest = Path(out_csv) if out_csv is not None else tree / "_merge_metric.csv"
    df.to_csv(dest, index=False)
    summary = summarize_concat(df)

    z_shard_csvs = sorted(tree.glob(Z_SHARD_GLOB), key=_shard_sort_key)
    if z_shard_csvs:
        z_df = concat_shard_z_transitions(z_shard_csvs)
        if len(z_df):
            z_dest = mm.z_csv_path_for(dest, tree)
            z_df.to_csv(z_dest, index=False)
            summary.update(summarize_concat_z(z_df))

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
