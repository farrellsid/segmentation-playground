"""Create a review-ready working copy of specific neurons from a source tree, for
manual GUI correction (gui.py). Copies each neuron's chain directories plus the
tree-root files the review GUI actually needs, filtered to just the requested
neurons so the GUI's chain queue is populated correctly instead of opening empty.
The source tree is never modified. `gui.py --source` calls this automatically now
(see its module docstring), so a manual run is only needed to force a specific
neuron back to a pristine copy.

Re-running for the SAME neuron replaces its chain directory fresh (a clean re-copy
from source, not a merge, so it always matches what is actually on the source tree,
this is the "force pristine" behaviour). Re-running for a DIFFERENT neuron does NOT
touch any other neuron already in `--out`: `_manifest.csv`/`_triage.csv` are now
MERGED (this neuron's rows replaced, every other neuron's rows kept), not
overwritten wholesale, a real bug an earlier version of this script had, found
when `gui.py --source` started calling it per-neuron on demand, which needs calling
it repeatedly for the SAME `--out` tree to actually accumulate instead of each call
erasing the last one's neurons from the manifest.

Real gap this fixes (original version): a first version of this working-copy setup
only copied the neuron directories and `_run_meta.json`. `sam2_utils.review_queue.
ReviewQueue` (what `gui.py` uses to build its chain list) reads `_manifest.csv` and
`_triage.csv` from the tree root; without them the GUI opens with an empty queue,
not an error, so the gap was silent until someone actually tried to use it.

    py -3 experiments/make_review_tree.py --source <source tree> \\
        --neurons AIYL,AIYR --out <new working tree>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# Tree-root files the review GUI reads. _manifest.csv / _triage.csv are the real
# blocker (ReviewQueue); _run_meta.json carries the run's scale/preset for tools
# that read it (dense_video.py, this repo's own scripts). _labels.csv / _review.csv
# are review-STATE accumulators (what a human already decided), deliberately not
# copied: a fresh working copy should start with a clean review session, not
# inherit disposition rows from whatever review already happened on the source.
NEURON_FILTERED_CSVS = ("_manifest.csv", "_triage.csv")
COPY_VERBATIM = ("_run_meta.json",)


def make_review_tree(source: Path, neurons: list[str], out: Path) -> None:
    if source.resolve() == out.resolve():
        raise SystemExit("[review-tree] --out must differ from --source")
    out.mkdir(parents=True, exist_ok=True)

    for neuron in neurons:
        src_dir = source / neuron
        if not src_dir.exists():
            print(f"[review-tree] WARNING: {src_dir} does not exist, skipping {neuron}")
            continue
        dst_dir = out / neuron
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        n_chains = sum(1 for p in dst_dir.glob("chain_*") if p.is_dir())
        print(f"[review-tree] copied {neuron}: {n_chains} chains")

    for name in NEURON_FILTERED_CSVS:
        src_csv = source / name
        if not src_csv.exists():
            print(f"[review-tree] WARNING: {src_csv} not found, GUI queue for this tree "
                 f"will be incomplete without it")
            continue
        df = pd.read_csv(src_csv)
        fresh = df[df["neuron"].isin(neurons)]
        dst_csv = out / name
        if dst_csv.exists():
            existing = pd.read_csv(dst_csv)
            kept = existing[~existing["neuron"].isin(neurons)]   # everyone else, untouched
            merged = pd.concat([kept, fresh], ignore_index=True)
        else:
            merged = fresh
        merged.to_csv(dst_csv, index=False)
        print(f"[review-tree] wrote {name}: {len(merged)} rows total "
             f"({len(fresh)} for {neurons}, {len(merged) - len(fresh)} kept from other neurons)")

    for name in COPY_VERBATIM:
        src_file = source / name
        if src_file.exists():
            shutil.copy2(src_file, out / name)
            print(f"[review-tree] copied {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--neurons", required=True, help="comma-separated neuron names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    make_review_tree(Path(args.source), neurons, Path(args.out))
    print(f"[review-tree] done: py -3 gui.py --output-root \"{args.out}\" --neuron {neurons[0]}")


if __name__ == "__main__":
    main()
