"""Overnight membrane-pass (mild-bleed) scoring, resumable, one neuron at a time.

`eval.merge_metric.score_run`'s per-chain loop has no error handling and writes
its CSV once at the end: a single unreadable frame (F: has dropped twice tonight
already) kills the whole run and loses everything. This wraps it the same way the
cluster's sharded eval array does (`eval.concat_merge_shards`), scoring ONE neuron
at a time into its own shard CSV, so a crash on neuron 15 does not lose neurons
1-14's already-written shards. Idempotent: a neuron whose shard CSV already exists
is skipped, so re-running this after a crash resumes rather than redoing work.

    py -3 experiments/overnight_membrane_scan.py --root <tree> --neurons AIML,AVER,...
    py -3 experiments/overnight_membrane_scan.py --root <tree> --neurons-file neurons.txt

Stitch the shards into one CSV at any point, complete or not:
    py -3 -m eval.concat_merge_shards --tree <tree> --shard-glob "_merge_metric.membrane_shard_*.csv" --out <tree>/_merge_metric_membrane.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.merge_metric import MembraneSource, load_node_table, run_scale, score_run

SHARD_PREFIX = "_merge_metric.membrane_shard_"
RETRIES = 3
RETRY_DELAY_S = 30


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--neurons", default=None, help="comma-separated neuron dir names")
    ap.add_argument("--neurons-file", default=None, help="one neuron name per line")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if args.neurons:
        neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    elif args.neurons_file:
        neurons = [ln.strip() for ln in Path(args.neurons_file).read_text().splitlines() if ln.strip()]
    else:
        ap.error("pass --neurons or --neurons-file")

    print(f"[overnight] {len(neurons)} neurons queued: {neurons}")
    annotate_df = load_node_table()
    scale = run_scale(root)
    membrane_source = MembraneSource(scale)

    done, failed = [], []
    for i, neuron in enumerate(neurons):
        shard_path = root / f"{SHARD_PREFIX}{neuron}.csv"
        if shard_path.exists():
            print(f"[overnight] ({i+1}/{len(neurons)}) {neuron}: shard already exists, skipping")
            done.append(neuron)
            continue

        print(f"[overnight] ({i+1}/{len(neurons)}) {neuron}: scoring ...")
        t0 = time.time()
        for attempt in range(1, RETRIES + 1):
            try:
                _per, summ = score_run(root, annotate_df=annotate_df, scale=scale,
                                       membrane_source=membrane_source,
                                       neurons=[neuron], out_csv=str(shard_path))
                dt = time.time() - t0
                print(f"[overnight]   {neuron}: done in {dt:.0f}s, "
                      f"{summ.get('n_frames')} frames, mild_bleed_rate={summ.get('mild_bleed_rate')}")
                done.append(neuron)
                break
            except Exception as e:
                print(f"[overnight]   {neuron}: attempt {attempt}/{RETRIES} failed: {e}")
                traceback.print_exc()
                if attempt < RETRIES:
                    print(f"[overnight]   retrying in {RETRY_DELAY_S}s (F: may need to reconnect)...")
                    time.sleep(RETRY_DELAY_S)
                else:
                    print(f"[overnight]   {neuron}: giving up after {RETRIES} attempts, moving on")
                    failed.append(neuron)

    print(f"[overnight] finished: {len(done)} done, {len(failed)} failed")
    if failed:
        print(f"[overnight] failed neurons (re-run this script to retry them): {failed}")


if __name__ == "__main__":
    main()
