"""Batch-render before / mask-seed-reprop / box-seed-reprop gif triples for every
chain in a corrected-chains manifest CSV (the same `neuron,chain_idx,anchor_z` format
`find_corrected_chains.py --out-csv` writes and `cluster/run_reprop_corrected_seed.sh`
consumes). Thin loop around `report_assets.py`'s `render_reprop_triple`, one call per
manifest row, writing to `<assets>/<neuron>/chain_NN_{before,mask,box}.gif`.

Idempotent: skips a row if all three output gifs already exist, so re-running after
more chains get reprop'd (or a rendering failure) only fills in the gaps.

    py -3 experiments/render_reprop_report.py \\
        --before "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIAL_AIAR" \\
        --mask-tree "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed_AIA" \\
        --box-tree "F:\\ZhenLab\\Data\\output_masks\\reprop_boxseed_AIA" \\
        --manifest cluster/corrected_chains_AIA.csv \\
        --assets "F:\\ZhenLab\\Data\\repo_offload\\report_assets\\reprop_AIA"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.report_assets import render_reprop_sides


def _read_manifest(path: Path) -> list[tuple[str, int, int]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["neuron"], int(row["chain_idx"]), int(row["anchor_z"])))
    return rows


def render_all(before: Path, mask_tree: Path, box_tree: Optional[Path],
               manifest: Path, assets: Path) -> None:
    """``box_tree`` may be None. A VARIANT=mask cluster run writes no box-seed tree,
    so demanding one would turn that deliberate half-the-GPU-time saving into a
    renderer that produces nothing at all."""
    trees = {"before": before, "mask": mask_tree}
    if box_tree is not None:
        trees["box"] = box_tree
    rows = _read_manifest(manifest)
    print(f"[render-reprop] {len(rows)} chain(s) in {manifest}, "
          f"sides: {', '.join(trees)}")
    skipped, rendered, failed = 0, 0, 0
    for neuron, chain_idx, _anchor_z in rows:
        out_dir = assets / neuron
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = {side: out_dir / f"chain_{chain_idx:02d}_{side}.gif" for side in trees}
        if all(p.exists() for p in outs.values()):
            skipped += 1
            continue
        try:
            render_reprop_sides(trees, neuron, [chain_idx], outs)
            rendered += 1
        except SystemExit as e:
            print(f"  [render-reprop] {neuron} chain_{chain_idx:02d}: skipped, {e}")
            failed += 1
    print(f"[render-reprop] done: {rendered} rendered, {skipped} already present, "
          f"{failed} skipped (no reprop masks)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="pre-reprop working tree")
    ap.add_argument("--mask-tree", required=True, help="mask-seed reprop output tree")
    ap.add_argument("--box-tree",
                    help="box-seed reprop output tree; omit for a VARIANT=mask run")
    ap.add_argument("--manifest", required=True, help="neuron,chain_idx,anchor_z CSV")
    ap.add_argument("--assets", required=True, help="output root for the rendered gifs")
    args = ap.parse_args(argv)

    render_all(Path(args.before), Path(args.mask_tree),
              Path(args.box_tree) if args.box_tree else None,
              Path(args.manifest), Path(args.assets))


if __name__ == "__main__":
    main()
