"""Find which chains have a manually corrected ANCHOR mask, by comparing a working
(review) tree against its untouched source tree. Purely data-driven, a pixel diff on
the anchor frame's mask, not a GUI review-state lookup.

This session traced the GUI's review-state tracking (`_manifest.csv`'s status,
`sam2_utils/labels.py`'s per-frame role) before landing here: `resume_propagation`
(the only action that marks a chain `"corrected"`) runs its own local
video-propagation before saving, so by the time a chain shows that status its whole
tail has already changed, not just the one frame a human painted, there is no clean
record of "this single frame was hand-edited" to recover from disk state alone. For
the "only fix the seed" workflow this project settled on, the anchor mask is the
only frame that should ever change, so a direct pixel comparison of the anchor mask,
before vs after, is both simpler and more robust than any status flag.

    py -3 experiments/find_corrected_chains.py --source <original tree> --working <working tree> --neuron AIYL
    py -3 experiments/find_corrected_chains.py --source <original tree> --working <working tree> --neurons AIYL,AIYR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import pipeline

SCALE = 8


def _mask_at_z_full(tree: Path, neuron: str, chain_idx: int, z: int,
                    full_hw: tuple[int, int]) -> np.ndarray:
    """This chain's mask at z, placed into a full _sam canvas. All-False if the
    chain or that z's mask does not exist in this tree."""
    cdir = tree / neuron / f"chain_{chain_idx:02d}"
    full = np.zeros(full_hw, dtype=bool)
    if not cdir.exists():
        return full
    masks = pipeline.chain_masks_in_sam(cdir)
    if z not in masks:
        return full
    mask, x0, y0 = masks[z]
    h, w = mask.shape
    full[y0:y0 + h, x0:x0 + w] = mask
    return full


def find_corrected_chains(source: Path, working: Path, neurons: list[str]) -> list[tuple[str, int, int]]:
    """[(neuron, chain_idx, anchor_z), ...] for every chain whose anchor mask differs
    between `source` and `working`. The anchor z always comes from the WORKING
    tree's own state.json (the one being reviewed); a chain missing there is not
    considered, there is nothing to compare or propagate."""
    corrected = []
    for neuron in neurons:
        ndir = working / neuron
        if not ndir.exists():
            print(f"[find-corrected] {ndir} does not exist, skipping {neuron}")
            continue
        chain_idxs = sorted(int(p.name.split("_")[-1]) for p in ndir.glob("chain_*") if p.is_dir())
        for ci in chain_idxs:
            sj = ndir / f"chain_{ci:02d}" / "state.json"
            if not sj.exists():
                continue
            anchor_z = json.loads(sj.read_text()).get("anchor_catmaid_z")
            if anchor_z is None:
                continue
            _em, full_hw = pipeline.load_frame_sam(int(anchor_z), scale=SCALE)
            work_mask = _mask_at_z_full(working, neuron, ci, anchor_z, full_hw)
            src_mask = _mask_at_z_full(source, neuron, ci, anchor_z, full_hw)
            if not np.array_equal(work_mask, src_mask):
                corrected.append((neuron, ci, anchor_z))
    return corrected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--working", required=True)
    ap.add_argument("--neurons", required=True, help="comma-separated neuron names")
    ap.add_argument("--out-csv", default=None, help="also write neuron,chain_idx,anchor_z rows here")
    args = ap.parse_args(argv)

    neurons = [n.strip() for n in args.neurons.split(",") if n.strip()]
    corrected = find_corrected_chains(Path(args.source), Path(args.working), neurons)
    print(f"[find-corrected] {len(corrected)} chain(s) with a corrected anchor mask:")
    for neuron, ci, anchor_z in corrected:
        print(f"  {neuron} chain_{ci:02d}  anchor z={anchor_z}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["neuron", "chain_idx", "anchor_z"])
            w.writerows(corrected)
        print(f"[find-corrected] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
