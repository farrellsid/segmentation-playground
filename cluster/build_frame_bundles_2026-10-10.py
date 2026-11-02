"""Build proper review bundles (WITH frames) for the 2026-10-10 reprop batch, reusing
the already-backfilled frames (cluster/backfill_frames_2026-10-10.py) instead of
paying export_bundle.py's normal "always regenerate" cost again.

Pre-places each chain's backfilled frames at <dest>/<neuron>/chain_NN/frames/ before
calling export_bundle.py, which (per its own _frames_complete check) skips
regeneration when a complete set is already there, and still does the correct part:
copying masks and rewriting state.json's frames_dir to the bundle-relative "frames"
convention place_frames.py / gui.py expect.

    py -3 cluster/build_frame_bundles_2026-10-10.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

MASKS_SOURCE = Path("E:/ZhenLab/Data/output_masks/reprop_maskseed_2026-10-10")
FRAMES_SOURCE = Path("E:/ZhenLab/Data/output_masks/reprop_frames_2026-10-10")
MANIFEST = Path(__file__).resolve().parent / "corrected_chains_2026-10-10.csv"
OUT_ROOT = Path("C:/Users/User/Downloads/repropagated_2026-10-10")
REPO = Path(__file__).resolve().parent.parent

GROUPS = {
    "ALA": ["ALA"],
    "URB": ["URBL", "URBR"],
    "RID": ["RID"],
    "AVA": ["AVAL", "AVAR"],
}


def main() -> None:
    rows = pd.read_csv(MANIFEST)
    for group, neurons in GROUPS.items():
        dest = OUT_ROOT / f"{group}_bundle"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        group_rows = rows[rows["neuron"].isin(neurons)]
        for _, row in group_rows.iterrows():
            neuron, chain_idx = row["neuron"], int(row["chain_idx"])
            cidx = f"{chain_idx:02d}"
            src_frames = FRAMES_SOURCE / neuron / f"chain_{cidx}" / "frames"
            dst_frames = dest / neuron / f"chain_{cidx}" / "frames"
            dst_frames.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_frames, dst_frames)

        print(f"[build] {group}: {len(group_rows)} chain(s) pre-placed, running export_bundle.py ...")
        cmd = [sys.executable, "export_bundle.py",
              "--output-root", str(MASKS_SOURCE),
              "--neurons", *neurons,
              "--dest", str(dest),
              "--force"]
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"[build] export_bundle.py failed for {group}")
        n_skipped = result.stdout.count("frames already present, skipping")
        n_regen = result.stdout.count("regenerating frames from raw EM")
        print(f"[build] {group}: {n_skipped} chain(s) reused backfilled frames, "
             f"{n_regen} regenerated fresh (should be 0)")


if __name__ == "__main__":
    main()
