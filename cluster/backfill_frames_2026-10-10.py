"""Backfill persisted frames for the 2026-10-10 reprop batch (ALA, URB, RID, AVA),
locally, without re-running propagation.

The masks are already correct (produced by the real Narval reprop job); this script
only redoes the CPU-only crop step (prepare_chain_crop_frames / prepare_video_frames)
using the SAME crop_window and anchor_catmaid_z already recorded in each chain's
state.json, and persists the result so it can be zipped and sent to Lucinda without
her machine paying the crop cost per chain on open (see
docs/how-to/repropagation-round-protocol.md).

    py -3 cluster/backfill_frames_2026-10-10.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.crop import prepare_chain_crop_frames, prepare_video_frames
from pipeline.state import load_state
from sam2_utils import alignment, config

SCALE = 8
SOURCE = Path("E:/ZhenLab/Data/output_masks/reprop_maskseed_2026-10-10")
OUT_FRAMES = Path("E:/ZhenLab/Data/output_masks/reprop_frames_2026-10-10")
MANIFEST = Path(__file__).resolve().parent / "corrected_chains_2026-10-10.csv"


def main() -> None:
    rows = pd.read_csv(MANIFEST)
    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)

    t_start = time.time()
    total_frames = 0
    for i, row in rows.iterrows():
        neuron, chain_idx = row["neuron"], int(row["chain_idx"])
        chain_dir = SOURCE / neuron / f"chain_{chain_idx:02d}"
        state = load_state(chain_dir / "state.json")
        anchor_z = state.anchor_catmaid_z
        cw = alignment.CropWindow.from_dict(state.crop_window) if state.crop_window else None

        cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
        chain = cell_chains[chain_idx]

        t0 = time.time()
        if cw is not None:
            frames_dir, _frame_to_z, _anchor_idx, n_frames = prepare_chain_crop_frames(
                chain, annotate_df, cw, frames_root=config.FRAMES_ROOT,
                anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
        else:
            frames_dir, _frame_to_z, _anchor_idx, n_frames = prepare_video_frames(
                chain, annotate_df, scale=SCALE, frames_root=config.FRAMES_ROOT,
                anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
        dt = time.time() - t0

        persist_dir = OUT_FRAMES / neuron / f"chain_{chain_idx:02d}" / "frames"
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frames_dir, persist_dir)

        total_frames += n_frames
        print(f"[backfill] {i + 1}/{len(rows)} {neuron} chain_{chain_idx:02d}: "
             f"{n_frames} frame(s) in {dt:.1f}s -> {persist_dir}")

    elapsed = time.time() - t_start
    print(f"[backfill] done: {total_frames} frame(s) across {len(rows)} chain(s) "
         f"in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
