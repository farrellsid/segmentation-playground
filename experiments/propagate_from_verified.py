"""Run video-mode propagation seeded from MANUALLY VERIFIED masks at multiple frames,
not a single point-prompt anchor: the experimental counterpart to the normal pipeline,
built to test whether propagation does much better when every conditioning mask is
human-confirmed instead of automatic.

Reads the CURRENT on-disk masks at the given CATMAID z's from an already-corrected
working tree (e.g. after using `gui.py --output-root <tree> --neuron AIYL` to fix
frames), so correct frames in that tree BEFORE running this. Writes the propagated
result to a SEPARATE output tree, the source tree of verified masks is never modified.

Handles BOTH legacy full-frame `_sam` chains and tier-2 per-chain `_pcrop` crop chains,
matching whichever space the chain's own `state.json` (`crop_window`) already uses.
An earlier version of this script assumed every chain was legacy `_sam`; a real chain
in this dataset (AIYL chain_00) turned out to be tier-2, the same gap `gui.py`'s
`_ensure_local_frames` was fixed for. Each verified mask is read directly at its NATIVE
resolution (the saved PNG as-is, no `_sam`-downscaling remap, `chain_masks_in_sam`'s
remap is for cross-chain aggregation onto one shared grid, not for feeding a mask back
into a predictor), which for a tier-2 chain is already the full `_pcrop` crop window
with no offset math needed, confirmed from `pipeline/orchestrator.py`'s own comment
that tier-2 propagation stays entirely in `_pcrop` space.

Does NOT run QC scoring (run_qc) or write qc.csv; score the output tree afterward with
the normal tools (eval.merge_metric, gui.py for a visual look) like any other tree.

    py -3 experiments/propagate_from_verified.py \\
        --tree "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIYL_AIYR" \\
        --neuron AIYL --chain 0 --verified-z 1590,1608,1625 \\
        --out "F:\\ZhenLab\\Data\\output_masks\\propagated_from_verified_AIYL_AIYR"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.crop import prepare_chain_crop_frames, prepare_video_frames
from pipeline.masks import save_masks
from pipeline.propagate import propagate_from_verified_masks
from pipeline.state import ChainState, save_state
from sam2_utils import alignment, config, qc, setup

SCALE = 8


def _mask_native(chain_dir: Path, z: int):
    """A verified frame's mask exactly as saved, at native resolution (see module
    docstring for why this is not the chain_masks_in_sam _sam-remapped version)."""
    p = chain_dir / "masks" / f"mask_{z:04d}.png"
    if not p.exists():
        return None
    return qc._load_binary(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="working tree with the verified masks")
    ap.add_argument("--neuron", required=True)
    ap.add_argument("--chain", type=int, required=True, help="chain_idx")
    ap.add_argument("--verified-z", required=True,
                    help="comma-separated CATMAID z's to seed as conditioning frames")
    ap.add_argument("--out", required=True, help="output tree root (must not equal --tree)")
    ap.add_argument("--model-size", default="large")
    args = ap.parse_args(argv)

    src_tree = Path(args.tree)
    out_tree = Path(args.out)
    if src_tree.resolve() == out_tree.resolve():
        ap.error("--out must be a different tree from --tree; this never edits the source")
    verified_z = [int(z) for z in args.verified_z.split(",") if z.strip()]
    if not verified_z:
        ap.error("--verified-z must list at least one z")

    neuron, chain_idx = args.neuron, args.chain
    src_chain_dir = src_tree / neuron / f"chain_{chain_idx:02d}"
    print(f"[propfv] reading verified masks from {src_chain_dir}")

    with open(src_chain_dir / "state.json") as f:
        src_state = json.load(f)
    anchor_catmaid_z = src_state.get("anchor_catmaid_z")
    anchor_node_id = src_state.get("anchor_node_id")
    if anchor_catmaid_z is None:
        raise SystemExit(f"[propfv] {src_chain_dir}/state.json has no anchor_catmaid_z")
    crop_window_dict = src_state.get("crop_window")
    cw = alignment.CropWindow.from_dict(crop_window_dict) if crop_window_dict else None
    print(f"[propfv] {neuron} chain_{chain_idx:02d}, "
         f"space={'_pcrop tier-2 (crop_scale=' + str(cw.crop_scale) + ')' if cw else '_sam legacy'}")

    masks_native = {}
    for z in verified_z:
        m = _mask_native(src_chain_dir, z)
        if m is None:
            raise SystemExit(f"[propfv] no mask at z={z} in {src_chain_dir}")
        masks_native[z] = m

    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)
    cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
    if chain_idx >= len(cell_chains):
        raise SystemExit(f"[propfv] {neuron} has only {len(cell_chains)} chains in chains.json, "
                         f"asked for chain_idx={chain_idx}")
    chain = cell_chains[chain_idx]

    print(f"[propfv] preparing fresh {'_pcrop' if cw else '_sam'} video frames for "
         f"{neuron} chain_{chain_idx:02d} (anchor z={anchor_catmaid_z}) ...")
    if cw is not None:
        frames_dir, frame_to_z, anchor_frame_idx, n_frames = prepare_chain_crop_frames(
            chain, annotate_df, cw, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=anchor_catmaid_z, neuron=neuron, chain_idx=chain_idx)
    else:
        frames_dir, frame_to_z, anchor_frame_idx, n_frames = prepare_video_frames(
            chain, annotate_df, scale=SCALE, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=anchor_catmaid_z, neuron=neuron, chain_idx=chain_idx)
    z_to_frame = {z: f for f, z in frame_to_z.items()}
    print(f"[propfv] {n_frames} frames prepared, view at {frames_dir}")

    missing_frame = [z for z in verified_z if z not in z_to_frame]
    if missing_frame:
        raise SystemExit(f"[propfv] these verified z's fall outside this chain's z-range: "
                         f"{missing_frame}")

    masks_by_frame = {}
    for z in verified_z:
        masks_by_frame[z_to_frame[z]] = masks_native[z]
        print(f"[propfv]   seeding z={z} (frame {z_to_frame[z]}), "
             f"mask area={int(masks_native[z].sum())}")

    print(f"[propfv] building {args.model_size} video predictor ...")
    video_predictor, _ = setup.build_predictor(size=args.model_size, kind="video")

    print(f"[propfv] propagating from {len(masks_by_frame)} verified frames ...")
    video_segments, _frame_conf, _pred_iou = propagate_from_verified_masks(
        video_predictor, frames_dir, masks_by_frame, obj_id=1)
    print(f"[propfv] propagation done, {len(video_segments)} frames produced")

    out_chain_dir = out_tree / neuron / f"chain_{chain_idx:02d}"
    n_written = save_masks(video_segments, frame_to_z, out_chain_dir / "masks",
                           obj_id=1, mask_space_downscale=SCALE)
    print(f"[propfv] wrote {n_written} mask PNGs to {out_chain_dir / 'masks'}")

    state = ChainState(
        neuron=neuron, chain_idx=chain_idx, status="done",
        anchor_node_id=anchor_node_id, anchor_catmaid_z=anchor_catmaid_z,
        anchor_frame_idx=anchor_frame_idx,
        frames_dir=str(frames_dir), frame_to_z=frame_to_z, n_frames=n_frames,
        crop_window=crop_window_dict,
    )
    save_state(state, out_chain_dir / "state.json")
    print(f"[propfv] wrote {out_chain_dir / 'state.json'}")
    print(f"[propfv] verified frames used as conditioning: "
         f"{sorted(verified_z)} (source: {src_chain_dir})")
    print("[propfv] note: QC was not run on this output (no qc.csv). Score it with "
         "eval.merge_metric like any other tree, or open it in gui.py for a look.")


if __name__ == "__main__":
    main()
