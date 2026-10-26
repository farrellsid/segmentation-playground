"""Re-propagate a chain from its manually corrected ANCHOR mask, in two variants for
comparison: MASK-seed (the corrected mask fed directly as the conditioning mask) and
BOX-seed (a box derived from the corrected mask, fed alongside the chain's original
point prompts). Both reuse `pipeline.propagate.propagate()` unchanged, unlike
`propagate_from_verified.py` this needs no new pipeline code, "only fix the seed" is
exactly the single-anchor case `propagate()` already handles, this script only builds
the two flavours of seed and drives it.

Handles BOTH legacy full-frame `_sam` chains and tier-2 per-chain `_pcrop` crop chains,
matching whichever space the chain's own `state.json` (`crop_window`) already uses,
same real gap `gui.py`'s `_ensure_local_frames` was fixed for: a real chain from this
dataset (AIYL chain_00) turned out to be tier-2, not the legacy-only case this script
originally assumed. For a tier-2 chain, the saved mask PNG is already sized to the full
`_pcrop` crop window (no offset math needed) and `state.json`'s `prompts` are already
in that same native `_pcrop` space despite the "_sam" field naming, confirmed from
`pipeline/orchestrator.py`'s own comment on tier-2 seeding; reading the mask directly
(not through `chain_masks_in_sam`'s `_sam`-downscaling remap, which is for cross-chain
aggregation, not for feeding a mask back into a predictor) and reusing `state.prompts`
as-is is therefore correct for both spaces, with no cross-space conversion needed here.

Use `experiments/find_corrected_chains.py` first to find which chains actually have a
corrected anchor mask; running this on an unchanged chain reproduces the original
output (a real, if uninteresting, one-shot verification that nothing else drifted).

Writes each variant to its own SEPARATE output tree; the working (corrected) tree is
only ever read.

    py -3 experiments/propagate_from_corrected_seed.py \\
        --working "F:\\ZhenLab\\Data\\output_masks\\manual_verify_AIYL_AIYR" \\
        --neuron AIYL --chain 0 \\
        --out-mask "F:\\ZhenLab\\Data\\output_masks\\reprop_maskseed_AIYL_AIYR" \\
        --out-box  "F:\\ZhenLab\\Data\\output_masks\\reprop_boxseed_AIYL_AIYR" \\
        --out-frames "F:\\ZhenLab\\Data\\output_masks\\reprop_frames_AIYL_AIYR"

Batch mode (one Slurm task, many chains, ONE predictor build shared across all of
them, see docs/how-to/repropagation-round-protocol.md for why this exists: a round
of 286 single-chain tasks measured ~1-3 min of real propagation work each, but the
SAM2 checkpoint load at the top of every task was eating a large, unmeasured share
of that, paid 286 times over instead of once per batch). `--working` is the ONE
staged tree all chains in the manifest share (the manifest's own neuron column
disambiguates per chain, same flat <neuron>/chain_NN layout single-chain mode uses):

    py -3 experiments/propagate_from_corrected_seed.py \\
        --working /scratch/fsid/manual_verify_2026-10-25_batch \\
        --manifest cluster/corrected_chains_2026-10-25.csv \\
        --row-start 0 --row-count 15 \\
        --out-mask ... --out-box ... --out-frames ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.crop import prepare_chain_crop_frames, prepare_video_frames
from pipeline.masks import save_masks
from pipeline.predict import box_from_mask
from pipeline.propagate import propagate
from pipeline.state import ChainState, Prompts, load_state, save_state
from sam2_utils import alignment, config, qc, setup

SCALE = 8


def _anchor_mask_native(working: Path, neuron: str, chain_idx: int, anchor_z: int):
    """The corrected anchor mask exactly as saved, at NATIVE resolution: the full
    `_pcrop` crop window for a tier-2 chain, or the full `_sam` frame for a legacy
    chain. Not remapped to `_sam` (chain_masks_in_sam's job is aggregation across
    chains onto one shared grid; this needs the mask back in whatever space the
    fresh propagation itself will run in)."""
    mask_path = working / neuron / f"chain_{chain_idx:02d}" / "masks" / f"mask_{anchor_z:04d}.png"
    if not mask_path.exists():
        raise SystemExit(f"[reprop] no mask at {mask_path}")
    return qc._load_binary(mask_path)


def _save_variant(video_segments, frame_to_z, *, out_tree: Path, neuron: str, chain_idx: int,
                  src_state: ChainState, anchor_frame_idx: int, n_frames: int,
                  frames_dir: str, crop_window: dict | None) -> None:
    out_chain_dir = out_tree / neuron / f"chain_{chain_idx:02d}"
    n_written = save_masks(video_segments, frame_to_z, out_chain_dir / "masks",
                           obj_id=1, mask_space_downscale=SCALE)
    state = ChainState(
        neuron=neuron, chain_idx=chain_idx, status="done",
        anchor_node_id=src_state.anchor_node_id, anchor_catmaid_z=src_state.anchor_catmaid_z,
        anchor_frame_idx=anchor_frame_idx, frames_dir=str(frames_dir),
        frame_to_z=frame_to_z, n_frames=n_frames, crop_window=crop_window,
    )
    save_state(state, out_chain_dir / "state.json")
    print(f"[reprop]   wrote {n_written} masks + state.json to {out_chain_dir}")


def _propagate_one_chain(*, working: Path, neuron: str, chain_idx: int,
                         annotate_df: pd.DataFrame, all_chains: list,
                         video_predictor, args) -> None:
    """Everything main() used to do for a single chain, minus building the
    predictor (the caller already built it, once, and passes it in here so a
    batch of chains shares the same predictor instead of each paying the
    checkpoint-load cost separately)."""
    src_chain_dir = working / neuron / f"chain_{chain_idx:02d}"
    src_state = load_state(src_chain_dir / "state.json")
    if src_state.anchor_catmaid_z is None or src_state.prompts is None:
        raise SystemExit(f"[reprop] {src_chain_dir}/state.json is missing anchor_catmaid_z or prompts")
    anchor_z = src_state.anchor_catmaid_z
    cw = alignment.CropWindow.from_dict(src_state.crop_window) if src_state.crop_window else None
    print(f"[reprop] {neuron} chain_{chain_idx:02d}, anchor z={anchor_z}, "
         f"space={'_pcrop tier-2 (crop_scale=' + str(cw.crop_scale) + ')' if cw else '_sam legacy'}")

    cell_chains = [c for c in all_chains if c["cell_name"] == neuron]
    if chain_idx >= len(cell_chains):
        raise SystemExit(f"[reprop] {neuron} has only {len(cell_chains)} chains in chains.json, "
                         f"asked for chain_idx={chain_idx}")
    chain = cell_chains[chain_idx]

    print(f"[reprop] preparing fresh {'_pcrop' if cw else '_sam'} video frames ...")
    if cw is not None:
        frames_dir, frame_to_z, anchor_frame_idx, n_frames = prepare_chain_crop_frames(
            chain, annotate_df, cw, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
    else:
        frames_dir, frame_to_z, anchor_frame_idx, n_frames = prepare_video_frames(
            chain, annotate_df, scale=SCALE, frames_root=config.FRAMES_ROOT,
            anchor_catmaid_z=anchor_z, neuron=neuron, chain_idx=chain_idx)
    print(f"[reprop] {n_frames} frames prepared, anchor at frame {anchor_frame_idx}")

    if args.out_frames:
        persist_dir = Path(args.out_frames) / neuron / f"chain_{chain_idx:02d}" / "frames"
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frames_dir, persist_dir)
        frames_dir = str(persist_dir)
        print(f"[reprop] frames persisted to {frames_dir}")

    corrected_mask = _anchor_mask_native(working, neuron, chain_idx, anchor_z)
    print(f"[reprop] corrected anchor mask area={int(corrected_mask.sum())}, "
         f"native shape={corrected_mask.shape}")

    crop_window_dict = src_state.crop_window  # already a plain dict, reuse verbatim for outputs

    if not args.skip_mask_variant:
        print("[reprop] MASK-seed variant: propagating ...")
        video_segments, _frame_conf, _pred_iou = propagate(
            video_predictor, frames_dir, src_state.prompts, anchor_frame_idx,
            obj_id=1, seed_mask=True, mask_anchor=corrected_mask)
        _save_variant(video_segments, frame_to_z, out_tree=Path(args.out_mask),
                     neuron=neuron, chain_idx=chain_idx, src_state=src_state,
                     anchor_frame_idx=anchor_frame_idx, n_frames=n_frames,
                     frames_dir=frames_dir, crop_window=crop_window_dict)

    if not args.skip_box_variant:
        box = box_from_mask(corrected_mask, margin=args.box_margin,
                            image_hw_sam=corrected_mask.shape)
        if box is None:
            print("[reprop] WARNING: corrected mask is empty, cannot derive a box, "
                 "skipping box-seed variant")
        else:
            print(f"[reprop] BOX-seed variant: box={box.tolist()}, propagating ...")
            box_prompts = Prompts(points_sam=src_state.prompts.points_sam,
                                  labels=src_state.prompts.labels, box_sam=box)
            video_segments, _frame_conf, _pred_iou = propagate(
                video_predictor, frames_dir, box_prompts, anchor_frame_idx, obj_id=1,
                seed_box=True, seed_points=True, seed_negatives=args.seed_negatives)
            _save_variant(video_segments, frame_to_z, out_tree=Path(args.out_box),
                         neuron=neuron, chain_idx=chain_idx, src_state=src_state,
                         anchor_frame_idx=anchor_frame_idx, n_frames=n_frames,
                         frames_dir=frames_dir, crop_window=crop_window_dict)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--working", required=True, help="tree with the corrected anchor mask(s)")
    ap.add_argument("--neuron", help="single-chain mode: chain's neuron")
    ap.add_argument("--chain", type=int, help="single-chain mode: chain_idx")
    ap.add_argument("--manifest", help="batch mode: CSV with neuron,chain_idx columns "
                                       "(find_corrected_chains.py's --out-csv)")
    ap.add_argument("--row-start", type=int, default=0,
                    help="batch mode: first manifest data row this task owns (0-based, "
                         "after the header)")
    ap.add_argument("--row-count", type=int, default=0,
                    help="batch mode: how many consecutive manifest rows this task owns")
    ap.add_argument("--out-mask", required=True, help="output tree for the mask-seed variant")
    ap.add_argument("--out-box", required=True, help="output tree for the box-seed variant")
    ap.add_argument("--out-frames", default=None,
                    help="persist each chain's prepared frames here (<out-frames>/<neuron>/"
                         "chain_NN/frames), and point both variants' state.json at that copy "
                         "instead of the ephemeral prep location. On Narval, frames prepare "
                         "into $SLURM_TMPDIR, which is node-local and deleted the moment the "
                         "job ends; without --out-frames a chain's frames exist only for the "
                         "lifetime of this one task. Omit to keep the old ephemeral-only "
                         "behaviour (masks only, no frames to ship).")
    ap.add_argument("--box-margin", type=int, default=10)
    ap.add_argument("--seed-negatives", action="store_true",
                    help="include the chain's original negative points in the box-seed variant")
    ap.add_argument("--model-size", default="large")
    ap.add_argument("--skip-mask-variant", action="store_true")
    ap.add_argument("--skip-box-variant", action="store_true")
    args = ap.parse_args(argv)

    batch_mode = args.manifest is not None
    if batch_mode and (args.neuron or args.chain is not None):
        raise SystemExit("[reprop] --manifest and --neuron/--chain are mutually exclusive")
    if not batch_mode and (args.neuron is None or args.chain is None):
        raise SystemExit("[reprop] single-chain mode needs both --neuron and --chain "
                         "(or use --manifest for batch mode)")

    working = Path(args.working)

    print("[reprop] loading shared CATMAID tables + chains.json once ...")
    annotate_df = pd.read_csv(config.CSV_PATH)
    xy = alignment.catmaid_to_tif(annotate_df["x"].values, annotate_df["y"].values)
    annotate_df["x_tif"], annotate_df["y_tif"] = xy[:, 0], xy[:, 1]
    with open(config.CHAINS_PATH) as f:
        all_chains = json.load(f)

    if batch_mode:
        manifest = pd.read_csv(args.manifest)
        if args.row_count <= 0:
            raise SystemExit("[reprop] --row-count must be > 0 in batch mode")
        rows = manifest.iloc[args.row_start:args.row_start + args.row_count]
        if len(rows) == 0:
            raise SystemExit(f"[reprop] no manifest rows in [{args.row_start}, "
                             f"{args.row_start + args.row_count}), manifest has {len(manifest)} rows")
        pairs = [(r["neuron"], int(r["chain_idx"])) for _, r in rows.iterrows()]
    else:
        pairs = [(args.neuron, args.chain)]

    print(f"[reprop] building {args.model_size} video predictor once for "
         f"{len(pairs)} chain(s) ...")
    video_predictor, _ = setup.build_predictor(size=args.model_size, kind="video")

    for neuron, chain_idx in pairs:
        _propagate_one_chain(working=working, neuron=neuron, chain_idx=chain_idx,
                             annotate_df=annotate_df, all_chains=all_chains,
                             video_predictor=video_predictor, args=args)

    print(f"[reprop] done, {len(pairs)} chain(s). Note: QC was not run on either output "
         f"(no qc.csv). Score with eval.merge_metric like any other tree.")


if __name__ == "__main__":
    main()
