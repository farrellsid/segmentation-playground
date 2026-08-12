"""Report assets for a hand-corrected propagation neuron: before/after chain renders,
a before/after whole-neuron merged render, and a before/after merge-metric comparison.
Fills the parts of the "Pipeline Report" template that are not the human's own
correction-time notes or start-frame screenshots: the "Improvements (According to
Metrics)" section, the "[Merged Render, before and after]" section, and the per-chain
"Before (Render)" / "After (Render)" table cells.

Headless, no napari: reuses `sam2_utils.video_viz.to_gif`/`to_mp4` (already
GUI-independent, gui_neuron.py's own "export overlay" button calls the same two
functions) instead of going through the neuron-review GUI's in-memory canvas. Builds
its own small frames_dir + segments dict directly from a tree's saved masks
(`pipeline.chain_masks_in_sam`) and the raw EM (`pipeline.load_frame_sam`), so it works
on ANY tree (a per-slice "before" tree or a `propagate_from_verified.py` "after" tree)
without needing that tree to have ever been opened in a GUI.

    # whole-neuron merged render, before vs after
    py -3 experiments/report_assets.py neuron-gif --tree <before-tree> --neuron AIYL --out before.gif
    py -3 experiments/report_assets.py neuron-gif --tree <after-tree>  --neuron AIYL --out after.gif

    # one chain, before vs after
    py -3 experiments/report_assets.py chain-gif --tree <before-tree> --neuron AIYL --chain 0 --out before_c0.gif
    py -3 experiments/report_assets.py chain-gif --tree <after-tree>  --neuron AIYL --chain 0 --out after_c0.gif

    # whole-neuron before/after metrics table
    py -3 experiments/report_assets.py metrics --before <before-tree> --after <after-tree> --neuron AIYL
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import pipeline
from sam2_utils import video_viz

SCALE = 8
TMP_ROOT = Path("scratch_render")


def _neuron_chain_idxs(tree: Path, neuron: str) -> list[int]:
    ndir = tree / neuron
    if not ndir.exists():
        raise SystemExit(f"[report] {ndir} does not exist")
    return sorted(int(p.name.split("_")[-1]) for p in ndir.glob("chain_*") if p.is_dir())


def build_view(tree: Path, neuron: str, chain_idxs: list[int], tmp_dir: Path
              ) -> tuple[Path, dict, dict]:
    """A 0-indexed JPEG sequence + {frame_idx: {chain_idx+1: mask}} over the UNION
    z-range of the given chains in `tree`, cropped to a FIXED window covering every
    mask across every z (padded), not the whole raw frame: a small chain's mask is
    imperceptible against a full worm cross-section, the same "invisible mask at
    full-frame scale" problem this session already hit with a micro_sam spot-check
    and fixed the same way there. Each chain gets a stable colour (chain_idx + 1, so
    a single-chain render and a whole-neuron render use the same colour for the same
    chain). Returns (frames_dir, segments, frame_to_z)."""
    PAD = 40
    chains_masks = {}
    all_z: set[int] = set()
    for ci in chain_idxs:
        cdir = tree / neuron / f"chain_{ci:02d}"
        if not cdir.exists():
            continue
        masks = pipeline.chain_masks_in_sam(cdir)
        if masks:
            chains_masks[ci] = masks
            all_z |= set(masks.keys())
    if not all_z:
        raise SystemExit(f"[report] no masks for {neuron} chains {chain_idxs} in {tree}")

    zs = sorted(all_z)
    frame_to_z = {i: z for i, z in enumerate(zs)}
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    _em0, full_hw = pipeline.load_frame_sam(zs[0], scale=SCALE)
    H, W = full_hw

    # pass 1: the fixed window, union of every mask's extent across every z
    xs0, ys0, xs1, ys1 = [], [], [], []
    for masks in chains_masks.values():
        for mask, x0, y0 in masks.values():
            if not mask.any():
                continue
            h, w = mask.shape
            xs0.append(x0); ys0.append(y0); xs1.append(x0 + w); ys1.append(y0 + h)
    if not xs0:
        raise SystemExit(f"[report] every mask for {neuron} chains {chain_idxs} in {tree} is empty")
    wx0, wy0 = max(0, min(xs0) - PAD), max(0, min(ys0) - PAD)
    wx1, wy1 = min(W, max(xs1) + PAD), min(H, max(ys1) + PAD)

    # pass 2: crop EM + masks to that fixed window for every z
    segments: dict[int, dict[int, np.ndarray]] = {}
    for i, z in enumerate(zs):
        em, _ = pipeline.load_frame_sam(z, scale=SCALE)
        em_rgb = em if em.ndim == 3 else np.stack([em] * 3, axis=-1)
        em_win = em_rgb[wy0:wy1, wx0:wx1]
        cv2.imwrite(str(tmp_dir / f"{i:05d}.jpg"),
                   cv2.cvtColor(em_win.astype(np.uint8), cv2.COLOR_RGB2BGR))
        frame_objs = {}
        for ci, masks in chains_masks.items():
            if z not in masks:
                continue
            mask, x0, y0 = masks[z]
            full = np.zeros((H, W), dtype=bool)
            h, w = mask.shape
            full[y0:y0 + h, x0:x0 + w] = mask
            win = full[wy0:wy1, wx0:wx1]
            if win.any():
                frame_objs[ci + 1] = win
        if frame_objs:
            segments[i] = frame_objs
    return tmp_dir, segments, frame_to_z


def render(tree: Path, neuron: str, chain_idxs: list[int], out_path: Path, *,
          fmt: str, preview_scale: int = 1, keep_frames: bool = False) -> Path:
    """preview_scale default is 1 (no further downscale): the window this crops to is
    already small (one or a few branches, tens to a few hundred px), unlike to_gif's
    own default of 4 which assumes a full SCALE=8 frame. A whole-neuron render
    spanning a wide-spread arbor may want preview_scale=2 to keep the file size
    reasonable; pass it explicitly."""
    tag = "_".join(f"c{ci:02d}" for ci in chain_idxs) if len(chain_idxs) <= 4 else "all"
    tmp_dir = TMP_ROOT / f"{neuron}_{tag}_{tree.name}"
    frames_dir, segments, _frame_to_z = build_view(tree, neuron, chain_idxs, tmp_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        video_viz.to_gif(segments, frames_dir, out_path, obj_id=None, preview_scale=preview_scale)
    elif fmt == "mp4":
        video_viz.to_mp4(segments, frames_dir, out_path, obj_id=None, preview_scale=preview_scale)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
    if not keep_frames:
        shutil.rmtree(tmp_dir)
    print(f"[report] wrote {out_path} ({len(segments)} frames, chains {chain_idxs})")
    return out_path


def metrics_before_after(before_tree: Path, after_tree: Path, neuron: str) -> None:
    from eval.merge_metric import format_summary, load_node_table, score_run

    annotate_df = load_node_table()
    _per_b, summ_b = score_run(before_tree, annotate_df=annotate_df,
                               neurons=[neuron], membrane_source=None)
    _per_a, summ_a = score_run(after_tree, annotate_df=annotate_df,
                               neurons=[neuron], membrane_source=None)
    print(format_summary(f"{neuron} (before, {before_tree.name})", summ_b))
    print(format_summary(f"{neuron} (after,  {after_tree.name})", summ_a))
    print()
    for key in ("foreign_frame_rate", "dropout_rate", "total_foreign_nodes"):
        b, a = summ_b.get(key), summ_a.get(key)
        if b is None or a is None:
            continue
        delta = f"{a - b:+.3f}" if isinstance(b, float) else f"{a - b:+d}"
        print(f"  {key:<22} {b} -> {a}  ({delta})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_neuron = sub.add_parser("neuron-gif", help="whole-neuron merged render")
    p_neuron.add_argument("--tree", required=True)
    p_neuron.add_argument("--neuron", required=True)
    p_neuron.add_argument("--out", required=True)
    p_neuron.add_argument("--fmt", choices=["gif", "mp4"], default="gif")

    p_chain = sub.add_parser("chain-gif", help="single-chain render")
    p_chain.add_argument("--tree", required=True)
    p_chain.add_argument("--neuron", required=True)
    p_chain.add_argument("--chain", type=int, required=True)
    p_chain.add_argument("--out", required=True)
    p_chain.add_argument("--fmt", choices=["gif", "mp4"], default="gif")

    p_metrics = sub.add_parser("metrics", help="before/after merge-metric comparison")
    p_metrics.add_argument("--before", required=True)
    p_metrics.add_argument("--after", required=True)
    p_metrics.add_argument("--neuron", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "neuron-gif":
        tree = Path(args.tree)
        chain_idxs = _neuron_chain_idxs(tree, args.neuron)
        render(tree, args.neuron, chain_idxs, Path(args.out), fmt=args.fmt)
    elif args.cmd == "chain-gif":
        render(Path(args.tree), args.neuron, [args.chain], Path(args.out), fmt=args.fmt)
    elif args.cmd == "metrics":
        metrics_before_after(Path(args.before), Path(args.after), args.neuron)


if __name__ == "__main__":
    main()
