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

    # whole-neuron merged render, before vs after, EACH SIDE computes its own window
    py -3 experiments/report_assets.py neuron-gif --tree <before-tree> --neuron AIYL --out before.gif
    py -3 experiments/report_assets.py neuron-gif --tree <after-tree>  --neuron AIYL --out after.gif

    # one chain, before vs after, EACH SIDE computes its own window
    py -3 experiments/report_assets.py chain-gif --tree <before-tree> --neuron AIYL --chain 0 --out before_c0.gif
    py -3 experiments/report_assets.py chain-gif --tree <after-tree>  --neuron AIYL --chain 0 --out after_c0.gif

    # one chain, before vs after, BOTH SIDES cropped to the AFTER (revised) mask's own
    # window: use this instead of two separate chain-gif calls whenever the correction
    # may have changed the mask's extent (e.g. the lasso tool extending it outward),
    # since two independently-computed windows can drift apart and make the pair
    # visually misleading (the after render's own new area cropped off by the
    # before-derived window, or vice versa).
    py -3 experiments/report_assets.py chain-gif-pair --before <before-tree> --after <after-tree> \
        --neuron AIYL --chain 0 --out-before before_c0.gif --out-after after_c0.gif

    # whole-neuron merged render, before vs after, BOTH SIDES cropped to the AFTER tree's window
    py -3 experiments/report_assets.py neuron-gif-pair --before <before-tree> --after <after-tree> \
        --neuron AIYL --out-before before.gif --out-after after.gif

    # one chain, before / mask-seed-reprop / box-seed-reprop, ALL THREE cropped to the
    # union of both reprop trees' windows: use for comparing the two re-propagation
    # seeding strategies against each other and against the pre-reprop state.
    py -3 experiments/report_assets.py chain-gif-triple --before <before-tree> \
        --mask-tree <mask-seed-reprop-tree> --box-tree <box-seed-reprop-tree> \
        --neuron AIYL --chain 0 --out-before before_c0.gif --out-mask mask_c0.gif \
        --out-box box_c0.gif

    # whole-neuron merged render, before / mask-seed-reprop / box-seed-reprop, every
    # reprop'd chain from the manifest overlaid in ONE crop, all three cropped to the
    # union of both reprop trees' windows
    py -3 experiments/report_assets.py neuron-gif-triple --before <before-tree> \\
        --mask-tree <mask-seed-reprop-tree> --box-tree <box-seed-reprop-tree> \\
        --manifest cluster/corrected_chains_AIA.csv --neuron AIYL \\
        --out-before before.gif --out-mask mask.gif --out-box box.gif

    # whole-neuron before/after metrics table
    py -3 experiments/report_assets.py metrics --before <before-tree> --after <after-tree> --neuron AIYL

    # before/mask-seed/box-seed metrics table, restricted to a manifest's reprop'd
    # chains, one table per neuron in the manifest
    py -3 experiments/report_assets.py metrics-reprop --before <before-tree> \\
        --mask-tree <mask-seed-reprop-tree> --box-tree <box-seed-reprop-tree> \\
        --manifest cluster/corrected_chains_AIA.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Optional

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


def _load_chains_masks(tree: Path, neuron: str, chain_idxs: list[int]) -> dict:
    """{chain_idx: chain_masks_in_sam(chain_dir)} for every chain that has masks in
    `tree`. Shared by build_view and render_before_after so a window computed from
    ONE tree's masks (typically the AFTER/revised tree) can be reused when rendering
    a DIFFERENT tree (typically the BEFORE tree)."""
    chains_masks = {}
    for ci in chain_idxs:
        cdir = tree / neuron / f"chain_{ci:02d}"
        if not cdir.exists():
            continue
        masks = pipeline.chain_masks_in_sam(cdir)
        if masks:
            chains_masks[ci] = masks
    return chains_masks


def padded_window(bbox: tuple[int, int, int, int], full_hw: tuple,
                  pad_frac: float = 0.25, min_pad: int = 15
                  ) -> tuple[int, int, int, int]:
    """Expand a (bx0, by0, bx1, by1) bbox by a margin PROPORTIONAL to its own
    width/height, not a fixed absolute pixel count: a fixed pad (the original
    design, 40px every side) swamps a small neurite's already-small bbox in mostly
    empty margin, exactly backwards from what makes a small mask legible in a
    report table cell. A large mask instead gets a proportionally large margin, so
    a long chain still shows real surrounding context. ``min_pad`` is an absolute
    floor so a genuinely tiny mask (a few px) does not end up with an unusably
    thin sliver of context from a tiny fraction of a tiny number."""
    H, W = full_hw
    bx0, by0, bx1, by1 = bbox
    bw, bh = bx1 - bx0, by1 - by0
    pad_x = max(min_pad, int(round(bw * pad_frac)))
    pad_y = max(min_pad, int(round(bh * pad_frac)))
    wx0, wy0 = max(0, bx0 - pad_x), max(0, by0 - pad_y)
    wx1, wy1 = min(W, bx1 + pad_x), min(H, by1 + pad_y)
    return wx0, wy0, wx1, wy1


def _compute_window(chains_masks: dict, full_hw: tuple, pad_frac: float = 0.25,
                    min_pad: int = 15) -> tuple[int, int, int, int]:
    """(wx0, wy0, wx1, wy1) in _sam px: the union of every mask's extent across every
    z in ``chains_masks``, padded (see padded_window) and clipped to ``full_hw``.
    Extracted out of build_view so a window computed from one tree can be reused
    verbatim when rendering a different tree (see render_before_after): a
    correction that changes a mask's spatial extent (e.g. the lasso tool extending
    it outward) would otherwise make the before and after renders crop to two
    different, drifting windows, misleading rather than clarifying the comparison."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for masks in chains_masks.values():
        for mask, x0, y0 in masks.values():
            if not mask.any():
                continue
            h, w = mask.shape
            xs0.append(x0); ys0.append(y0); xs1.append(x0 + w); ys1.append(y0 + h)
    if not xs0:
        raise SystemExit("[report] every mask given to _compute_window is empty")
    bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
    wx0, wy0, wx1, wy1 = padded_window(bbox, full_hw, pad_frac, min_pad)
    return wx0, wy0, wx1, wy1


def build_view(tree: Path, neuron: str, chain_idxs: list[int], tmp_dir: Path, *,
               window: Optional[tuple] = None) -> tuple[Path, dict, dict]:
    """A 0-indexed JPEG sequence + {frame_idx: {chain_idx+1: mask}} over the UNION
    z-range of the given chains in `tree`, cropped to a FIXED window covering every
    mask across every z (padded), not the whole raw frame: a small chain's mask is
    imperceptible against a full worm cross-section, the same "invisible mask at
    full-frame scale" problem this session already hit with a micro_sam spot-check
    and fixed the same way there. Each chain gets a stable colour (chain_idx + 1, so
    a single-chain render and a whole-neuron render use the same colour for the same
    chain). ``window`` (wx0, wy0, wx1, wy1 in _sam px), when given, is used AS-IS
    instead of computed from this tree's own masks: this is what lets a before/after
    pair share one window via render_before_after, typically the AFTER/revised
    tree's own extent. Returns (frames_dir, segments, frame_to_z)."""
    chains_masks = _load_chains_masks(tree, neuron, chain_idxs)
    all_z = {z for masks in chains_masks.values() for z in masks}
    if not all_z:
        raise SystemExit(f"[report] no masks for {neuron} chains {chain_idxs} in {tree}")

    zs = sorted(all_z)
    frame_to_z = {i: z for i, z in enumerate(zs)}
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    _em0, full_hw = pipeline.load_frame_sam(zs[0], scale=SCALE)
    H, W = full_hw

    if window is None:
        window = _compute_window(chains_masks, full_hw)
    wx0, wy0, wx1, wy1 = window

    # crop EM + masks to that window for every z
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
          fmt: str, preview_scale: int = 1, keep_frames: bool = False,
          window: Optional[tuple] = None) -> Path:
    """preview_scale default is 1 (no further downscale): the window this crops to is
    already small (one or a few branches, tens to a few hundred px), unlike to_gif's
    own default of 4 which assumes a full SCALE=8 frame. A whole-neuron render
    spanning a wide-spread arbor may want preview_scale=2 to keep the file size
    reasonable; pass it explicitly. ``window``, when given, is forwarded to
    build_view as-is (see render_before_after).

    A single-chain render (the common case, chain-gif/chain-gif-pair) uses one
    fixed, deliberately chosen high-contrast color (video_viz.HIGHLIGHT_COLOR) for
    its one mask, instead of the chain-index-dependent palette cycling: with only
    one object ever on screen, per-object color cycling picks an arbitrary,
    sometimes low-contrast entry for no benefit. A multi-chain render (neuron-gif)
    keeps per-chain colors, several simultaneous objects still need to be told
    apart."""
    tag = "_".join(f"c{ci:02d}" for ci in chain_idxs) if len(chain_idxs) <= 4 else "all"
    tmp_dir = TMP_ROOT / f"{neuron}_{tag}_{tree.name}"
    frames_dir, segments, _frame_to_z = build_view(tree, neuron, chain_idxs, tmp_dir,
                                                   window=window)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    color = video_viz.HIGHLIGHT_COLOR if len(chain_idxs) == 1 else None
    if fmt == "gif":
        video_viz.to_gif(segments, frames_dir, out_path, obj_id=None,
                         preview_scale=preview_scale, color=color)
    elif fmt == "mp4":
        video_viz.to_mp4(segments, frames_dir, out_path, obj_id=None,
                         preview_scale=preview_scale, color=color)
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
    if not keep_frames:
        shutil.rmtree(tmp_dir)
    print(f"[report] wrote {out_path} ({len(segments)} frames, chains {chain_idxs})")
    return out_path


def render_before_after(before_tree: Path, after_tree: Path, neuron: str,
                        chain_idxs: list[int], out_before: Path, out_after: Path, *,
                        fmt: str = "gif", preview_scale: int = 1) -> tuple[Path, Path]:
    """Render a before/after pair to the SAME crop window, computed once from the
    AFTER (revised) tree's own masks and reused for both sides. Without this, each
    side would compute its own window from its own masks, and a correction that
    changes a mask's spatial extent (e.g. the lasso tool extending it outward, or a
    box-seed re-predict pulling it in) makes the two windows drift apart: the before
    render can crop off the very area the after render newly covers, or vice versa,
    undermining the comparison the report exists to make."""
    after_masks = _load_chains_masks(after_tree, neuron, chain_idxs)
    all_z = {z for masks in after_masks.values() for z in masks}
    if not all_z:
        raise SystemExit(f"[report] no AFTER masks for {neuron} chains {chain_idxs} in {after_tree}")
    _em0, full_hw = pipeline.load_frame_sam(min(all_z), scale=SCALE)
    window = _compute_window(after_masks, full_hw)

    before_path = render(before_tree, neuron, chain_idxs, out_before,
                         fmt=fmt, preview_scale=preview_scale, window=window)
    after_path = render(after_tree, neuron, chain_idxs, out_after,
                        fmt=fmt, preview_scale=preview_scale, window=window)
    return before_path, after_path


def render_reprop_triple(before_tree: Path, mask_tree: Path, box_tree: Path, neuron: str,
                         chain_idxs: list[int], out_before: Path, out_mask: Path,
                         out_box: Path, *, fmt: str = "gif", preview_scale: int = 1
                         ) -> tuple[Path, Path, Path]:
    """Render a before / mask-seed-reprop / box-seed-reprop triple to ONE shared crop
    window, computed from the UNION of both reprop trees' masks (not just one, since
    neither reprop variant is more "correct" a priori the way a human-revised AFTER
    tree was in render_before_after, comparing them is the whole point). Both reprop
    trees only cover the chains that actually got re-corrected and re-propagated (the
    other chains never ran through propagate_from_corrected_seed.py at all), so
    ``chain_idxs`` should already be limited to that set (e.g. from
    find_corrected_chains), not a neuron's full chain list."""
    mask_masks = _load_chains_masks(mask_tree, neuron, chain_idxs)
    box_masks = _load_chains_masks(box_tree, neuron, chain_idxs)
    all_z = {z for masks in mask_masks.values() for z in masks} | \
            {z for masks in box_masks.values() for z in masks}
    if not all_z:
        raise SystemExit(f"[report] no reprop masks for {neuron} chains {chain_idxs} "
                         f"in {mask_tree} or {box_tree}")
    _em0, full_hw = pipeline.load_frame_sam(min(all_z), scale=SCALE)
    # union the two trees' mask sets before computing one window over both, so a chain
    # where box-seed and mask-seed diverge spatially still gets a window covering
    # whichever one reaches further, never just one variant's own footprint.
    combined = {ci: {**mask_masks.get(ci, {}), **box_masks.get(ci, {})} for ci in chain_idxs}
    window = _compute_window(combined, full_hw)

    before_path = render(before_tree, neuron, chain_idxs, out_before,
                         fmt=fmt, preview_scale=preview_scale, window=window)
    mask_path = render(mask_tree, neuron, chain_idxs, out_mask,
                       fmt=fmt, preview_scale=preview_scale, window=window)
    box_path = render(box_tree, neuron, chain_idxs, out_box,
                      fmt=fmt, preview_scale=preview_scale, window=window)
    return before_path, mask_path, box_path


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


def metrics_reprop(before_tree: Path, mask_tree: Path, box_tree: Path, neuron: str,
                   chain_idxs: list[int]) -> None:
    """Merge-metric comparison across before / mask-seed-reprop / box-seed-reprop,
    restricted to ``chain_idxs`` (the manifest's reprop'd chains for this neuron),
    not the neuron's full chain list: the before tree still has every chain the
    neuron ever had, but the two reprop trees only ever contain the chains that
    actually went through propagate_from_corrected_seed.py, so scoring the before
    tree unfiltered would average in dozens of untouched chains and make the
    comparison meaningless. mask_tree/box_tree are filtered too, defensively, in
    case that assumption about their contents ever stops holding."""
    from eval.merge_metric import format_summary, load_node_table, score_run, summarize

    annotate_df = load_node_table()
    chain_set = set(chain_idxs)
    summaries = {}
    for label, tree in (("before", before_tree), ("mask-seed", mask_tree),
                        ("box-seed", box_tree)):
        # The reprop trees are written per-chain by propagate_from_corrected_seed.py,
        # not a full batch.py run, so they never get a _run_meta.json for score_run
        # to read the grid scale from; pass it explicitly (matches this module's own
        # SCALE, which every reprop tree is built on).
        # membrane_source="auto" (score_run's own default) runs the membrane pass,
        # needed for mean_underfill_fraction: the Phase-0 foreign-node signal alone
        # only sees OVERFILL (a mask reaching a neighbour's centreline), it has no
        # notion of a mask falling short of its own cell's true extent.
        per, _summ = score_run(tree, annotate_df=annotate_df, neurons=[neuron],
                               membrane_source="auto", scale=SCALE)
        per_sub = per[per["chain_idx"].isin(chain_set)] if len(per) else per
        summ = summarize(per_sub)
        summaries[label] = summ
        print(format_summary(f"{neuron} ({label}, {tree.name})", summ))
    print()
    base = summaries["before"]
    for label in ("mask-seed", "box-seed"):
        s = summaries[label]
        print(f"  {label} vs before:")
        for key in ("foreign_frame_rate", "dropout_rate", "total_foreign_nodes",
                   "mild_bleed_rate", "spanning_merge_rate", "mean_boundary_on_membrane",
                   "mean_underfill_fraction"):
            b, a = base.get(key), s.get(key)
            if b is None or a is None:
                continue
            delta = f"{a - b:+.3f}" if isinstance(b, float) else f"{a - b:+d}"
            print(f"    {key:<24} {b} -> {a}  ({delta})")


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

    p_chain_pair = sub.add_parser(
        "chain-gif-pair",
        help="before/after single-chain render, both sides cropped to the AFTER tree's window")
    p_chain_pair.add_argument("--before", required=True)
    p_chain_pair.add_argument("--after", required=True)
    p_chain_pair.add_argument("--neuron", required=True)
    p_chain_pair.add_argument("--chain", type=int, required=True)
    p_chain_pair.add_argument("--out-before", required=True)
    p_chain_pair.add_argument("--out-after", required=True)
    p_chain_pair.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_chain_pair.add_argument("--preview-scale", type=int, default=1)

    p_chain_triple = sub.add_parser(
        "chain-gif-triple",
        help="before / mask-seed-reprop / box-seed-reprop single-chain render, "
             "all three cropped to the union of both reprop trees' windows")
    p_chain_triple.add_argument("--before", required=True)
    p_chain_triple.add_argument("--mask-tree", required=True)
    p_chain_triple.add_argument("--box-tree", required=True)
    p_chain_triple.add_argument("--neuron", required=True)
    p_chain_triple.add_argument("--chain", type=int, required=True)
    p_chain_triple.add_argument("--out-before", required=True)
    p_chain_triple.add_argument("--out-mask", required=True)
    p_chain_triple.add_argument("--out-box", required=True)
    p_chain_triple.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_chain_triple.add_argument("--preview-scale", type=int, default=1)

    p_neuron_triple = sub.add_parser(
        "neuron-gif-triple",
        help="before / mask-seed-reprop / box-seed-reprop whole-neuron merged render "
             "(every reprop'd chain overlaid in one crop), all three cropped to the "
             "union of both reprop trees' windows")
    p_neuron_triple.add_argument("--before", required=True)
    p_neuron_triple.add_argument("--mask-tree", required=True)
    p_neuron_triple.add_argument("--box-tree", required=True)
    p_neuron_triple.add_argument("--manifest", required=True,
                                 help="neuron,chain_idx,anchor_z CSV, filtered to --neuron's rows")
    p_neuron_triple.add_argument("--neuron", required=True)
    p_neuron_triple.add_argument("--out-before", required=True)
    p_neuron_triple.add_argument("--out-mask", required=True)
    p_neuron_triple.add_argument("--out-box", required=True)
    p_neuron_triple.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_neuron_triple.add_argument("--preview-scale", type=int, default=2)

    p_neuron_pair = sub.add_parser(
        "neuron-gif-pair",
        help="before/after whole-neuron render, both sides cropped to the AFTER tree's window")
    p_neuron_pair.add_argument("--before", required=True)
    p_neuron_pair.add_argument("--after", required=True)
    p_neuron_pair.add_argument("--neuron", required=True)
    p_neuron_pair.add_argument("--out-before", required=True)
    p_neuron_pair.add_argument("--out-after", required=True)
    p_neuron_pair.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_neuron_pair.add_argument("--preview-scale", type=int, default=1)

    p_metrics = sub.add_parser("metrics", help="before/after merge-metric comparison")
    p_metrics.add_argument("--before", required=True)
    p_metrics.add_argument("--after", required=True)
    p_metrics.add_argument("--neuron", required=True)

    p_metrics_reprop = sub.add_parser(
        "metrics-reprop",
        help="before/mask-seed/box-seed merge-metric comparison, restricted to a "
             "manifest's reprop'd chains, one comparison per neuron in the manifest")
    p_metrics_reprop.add_argument("--before", required=True)
    p_metrics_reprop.add_argument("--mask-tree", required=True)
    p_metrics_reprop.add_argument("--box-tree", required=True)
    p_metrics_reprop.add_argument("--manifest", required=True,
                                  help="neuron,chain_idx,anchor_z CSV")

    args = ap.parse_args(argv)

    if args.cmd == "neuron-gif":
        tree = Path(args.tree)
        chain_idxs = _neuron_chain_idxs(tree, args.neuron)
        render(tree, args.neuron, chain_idxs, Path(args.out), fmt=args.fmt)
    elif args.cmd == "chain-gif":
        render(Path(args.tree), args.neuron, [args.chain], Path(args.out), fmt=args.fmt)
    elif args.cmd == "chain-gif-pair":
        render_before_after(Path(args.before), Path(args.after), args.neuron, [args.chain],
                            Path(args.out_before), Path(args.out_after), fmt=args.fmt,
                            preview_scale=args.preview_scale)
    elif args.cmd == "chain-gif-triple":
        render_reprop_triple(Path(args.before), Path(args.mask_tree), Path(args.box_tree),
                             args.neuron, [args.chain], Path(args.out_before),
                             Path(args.out_mask), Path(args.out_box), fmt=args.fmt,
                             preview_scale=args.preview_scale)
    elif args.cmd == "neuron-gif-triple":
        chain_idxs = []
        with open(args.manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row["neuron"] == args.neuron:
                    chain_idxs.append(int(row["chain_idx"]))
        chain_idxs.sort()
        render_reprop_triple(Path(args.before), Path(args.mask_tree), Path(args.box_tree),
                             args.neuron, chain_idxs, Path(args.out_before),
                             Path(args.out_mask), Path(args.out_box), fmt=args.fmt,
                             preview_scale=args.preview_scale)
    elif args.cmd == "neuron-gif-pair":
        after_tree = Path(args.after)
        chain_idxs = _neuron_chain_idxs(after_tree, args.neuron)
        render_before_after(Path(args.before), after_tree, args.neuron, chain_idxs,
                            Path(args.out_before), Path(args.out_after), fmt=args.fmt,
                            preview_scale=args.preview_scale)
    elif args.cmd == "metrics":
        metrics_before_after(Path(args.before), Path(args.after), args.neuron)
    elif args.cmd == "metrics-reprop":
        by_neuron: dict[str, list[int]] = {}
        with open(args.manifest, newline="") as f:
            for row in csv.DictReader(f):
                by_neuron.setdefault(row["neuron"], []).append(int(row["chain_idx"]))
        for neuron, chain_idxs in by_neuron.items():
            metrics_reprop(Path(args.before), Path(args.mask_tree), Path(args.box_tree),
                           neuron, sorted(chain_idxs))
            print()


if __name__ == "__main__":
    main()
