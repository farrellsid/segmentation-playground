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
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            # the bbox of the mask's CONTENT, not of the array holding it. A tier-2
            # `_pcrop` chain stores its mask in a crop-sized array and a legacy `_sam`
            # chain stores one the size of the whole frame at x0=y0=0, so taking the
            # array bounds sizes the window by which space the chain happens to live
            # in rather than by how big the neuron is. On a legacy chain that means
            # the whole cross-section for a blob of a few thousand px (AIZL chain_26:
            # a 126x95 mask on a 1154x1152 array), and since padded_window's margin is
            # proportional to the bbox it feeds an oversized pad on top.
            xs0.append(x0 + int(xs.min())); xs1.append(x0 + int(xs.max()) + 1)
            ys0.append(y0 + int(ys.min())); ys1.append(y0 + int(ys.max()) + 1)
    if not xs0:
        raise SystemExit("[report] every mask given to _compute_window is empty")
    bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
    wx0, wy0, wx1, wy1 = padded_window(bbox, full_hw, pad_frac, min_pad)
    return wx0, wy0, wx1, wy1


def sam_hw(z: int) -> tuple[int, int]:
    """The (H, W) of the _sam frame the masks actually live on.

    `pipeline.load_frame_sam(z, scale=SCALE)` returns `(image_sam, full_hw)` where
    `full_hw` is the PRE-downscale shape (~9216x9230), kept only so a caller can map
    back to full res. It is NOT the shape of the image it comes back with (~1152x1154),
    and every window/canvas in this module is in _sam px. Passing `full_hw` where the
    _sam shape belongs, which this module used to do, allocates a 64x oversized mask
    canvas per z per chain (a real contributor to the RAM-exhaustion imread failures
    these renders hit) and lets `padded_window` clip to a bound 8x too large, so a mask
    near the frame edge gets a window running past the EM and `_overlay` then resizes
    the mask down onto a shorter EM crop, misregistering it against the background."""
    em, _full_hw = pipeline.load_frame_sam(z, scale=SCALE)
    return em.shape[0], em.shape[1]


def common_window_size(windows: dict[int, tuple], full_hw: tuple) -> tuple[int, int]:
    """(w, h): the smallest size that every window in ``windows`` fits inside, capped
    at the frame. One size for all of them because a GIF wants a single canvas, and
    because a tour that changed shape at every cut would be unreadable."""
    if not windows:
        raise ValueError("common_window_size needs at least one window")
    H, W = full_hw
    w = min(W, max(x1 - x0 for x0, _y0, x1, _y1 in windows.values()))
    h = min(H, max(y1 - y0 for _x0, y0, _x1, y1 in windows.values()))
    return w, h


def center_window(window: tuple, size_wh: tuple[int, int], full_hw: tuple) -> tuple:
    """``window`` grown about its own centre to exactly ``size_wh``, slid (never
    clipped) to stay inside ``full_hw``. Sliding rather than clipping matters: a short
    crop makes the EM frame and the mask canvas disagree in size, and `_overlay`
    silently resizes the mask to fit, which reads as a mask that does not line up with
    the background."""
    want_w, want_h = size_wh
    H, W = full_hw
    if want_w > W or want_h > H:
        raise ValueError(f"window {size_wh} does not fit the frame {(W, H)}")
    x0, y0, x1, y1 = window
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    nx0 = max(0, min(cx - want_w // 2, W - want_w))
    ny0 = max(0, min(cy - want_h // 2, H - want_h))
    return nx0, ny0, nx0 + want_w, ny0 + want_h


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

    H, W = sam_hw(zs[0])

    if window is None:
        window = _compute_window(chains_masks, (H, W))
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
    window = _compute_window(after_masks, sam_hw(min(all_z)))

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
    trees = {"before": before_tree, "mask": mask_tree, "box": box_tree}
    outs = {"before": out_before, "mask": out_mask, "box": out_box}
    rendered = render_reprop_sides(trees, neuron, chain_idxs, outs, fmt=fmt,
                                   preview_scale=preview_scale)
    return rendered["before"], rendered["mask"], rendered["box"]


def reprop_window(trees: dict[str, Path], neuron: str, chain_idxs: list[int], *,
                  seed_keys: tuple[str, ...] = ("mask", "box")) -> tuple:
    """One crop window covering every reprop variant present in ``trees``.

    Sized from the reprop trees only, never the before tree: the before tail is the
    thing under test, and letting it set the frame would hide exactly the overfill
    the comparison exists to show.

    The variants' windows are unioned as BOXES, not by merging their mask dicts.
    A dict merge is keyed by z, so `{**mask_masks, **box_masks}` silently keeps only
    the later tree's mask at any z both cover, which is every z, meaning the window
    came from box-seed alone despite the code reading like a union."""
    seeds = [(k, _load_chains_masks(trees[k], neuron, chain_idxs))
             for k in seed_keys if k in trees]
    seeds = [(k, m) for k, m in seeds if m]
    if not seeds:
        named = ", ".join(str(trees[k]) for k in seed_keys if k in trees)
        raise SystemExit(f"[report] no reprop masks for {neuron} chains {chain_idxs} in {named}")
    all_z = {z for _k, m in seeds for masks in m.values() for z in masks}
    frame_hw = sam_hw(min(all_z))
    boxes = [_compute_window(m, frame_hw) for _k, m in seeds]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def render_reprop_sides(trees: dict[str, Path], neuron: str, chain_idxs: list[int],
                        outs: dict[str, Path], *, fmt: str = "gif",
                        preview_scale: int = 1) -> dict[str, Path]:
    """Render one gif per entry in ``trees`` to a single shared crop window, so the
    outputs are comparable frame for frame.

    ``trees`` may hold just "before" and "mask": a VARIANT=mask cluster run writes no
    box-seed tree, mask-seed having beaten box-seed on overfill on all four AIA/AIY
    sides, and a renderer that insists on all three turns that saving into a hard
    failure. Both reprop trees only cover the chains that actually went through
    propagate_from_corrected_seed.py, so ``chain_idxs`` should already be limited to
    a find_corrected_chains manifest rather than a neuron's full chain list."""
    missing = set(trees) - set(outs)
    if missing:
        raise ValueError(f"no output path for tree(s): {sorted(missing)}")
    window = reprop_window(trees, neuron, chain_idxs)
    return {side: render(tree, neuron, chain_idxs, outs[side], fmt=fmt,
                         preview_scale=preview_scale, window=window)
            for side, tree in trees.items()}


def tour_plan(mask_tree: Path, box_tree: Optional[Path], neuron: str,
              chain_idxs: list[int]) -> tuple[list[tuple[int, list[int], tuple]], tuple[int, int]]:
    """Plan a whole-neuron merged render as a TOUR: one static window per chain,
    visited in z order, every window padded out to one common size.

    Why not one window over the whole neuron (what this used to do): measured on the
    2026-08-13 reprop trees, a median of ONE chain is present at any given z while the
    masks travel 500-700px down the stack, so a single union window is nearly the whole
    worm cross-section and the mask covers a median 0.11-0.27% of it (AIAR: 84 of 176
    frames under 0.1%, and the docx's middle-frame thumbnail landed on a visually blank
    one). Per-chain windows are median 262x273 instead of 622x1031.

    Each chain's window comes from the union of both reprop trees' masks, the same rule
    render_reprop_triple uses and for the same reason: neither variant is more "correct"
    a priori, so cropping to one of them would beg the question.

    Returns ([(chain_idx, zs, window), ...] in z order, (w, h) common size)."""
    masks_by_tree = [_load_chains_masks(t, neuron, chain_idxs)
                     for t in (mask_tree, box_tree) if t is not None]
    zs_by_chain = {ci: sorted({z for m in masks_by_tree for z in m.get(ci, {})})
                   for ci in chain_idxs}
    zs_by_chain = {ci: zs for ci, zs in zs_by_chain.items() if zs}
    if not zs_by_chain:
        raise SystemExit(f"[report] no reprop masks for {neuron} chains {chain_idxs}")

    frame_hw = sam_hw(min(zs[0] for zs in zs_by_chain.values()))
    # union the two variants' WINDOWS, not their mask dicts: a dict merge is keyed by
    # z, so the second tree's mask at a z replaces the first's instead of covering
    # both, and a chain where the variants diverge spatially would get cropped to
    # whichever tree merged last. Computing each variant's window and taking the
    # bounding box of those is the union that was actually intended.
    raw = {}
    for ci, zs in zs_by_chain.items():
        per_tree = [_compute_window({ci: m[ci]}, frame_hw)
                    for m in masks_by_tree if m.get(ci)]
        raw[ci] = (min(w[0] for w in per_tree), min(w[1] for w in per_tree),
                   max(w[2] for w in per_tree), max(w[3] for w in per_tree))
    combined = zs_by_chain
    size = common_window_size(raw, frame_hw)
    plan = [(ci, combined[ci], center_window(raw[ci], size, frame_hw))
            for ci in combined]
    plan.sort(key=lambda item: (item[1][0], item[0]))   # by first z, then chain idx
    return plan, size


def build_tour_view(trees: dict[str, Path], neuron: str, chain_idxs: list[int],
                    tmp_dir: Path, plan: list[tuple[int, list[int], tuple]]
                    ) -> tuple[Path, dict[str, dict], list[tuple[int, int]]]:
    """One JPEG sequence + one `segments` dict PER TREE over the same frames.

    The EM is read once per frame and shared by every tree, not re-read per tree as
    three separate build_view calls would: the window at a given frame depends only on
    which chain the tour is visiting, never on which tree is being drawn, so the three
    variants are literally the same pixels with different overlays. That is a 3x cut in
    full-res reads (~255MB each), which is the load that produced the transient imread
    failures `pipeline.frames.load_frame_sam` now retries around.

    It also forces the three gifs to share a frame index -> (chain, z) mapping, so
    frame k of the before gif and frame k of the mask gif are the same slice of the
    same chain by construction rather than by coincidence.

    Returns (frames_dir, {side: segments}, [(chain_idx, z), ...] per frame)."""
    masks = {side: _load_chains_masks(tree, neuron, chain_idxs)
             for side, tree in trees.items()}
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    segments: dict[str, dict[int, dict[int, np.ndarray]]] = {s: {} for s in trees}
    frame_meta: list[tuple[int, int]] = []
    H, W = sam_hw(plan[0][1][0])

    i = 0
    for chain_idx, zs, (wx0, wy0, wx1, wy1) in plan:
        for z in zs:
            em, _ = pipeline.load_frame_sam(z, scale=SCALE)
            em_rgb = em if em.ndim == 3 else np.stack([em] * 3, axis=-1)
            em_win = np.ascontiguousarray(em_rgb[wy0:wy1, wx0:wx1].astype(np.uint8))
            # a tour cuts between chains, so burn in which chain and slice you are
            # looking at; without it ~20 hard cuts per neuron are unreadable
            _stamp(em_win, f"chain_{chain_idx:02d}  z={z}")
            cv2.imwrite(str(tmp_dir / f"{i:05d}.jpg"),
                        cv2.cvtColor(em_win, cv2.COLOR_RGB2BGR))
            for side in trees:
                frame_objs = {}
                # every chain that shows up inside this window, not just the one being
                # visited: a neighbouring chain in view is exactly the context a
                # "merged" render is for, and each keeps its own stable colour
                for ci, chain_masks in masks[side].items():
                    if z not in chain_masks:
                        continue
                    mask, x0, y0 = chain_masks[z]
                    full = np.zeros((H, W), dtype=bool)
                    h, w = mask.shape
                    full[y0:y0 + h, x0:x0 + w] = mask
                    win = full[wy0:wy1, wx0:wx1]
                    if win.any():
                        frame_objs[ci + 1] = win
                # recorded even when EMPTY: video_viz._frame_indices keys off dict
                # membership, so dropping a maskless frame would shorten that one
                # variant's gif and silently break the frame-for-frame alignment
                # between the three outputs that the whole comparison rests on
                segments[side][i] = frame_objs
            frame_meta.append((chain_idx, z))
            i += 1
    return tmp_dir, segments, frame_meta


def _stamp(img_rgb: np.ndarray, text: str) -> None:
    """Burn a small caption into the top-left, in place. Drawn twice, dark then light,
    so it stays readable over both the bright cytoplasm and the dark membrane it may
    land on."""
    org = (6, 16)
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img_rgb, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, thickness, cv2.LINE_AA)


def render_reprop_tour(trees: dict[str, Path], neuron: str, chain_idxs: list[int],
                       outs: dict[str, Path], *, mask_tree_key: str = "mask",
                       box_tree_key: str = "box", fmt: str = "gif",
                       preview_scale: int = 1, keep_frames: bool = False
                       ) -> list[tuple[int, int]]:
    """Whole-neuron merged render as a per-chain tour, one output per entry in
    ``trees`` (keys must match ``outs``). ``trees`` may omit the box-seed variant: a
    VARIANT=mask cluster run writes no box tree at all, and the old triple renderer
    aborted all three gifs when one tree had no masks.

    preview_scale defaults to 1 here, unlike the old neuron-triple's 2: the window is
    now a chain's own extent (median 262x273) rather than the whole cross-section, so
    halving it again would throw away the legibility this change exists to buy."""
    plan, size = tour_plan(trees[mask_tree_key], trees.get(box_tree_key), neuron,
                           chain_idxs)
    n_frames = sum(len(zs) for _ci, zs, _w in plan)
    print(f"[report] {neuron} tour: {len(plan)} chains, {n_frames} frames, "
          f"window {size[0]}x{size[1]}")

    tmp_dir = TMP_ROOT / f"{neuron}_tour"
    frames_dir, segments, frame_meta = build_tour_view(trees, neuron, chain_idxs,
                                                       tmp_dir, plan)
    for side, out_path in outs.items():
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(segments.get(side, {}).values()):
            print(f"[report]   {side}: no masks anywhere in the tour, skipped")
            continue
        writer = video_viz.to_gif if fmt == "gif" else video_viz.to_mp4
        writer(segments[side], frames_dir, out_path, obj_id=None,
               preview_scale=preview_scale, color=None)
        print(f"[report]   wrote {out_path} ({len(segments[side])} frames)")
    if not keep_frames:
        shutil.rmtree(tmp_dir)
    return frame_meta


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


def _manifest_chain_idxs(manifest: str, neuron: str) -> list[int]:
    """The chain indices a corrected-chains manifest lists for one neuron, sorted.
    A manifest covers several neurons (AUA's holds both AUAL and AUAR), and only the
    chains it names went through reprop, so a neuron's full on-disk chain list is the
    wrong set to render or score against."""
    with open(manifest, newline="") as f:
        return sorted(int(row["chain_idx"]) for row in csv.DictReader(f)
                      if row["neuron"] == neuron)


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
        help="before / mask-seed-reprop / box-seed-reprop whole-neuron merged render, "
             "a TOUR: one static window per chain, visited in z order, every window "
             "padded to one common size. --box-tree is optional (VARIANT=mask runs)")
    p_neuron_triple.add_argument("--before", required=True)
    p_neuron_triple.add_argument("--mask-tree", required=True)
    p_neuron_triple.add_argument("--box-tree",
                                 help="omit for a VARIANT=mask run that wrote no box tree")
    p_neuron_triple.add_argument("--manifest", required=True,
                                 help="neuron,chain_idx,anchor_z CSV, filtered to --neuron's rows")
    p_neuron_triple.add_argument("--neuron", required=True)
    p_neuron_triple.add_argument("--out-before", required=True)
    p_neuron_triple.add_argument("--out-mask", required=True)
    p_neuron_triple.add_argument("--out-box")
    p_neuron_triple.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_neuron_triple.add_argument("--preview-scale", type=int, default=1)

    p_neuron_tour = sub.add_parser(
        "neuron-gif-tour",
        help="whole-neuron merged render of ONE tree, tour-framed (one static window "
             "per chain, visited in z order). The single-tree counterpart of "
             "neuron-gif-triple, for when there is nothing to compare against yet")
    p_neuron_tour.add_argument("--tree", required=True)
    p_neuron_tour.add_argument("--manifest", required=True,
                               help="neuron,chain_idx,anchor_z CSV, filtered to --neuron's rows")
    p_neuron_tour.add_argument("--neuron", required=True)
    p_neuron_tour.add_argument("--out", required=True)
    p_neuron_tour.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    p_neuron_tour.add_argument("--preview-scale", type=int, default=1)

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
        chain_idxs = _manifest_chain_idxs(args.manifest, args.neuron)
        trees = {"before": Path(args.before), "mask": Path(args.mask_tree)}
        outs = {"before": Path(args.out_before), "mask": Path(args.out_mask)}
        if args.box_tree:
            if not args.out_box:
                ap.error("--box-tree given without --out-box")
            trees["box"] = Path(args.box_tree)
            outs["box"] = Path(args.out_box)
        render_reprop_tour(trees, args.neuron, chain_idxs, outs, fmt=args.fmt,
                           preview_scale=args.preview_scale)
    elif args.cmd == "neuron-gif-tour":
        chain_idxs = _manifest_chain_idxs(args.manifest, args.neuron)
        render_reprop_tour({"tree": Path(args.tree)}, args.neuron, chain_idxs,
                           {"tree": Path(args.out)}, mask_tree_key="tree",
                           fmt=args.fmt, preview_scale=args.preview_scale)
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
