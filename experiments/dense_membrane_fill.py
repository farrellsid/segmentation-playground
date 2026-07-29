"""Visual check: apply membrane autofill to EVERY neuron on one dense frame.

So far the autofill (grow-to-membrane) idea has only been judged by metrics on sampled
single masks. This renders it on a whole dense frame so the effect is actually visible:
raw per-slice merge vs the same frame after each neuron is grown to its ridge walls, plus
a panel showing only the pixels the fill added (green) so leaks across weak membranes stand
out. CPU only, no model, reuses the cached dense-overlay index and the existing grow.

Each neuron is grown INSIDE its own bbox+pad window (not on the globally-connected free
space, which would let every cell balloon into all the extracellular space at once), which
mirrors how the single-mask autofill works.

    py -3 experiments/dense_membrane_fill.py                 # z=1456, the current-work frame
    py -3 experiments/dense_membrane_fill.py --z 1472 --pad 16
    py -3 experiments/dense_membrane_fill.py --mm-window 1 --mm-combine median
    py -3 experiments/dense_membrane_fill.py --sweep-temporal    # grid window/combine vs baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from experiments import dense_overlay as do
from experiments.membrane_autofill_demo import grow_to_membrane
from sam2_utils import membrane as mb
from sam2_utils.perframe import resolve_overlaps_watershed
from pipeline.predict import _point_in_mask
from eval.merge_metric import load_node_table, nodes_by_z, DEFAULT_RADIUS

INDEX_CACHE = Path("docs/figures/sam3-bakeoff/dense-overlay/_index.json")
OUT_DIR = Path("docs/figures/presentation/dense-membrane-fill")
SCALE = 8


def foreign_count(mask_full: np.ndarray, nodes, neuron: str, radius: int) -> int:
    """How many OTHER neurons' skeleton nodes this full-frame mask engulfs (the merge/bleed
    metric: a foreign node inside the mask == a merge/leak into that neuron)."""
    n = 0
    for x, y, cell, *_ in nodes:
        if cell != neuron and _point_in_mask(mask_full, x, y, radius):
            n += 1
    return n


def _shift_diag(crops, center_i, max_shift=5):
    """Diagnostic only, not used to align anything: the raw (pre-clamp) phase-correlation
    shift magnitude for each non-reference crop against crops[center_i], plus whether that
    raw shift exceeds max_shift.

    register_crops clamps every estimated shift to +/- max_shift before applying it, so a
    real slice-to-slice jitter bigger than the clamp is not skipped, it is truncated and
    then APPLIED, misaligning the crop rather than leaving it alone. That is a different
    failure mode from "registration found near-zero shift and combining still blurred
    things", so this logs which one is actually happening rather than guessing. Duplicates
    register_crops's shift-estimation call locally so register_crops's own return type and
    contract stay untouched."""
    from skimage.registration import phase_cross_correlation

    ref = crops[center_i].astype(np.float32)
    raw_mags, clamped = [], 0
    for i, crop in enumerate(crops):
        if i == center_i:
            continue
        moving = crop.astype(np.float32)
        shift, _error, _diffphase = phase_cross_correlation(ref, moving, upsample_factor=1)
        mag = float(np.max(np.abs(shift)))
        raw_mags.append(mag)
        if mag > max_shift:
            clamped += 1
    return raw_mags, clamped


def grow_all(nmasks: dict, frames: dict, center_z: int, window: int, combine: str,
            pad: int, nodes, radius: int):
    """Grow each neuron to its ridge walls once, UNCAPPED, inside a local bbox+pad window.

    The membrane map for each neuron's crop is built from `frames`, a {z: em_gray} dict
    covering at least [center_z - window, center_z + window] (missing z, e.g. near the
    stack's edge, are simply absent and drop out of the window). window=0 uses only
    frames[center_z], reproducing the original single-slice path exactly: register_crops
    and project_crops are no-ops on a length-1 list.

    Returns {neuron: rec} with the raw and grown full-frame masks, their areas, underfill,
    the raw-mask centroid seed, and the foreign-node count each mask engulfs (raw vs grown).
    Growing uncapped lets a cap be applied afterwards as a cheap post-filter, so a cap sweep
    needs only this one grow pass. When window > 0, also prints a one-line shift-clamp
    diagnostic (see _shift_diag) across all crops processed in this call."""
    H, W = frames[center_z].shape[:2]
    recs = {}
    all_raw_mags: list[float] = []
    all_clamped = 0
    for n, full in nmasks.items():
        ys, xs = np.where(full)
        if ys.size == 0:
            continue
        y1, y2 = max(0, ys.min() - pad), min(H, ys.max() + pad)
        x1, x2 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        win = full[y1:y2, x1:x2]
        zs = [z for z in range(center_z - window, center_z + window + 1) if z in frames]
        crops = [frames[z][y1:y2, x1:x2] for z in zs]
        if len(crops) > 1:
            center_i = zs.index(center_z)
            raw_mags, clamped = _shift_diag(crops, center_i)
            all_raw_mags.extend(raw_mags)
            all_clamped += clamped
            crops = mb.register_crops(crops, center=center_i)
        em_crop = mb.project_crops(crops, combine=combine)
        mem = mb.membrane_map(em_crop)
        grown, _capped = grow_to_membrane(win, mem, cap=1e9)  # no clamp; cap applied later
        g_full = np.zeros((H, W), bool)
        g_full[y1:y2, x1:x2] = grown
        recs[n] = {
            "raw": full, "grown": g_full,
            "raw_area": int(win.sum()), "grown_area": int(grown.sum()),
            "raw_uf": float(mb.underfill_fraction(win, mem)),
            "grown_uf": float(mb.underfill_fraction(grown, mem)),
            "seed": (float(xs.mean()), float(ys.mean())),  # raw-centroid, trusted core
            "raw_foreign": foreign_count(full, nodes, n, radius),
            "grown_foreign": foreign_count(g_full, nodes, n, radius),
        }
    if window > 0 and all_raw_mags:
        print(f"[fill]   shift stats (window={window}, combine={combine}): "
              f"{len(all_raw_mags)} crops, {all_clamped} clamped (|raw shift| > 5px), "
              f"raw shift range [{min(all_raw_mags):.1f}, {max(all_raw_mags):.1f}]px")
    return recs


def apply_cap(recs: dict, cap: float, uf_min: float = 0.0):
    """Pick grown-or-raw per neuron. Grow ONLY cells that are conclusively underfilling
    (raw underfill >= uf_min); leave the rest untouched. Of those, a grow that blows past
    cap*raw_area reverts to raw (runaway guard). uf_min=0 grows everything.
    Returns (chosen {neuron: full mask}, metrics dict). Foreign metrics use the merge/bleed
    rule: `foreign` = total other-neuron nodes engulfed; `bleed_cells` = cells engulfing >=1;
    `new_bleed` = cells that engulf a foreign node ONLY after being grown (leak the fill
    introduced)."""
    chosen, ufs = {}, []
    a_before = a_after = capped = filled = skipped = 0
    foreign = bleed_cells = new_bleed = 0
    raw_foreign = raw_bleed_cells = 0
    for n, r in recs.items():
        a_before += r["raw_area"]
        raw_foreign += r["raw_foreign"]
        raw_bleed_cells += (r["raw_foreign"] > 0)
        grow_it = r["raw_uf"] >= uf_min and r["grown_area"] <= cap * max(r["raw_area"], 1)
        if not grow_it:
            if r["raw_uf"] < uf_min:
                skipped += 1
            else:
                capped += 1
            chosen[n] = r["raw"]; ufs.append(r["raw_uf"]); a_after += r["raw_area"]
            foreign += r["raw_foreign"]; bleed_cells += (r["raw_foreign"] > 0)
        else:
            filled += 1
            chosen[n] = r["grown"]; ufs.append(r["grown_uf"]); a_after += r["grown_area"]
            foreign += r["grown_foreign"]; bleed_cells += (r["grown_foreign"] > 0)
            new_bleed += (r["grown_foreign"] > 0 and r["raw_foreign"] == 0)
    m = dict(uf=float(np.mean(ufs)), a_before=a_before, a_after=a_after,
             capped=capped, filled=filled, skipped=skipped,
             foreign=foreign, bleed_cells=bleed_cells, new_bleed=new_bleed,
             raw_foreign=raw_foreign, raw_bleed_cells=raw_bleed_cells)
    return chosen, m


def contested_px(masks: dict, hw) -> int:
    """Pixels claimed by 2+ neurons: a direct proxy for leak-into-neighbour."""
    count = np.zeros(hw, np.uint16)
    for m in masks.values():
        count[m] += 1
    return int((count >= 2).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--pad", type=int, default=18, help="window pad around each mask, px")
    ap.add_argument("--cap", type=float, default=5.0, help="runaway guard: max area multiple")
    ap.add_argument("--uf-min", type=float, default=0.0,
                    help="only grow cells with raw underfill >= this (0 = grow all)")
    ap.add_argument("--mm-window", type=int, default=0,
                    help="temporal projection window radius in z-slices (0 = single-slice, "
                         "today's behaviour)")
    ap.add_argument("--mm-combine", choices=["median", "mean", "max", "min"], default="median",
                    help="combine statistic across the window (only used when --mm-window > 0)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the runaway cap and the underfill gate")
    ap.add_argument("--sweep-temporal", action="store_true",
                    help="grid a few (window, combine) settings and print the same "
                         "bleed/underfill table as --sweep")
    ap.add_argument("--arbitrate", action="store_true",
                    help="also resolve overlaps with a ridge-aware joint watershed (no overlaps)")
    ap.add_argument("--index", default=str(INDEX_CACHE))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.loads(Path(args.index).read_text())
    lut = do.build_palette(len(idx["neuron_id"]))

    t0 = time.time()
    em, _ = pipeline.load_frame_sam(args.z, scale=SCALE)
    h8, w8 = em.shape[:2]
    em_gray = em.mean(axis=2) if em.ndim == 3 else em
    frames = {args.z: em_gray}
    loaded_z, skipped_z = [], []
    for dz in range(1, args.mm_window + 1):
        for zz in (args.z - dz, args.z + dz):
            try:
                fz, _ = pipeline.load_frame_sam(zz, scale=SCALE)
            except Exception:
                skipped_z.append(zz)
                continue
            frames[zz] = fz.mean(axis=2) if fz.ndim == 3 else fz
            loaded_z.append(zz)
    if args.mm_window > 0:
        print(f"[fill] mm-window frames: loaded {sorted(loaded_z)}, skipped {sorted(skipped_z)}")
    nmasks = do.neuron_masks_at_z(args.z, idx, SCALE, h8, w8)
    if not nmasks:
        print(f"[fill] z={args.z}: no neuron masks"); return
    print(f"[fill] z={args.z}: {len(nmasks)} neurons; growing once, uncapped (pad {args.pad}) ...")

    nbz = nodes_by_z(load_node_table(), SCALE)
    nodes = nbz.get(args.z, [])
    n_cells = len({c for _x, _y, c, *_ in nodes})
    print(f"[fill] {len(nodes)} skeleton nodes at z={args.z} ({n_cells} distinct cells), "
          f"foreign radius {DEFAULT_RADIUS}")

    raw, _c, order = do.stack_labelmap(nmasks, idx["neuron_id"], h8, w8)
    recs = grow_all(nmasks, frames, args.z, args.mm_window, args.mm_combine, args.pad,
                    nodes, DEFAULT_RADIUS)
    raw_uf = float(np.mean([r["raw_uf"] for r in recs.values()]))
    raw_foreign = sum(r["raw_foreign"] for r in recs.values())
    raw_bleed = sum(r["raw_foreign"] > 0 for r in recs.values())
    print(f"[fill] grew {len(recs)} neurons in {time.time()-t0:.0f}s; raw mean underfill {raw_uf:.3f}, "
          f"raw foreign nodes {raw_foreign} in {raw_bleed} cells")

    # (a) sweeps: does a tighter cap, or gating on underfill, cut the FOREIGN-NODE bleed?
    if args.sweep:
        print("\n[sweep cap, grow all]  cap    mean_uf   area%   foreign  bleed_cells  new_bleed  contested")
        for cap in (1.25, 1.5, 2.0, 3.0, 5.0, 1e9):
            ch, m = apply_cap(recs, cap)
            cont = contested_px(ch, (h8, w8))
            tag = "inf" if cap > 1e8 else f"{cap:>4.2f}"
            print(f"[sweep]  {tag:>5}  {m['uf']:6.3f}  {100*(m['a_after']-m['a_before'])/max(1,m['a_before']):+5.0f}%  "
                  f"{m['foreign']:>6}   {m['bleed_cells']:>3}/{len(recs)}     {m['new_bleed']:>4}     {cont:>7}")
        print(f"\n[sweep underfill gate, cap {args.cap}x]  uf_min  filled  mean_uf   area%   foreign  bleed_cells  new_bleed  contested")
        for um in (0.0, 0.4, 0.5, 0.6, 0.7):
            ch, m = apply_cap(recs, args.cap, um)
            cont = contested_px(ch, (h8, w8))
            print(f"[sweep]  {um:>5.2f}   {m['filled']:>3}/{len(recs)}  {m['uf']:6.3f}  "
                  f"{100*(m['a_after']-m['a_before'])/max(1,m['a_before']):+5.0f}%  {m['foreign']:>6}   "
                  f"{m['bleed_cells']:>3}/{len(recs)}     {m['new_bleed']:>4}     {cont:>7}")
        print(f"\n[sweep]  baseline raw: foreign {raw_foreign}, bleed_cells {raw_bleed}/{len(recs)}\n")

    if args.sweep_temporal:
        max_w = 2
        frames_wide = dict(frames)
        loaded_w, skipped_w = [], []
        for dz in range(args.mm_window + 1, max_w + 1):
            for zz in (args.z - dz, args.z + dz):
                if zz in frames_wide:
                    continue
                try:
                    fz, _ = pipeline.load_frame_sam(zz, scale=SCALE)
                except Exception:
                    skipped_w.append(zz)
                    continue
                frames_wide[zz] = fz.mean(axis=2) if fz.ndim == 3 else fz
                loaded_w.append(zz)
        print(f"[fill] sweep-temporal widening frames: loaded {sorted(loaded_w)}, "
              f"skipped {sorted(skipped_w)}")
        print("\n[sweep temporal]  window  combine   filled   mean_uf   area%   foreign  "
              "bleed_cells  new_bleed  contested")
        for w in (0, 1, 2):
            combos = ("median",) if w == 0 else ("median", "mean", "max")
            for combine in combos:
                if w == 0 and combine == "median" and args.mm_window == 0:
                    # identical to the already-computed baseline pass above (window=0 only
                    # ever reads frames[center_z], regardless of which frames dict is passed
                    # in), so reuse it instead of recomputing all 117 neurons from scratch.
                    recs_t = recs
                else:
                    recs_t = grow_all(nmasks, frames_wide, args.z, w, combine, args.pad,
                                      nodes, DEFAULT_RADIUS)
                ch, m = apply_cap(recs_t, args.cap, args.uf_min)
                cont = contested_px(ch, (h8, w8))
                print(f"[sweept]  {w:>4}  {combine:>8}  {m['filled']:>3}/{len(recs_t)}  "
                      f"{m['uf']:6.3f}  "
                      f"{100*(m['a_after']-m['a_before'])/max(1,m['a_before']):+5.0f}%  "
                      f"{m['foreign']:>6}   {m['bleed_cells']:>3}/{len(recs_t)}     "
                      f"{m['new_bleed']:>4}     {cont:>7}")
        print()

    # chosen fill at the requested cap + underfill gate
    chosen, m = apply_cap(recs, args.cap, args.uf_min)
    uf_after, a_before, a_after, capped, n_fill, n_skip = (
        m["uf"], m["a_before"], m["a_after"], m["capped"], m["filled"], m["skipped"])
    filled, _c2, _o2 = do.stack_labelmap(chosen, idx["neuron_id"], h8, w8)
    added = (filled > 0) & (raw == 0)
    cont_stacked = contested_px(chosen, (h8, w8))
    print(f"[fill] cap {args.cap}x, uf_min {args.uf_min}: underfill {raw_uf:.3f} -> {uf_after:.3f}, "
          f"area +{100*(a_after-a_before)/max(1,a_before):.0f}%, grew {n_fill}, capped {capped}, "
          f"skipped-lowuf {n_skip}; foreign nodes {raw_foreign} -> {m['foreign']} "
          f"(bleed cells {m['bleed_cells']}/{len(recs)}, {m['new_bleed']} new from fill), "
          f"{cont_stacked} contested px")

    # (b) fill-then-arbitrate: ridge-aware joint watershed over the grown masks, seeds from
    # the trusted raw-mask centroids -> an overlap-free partition of the grown union.
    arb = None
    if args.arbitrate:
        ta = time.time()
        mem_full = mb.membrane_map(em_gray.astype(np.float32))
        names = list(chosen)
        lab = resolve_overlaps_watershed([chosen[n] for n in names],
                                         [recs[n]["seed"] for n in names], mem_full)
        arb = np.zeros((h8, w8), np.uint16)
        for i, n in enumerate(names):
            arb[lab == i + 1] = idx["neuron_id"][n]
        print(f"[arb] joint ridge-watershed removed all {cont_stacked} contested px "
              f"({time.time()-ta:.0f}s); overlap-free partition of the grown union")

    # render
    raw_rgb = do.colorize_over_em(em, raw, lut, alpha=0.5)
    filled_rgb = do.colorize_over_em(em, filled, lut, alpha=0.5)
    if arb is not None:
        arb_rgb = do.colorize_over_em(em, arb, lut, alpha=0.5)
        panels = [(raw_rgb, f"raw per-slice merge  (z={args.z}, {len(nmasks)} neurons)\n"
                            f"mean underfill {raw_uf:.2f}"),
                  (filled_rgb, f"fill, stacked  (higher-id wins)\n"
                               f"underfill {uf_after:.2f}, {cont_stacked} overlap px"),
                  (arb_rgb, "fill, ridge-watershed arbitrated\noverlap-free partition")]
        fname = f"dense_fill_arbitrated_z{args.z}.png"
    else:
        add_rgb = np.stack([em_gray.astype(np.uint8)] * 3, axis=2)
        add_rgb[added] = (0, 230, 0)
        panels = [(raw_rgb, f"raw per-slice merge  (z={args.z}, {len(nmasks)} neurons)\n"
                            f"mean underfill {raw_uf:.2f}"),
                  (filled_rgb, f"after membrane fill  (grow to ridge walls)\n"
                               f"underfill {uf_after:.2f}, area "
                               f"+{100*(a_after-a_before)/max(1,a_before):.0f}%, {capped} capped"),
                  (add_rgb, "pixels the fill ADDED (green)\nleaks across weak membranes show here")]
        fname = f"dense_fill_z{args.z}.png"

    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.8))
    for a, (img, title) in zip(ax, panels):
        a.imshow(img); a.set_title(title, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Membrane autofill on a whole dense frame", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out_dir / fname
    fig.savefig(f, dpi=130); plt.close(fig)
    print(f"[fill] wrote {f}")


if __name__ == "__main__":
    main()
