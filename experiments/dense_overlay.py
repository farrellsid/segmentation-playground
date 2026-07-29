"""Merge all SAM3 per-slice neuron masks into ONE dense per-frame labelmap, LOCAL only.

No model runs, no GPU, no Narval. This reads the extracted SAM3 masks from disk, remaps
each chain's crop-space mask back into a common frame at a reduced render scale (scale 8,
the same backdrop the existing overlays use), and stacks them by neuron into a single
uint16 labelmap where the pixel value is the neuron id (0 = background).

Coordinate remap reuses the pattern in experiments/sam3_overlay_disk.py: each chain's
state.json carries a crop_window, we rebuild it as an alignment.CropWindow, and place the
crop mask back into the full frame using the crop's origin_tif/size_tif (the same numbers
CropWindow.slice_tif crops with), divided by the render scale. EM backdrop comes from
pipeline.load_frame_sam(z, scale=8), exactly as the disk overlay fetches it.

Each neuron owns MANY chains (contiguous skeleton runs); every chain of a neuron shares
one neuron id, so a neuron's mask on a frame is the union of its chains' masks there.
Pixels claimed by two or more DIFFERENT neurons are expected (the per-slice masks were cut
independently, one crop per chain). We report those contested pixels and also show an
OPTIONAL arbitration pass (resolve_overlaps_argmax) side by side with the raw merge. We do
NOT claim arbitration is better here: it is shown so the effect is visible, and validating
it against the merge metric is left as a TODO.

Run a small bounded pass first (this is the default):

    py -3 experiments/dense_overlay.py --z-start 1560 --z-end 1600 --stride 4

A full local run over every covered z is then a one-liner (see --help for the exact range).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline
from sam2_utils import alignment, qc
from sam2_utils.perframe import resolve_overlaps_argmax

RE = Path(r"F:\ZhenLab\Data\output_masks\resolution_experiments")
DEFAULT_TREE = RE / "target_perslice_only_guard_sam3_merged"
RENDER_SCALE = 8  # match the EM backdrop the disk overlays use (load_frame_sam scale 8)


# ---------------------------------------------------------------------------
# index: scan every chain ONCE (state.json crop_window + which z it covers) so
# a per-frame merge never has to load a whole chain. Cached to disk as JSON.
# ---------------------------------------------------------------------------
def build_index(tree: Path, cache_path: Path, rebuild: bool = False) -> dict:
    if cache_path.exists() and not rebuild:
        return json.loads(cache_path.read_text())
    records = []           # one per chain
    malformed = []         # (path, reason)
    z_to_recs: dict[str, list[int]] = {}
    neuron_dirs = sorted(d for d in tree.iterdir() if d.is_dir())
    t0 = time.time()
    for nd in neuron_dirs:
        neuron = nd.name
        for cd in sorted(nd.glob("chain_*")):
            sj = cd / "state.json"
            if not sj.exists():
                malformed.append((str(cd), "no state.json"))
                continue
            try:
                st = json.loads(sj.read_text())
                cw = st["crop_window"]
                if cw is None:
                    raise KeyError("crop_window is null")
                _ = (cw["origin_tif"], cw["size_tif"], cw["crop_scale"], cw["sam_scale"])
            except Exception as e:  # malformed / missing crop window: skip, keep counting
                malformed.append((str(cd), repr(e)))
                continue
            zs = [z for z, _p in qc._iter_mask_paths(cd / "masks")]
            if not zs:
                malformed.append((str(cd), "no mask_*.png"))
                continue
            ri = len(records)
            records.append({"neuron": neuron, "chain": cd.name,
                            "masks_dir": str(cd / "masks"), "crop_window": cw, "zs": zs})
            for z in zs:
                z_to_recs.setdefault(str(z), []).append(ri)
    neurons = sorted({r["neuron"] for r in records})
    neuron_id = {n: i + 1 for i, n in enumerate(neurons)}   # 1..N, 0 = background
    idx = {"tree": str(tree), "records": records, "z_to_recs": z_to_recs,
           "neuron_id": neuron_id, "malformed": malformed,
           "scan_seconds": round(time.time() - t0, 1)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(idx))
    return idx


def make_cw(cwd: dict) -> alignment.CropWindow:
    return alignment.CropWindow(origin_tif=tuple(cwd["origin_tif"]),
                                size_tif=tuple(cwd["size_tif"]),
                                crop_scale=int(cwd["crop_scale"]),
                                sam_scale=int(cwd["sam_scale"]))


def frame8_slice(cw: alignment.CropWindow, scale: int, h8: int, w8: int):
    """Where this crop lands in the scale-`scale` full frame. Same origin_tif/size_tif
    CropWindow.slice_tif() uses, divided by the render scale and clipped to the frame."""
    x0, y0 = cw.origin_tif
    w, h = cw.size_tif
    r0, c0 = int(round(y0 / scale)), int(round(x0 / scale))
    r1, c1 = int(round((y0 + h) / scale)), int(round((x0 + w) / scale))
    return max(0, r0), min(h8, r1), max(0, c0), min(w8, c1)


def neuron_masks_at_z(z: int, idx: dict, scale: int, h8: int, w8: int):
    """Union each neuron's chain masks at z into one full-frame bool. Returns
    {neuron: full-frame bool mask}. Reads only the single mask_<z>.png per chain."""
    recs = idx["records"]
    rec_ids = idx["z_to_recs"].get(str(z), [])
    out: dict[str, np.ndarray] = {}
    for ri in rec_ids:
        r = recs[ri]
        mp = Path(r["masks_dir"]) / f"mask_{z}.png"
        if not mp.exists():
            continue
        small = qc._load_binary(mp)                       # crop-space bool (crop_hw)
        cw = make_cw(r["crop_window"])
        r0, r1, c0, c1 = frame8_slice(cw, scale, h8, w8)
        th, tw = r1 - r0, c1 - c0
        if th <= 0 or tw <= 0:
            continue
        rs = cv2.resize(small.astype(np.uint8), (tw, th),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        full = out.get(r["neuron"])
        if full is None:
            full = np.zeros((h8, w8), bool)
            out[r["neuron"]] = full
        full[r0:r1, c0:c1] |= rs
    return {n: m for n, m in out.items() if m.any()}


def stack_labelmap(neuron_masks: dict, neuron_id: dict, h8: int, w8: int):
    """Raw merge: neuron id per pixel, higher id wins on a contest (deterministic).
    Returns (labelmap uint16, count uint8, ordered neuron list)."""
    label = np.zeros((h8, w8), np.uint16)
    count = np.zeros((h8, w8), np.uint16)
    order = sorted(neuron_masks, key=lambda n: neuron_id[n])
    for n in order:
        m = neuron_masks[n]
        label[m] = neuron_id[n]
        count[m] += 1
    return label, count, order


def arbitrate(neuron_masks: dict, neuron_id: dict, order: list, h8: int, w8: int):
    """resolve_overlaps_argmax over the per-neuron masks; contested pixels go to the
    neuron whose mask centroid is nearest (centroid stands in for the skeleton node).
    Returns a uint16 labelmap. resolve_overlaps_watershed is the documented alternative."""
    masks = [neuron_masks[n] for n in order]
    seeds = []
    for m in masks:
        ys, xs = np.where(m)
        seeds.append((float(xs.mean()), float(ys.mean())))   # (x, y)
    lab = resolve_overlaps_argmax(masks, seeds)               # 0 bg, i+1 -> order[i]
    out = np.zeros((h8, w8), np.uint16)
    for i, n in enumerate(order):
        out[lab == i + 1] = neuron_id[n]
    return out


# ---------------------------------------------------------------------------
# colour + render
# ---------------------------------------------------------------------------
def build_palette(n: int) -> np.ndarray:
    """Deterministic distinct RGB per id, 1..n (index 0 = background = black)."""
    lut = np.zeros((n + 1, 3), np.uint8)
    for i in range(1, n + 1):
        h = (i * 0.61803398875) % 1.0            # golden-ratio hue spread
        s, v = 0.65, 1.0
        c = v * s
        x = c * (1 - abs(((h * 6) % 2) - 1))
        m = v - c
        seg = int(h * 6) % 6
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][seg]
        lut[i] = [int((rgb[0] + m) * 255), int((rgb[1] + m) * 255), int((rgb[2] + m) * 255)]
    return lut


def colorize_over_em(em_rgb: np.ndarray, label: np.ndarray, lut: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    base = em_rgb.astype(np.float32)
    col = lut[label].astype(np.float32)         # (H, W, 3)
    fg = label > 0
    out = base.copy()
    out[fg] = (1 - alpha) * base[fg] + alpha * col[fg]
    return out.astype(np.uint8)


def render_pair(em, raw_lab, arb_lab, lut, count, z, n_present, out_png):
    contested = int((count >= 2).sum())
    labeled = int((raw_lab > 0).sum())
    maxclaim = int(count.max()) if count.size else 0
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))
    ax[0].imshow(colorize_over_em(em, raw_lab, lut))
    ax[0].set_title(f"raw merge  z={z}  |  {n_present} neurons, {labeled} px, "
                    f"{contested} contested (max {maxclaim} claimants)", fontsize=9)
    ax[1].imshow(colorize_over_em(em, arb_lab, lut))
    ax[1].set_title(f"argmax-arbitrated  z={z}  |  contested px reassigned to nearest "
                    "centroid", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110); plt.close(fig)
    return {"z": z, "n_neurons": n_present, "labeled_px": labeled,
            "contested_px": contested, "max_claimants": maxclaim}


def process_z(z, idx, scale, lut, out_dir, save_npy=True, compress=False):
    t0 = time.time()
    em, full_hw = pipeline.load_frame_sam(z, scale=scale)
    h8, w8 = em.shape[:2]
    nmasks = neuron_masks_at_z(z, idx, scale, h8, w8)
    if not nmasks:
        return None
    raw, count, order = stack_labelmap(nmasks, idx["neuron_id"], h8, w8)
    arb = arbitrate(nmasks, idx["neuron_id"], order, h8, w8)
    if save_npy:
        if compress:   # ~100x smaller: the labelmap is >90% background zeros
            np.savez_compressed(out_dir / f"labelmap_z{z}.npz", raw=raw, arb=arb)
        else:
            np.save(out_dir / f"labelmap_z{z}_raw.npy", raw)
            np.save(out_dir / f"labelmap_z{z}_arb.npy", arb)
    stat = render_pair(em, raw, arb, lut, count, z, len(nmasks),
                       out_dir / f"dense_z{z}.png")
    stat["seconds"] = round(time.time() - t0, 2)
    stat["raw_npy_bytes"] = int(raw.nbytes)
    return stat


def write_index_html(out_dir: Path, stats: list, idx: dict, args):
    n_neu = len(idx["neuron_id"])
    tot_contested = sum(s["contested_px"] for s in stats)
    tot_labeled = sum(s["labeled_px"] for s in stats)
    frames_with_contest = sum(1 for s in stats if s["contested_px"] > 0)
    html = ["<!doctype html><meta charset=utf-8><title>SAM3 dense per-frame labelmap</title>",
            "<body style='font-family:sans-serif;background:#111;color:#eee;max-width:1400px;margin:auto'>",
            "<h2>SAM3 per-slice masks merged into one dense per-frame labelmap</h2>",
            "<p>Every neuron's per-slice mask remapped from its crop into a common scale-8 "
            "frame and stacked by neuron id (0 = background). Left: raw merge (a pixel claimed "
            "by 2+ neurons keeps the higher-id neuron). Right: optional argmax arbitration "
            "(contested pixels reassigned to the nearest mask centroid). No claim that "
            "arbitration is better here; it is shown so the effect is visible. Validating "
            "against the merge metric is a TODO. No model runs.</p>",
            f"<p><b>{len(stats)} frames</b>, {n_neu} neuron ids. Contested pixels: "
            f"{tot_contested} across {frames_with_contest}/{len(stats)} frames "
            f"({tot_labeled} labeled px total). Render scale {args.scale}.</p>",
            "<table border=1 cellpadding=4 style='border-collapse:collapse;font-size:13px'>",
            "<tr><th>z</th><th>neurons</th><th>labeled px</th><th>contested px</th>"
            "<th>max claimants</th><th>sec</th></tr>"]
    for s in stats:
        html.append(f"<tr><td>{s['z']}</td><td>{s['n_neurons']}</td><td>{s['labeled_px']}</td>"
                    f"<td>{s['contested_px']}</td><td>{s['max_claimants']}</td>"
                    f"<td>{s['seconds']}</td></tr>")
    html.append("</table>")
    for s in stats:
        html.append(f"<h3>z = {s['z']}</h3><img src='dense_z{s['z']}.png' style='max-width:100%'>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=str(DEFAULT_TREE))
    ap.add_argument("--z-start", type=int, default=1560)
    ap.add_argument("--z-end", type=int, default=1600)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--scale", type=int, default=RENDER_SCALE)
    ap.add_argument("--out", default="docs/figures/sam3-bakeoff/dense-overlay")
    ap.add_argument("--cache", default=None, help="index cache json (default: <out>/_index.json)")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--no-npy", action="store_true", help="skip saving labelmap files")
    ap.add_argument("--compress", action="store_true",
                    help="save one compressed labelmap_z<z>.npz (raw+arb) instead of two .npy; "
                         "~100x smaller since the map is mostly background")
    ap.add_argument("--dense-report", action="store_true",
                    help="just print the z coverage histogram (how many neurons per z) and exit")
    args = ap.parse_args(argv)

    tree = Path(args.tree)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out_dir / "_index.json"

    print(f"[dense] building/loading index for {tree.name} ...")
    idx = build_index(tree, cache, rebuild=args.rebuild_index)
    print(f"[dense] {len(idx['records'])} chains, {len(idx['neuron_id'])} neurons, "
          f"{len(idx['malformed'])} malformed/skipped; scan {idx.get('scan_seconds')}s")
    if idx["malformed"]:
        for p, why in idx["malformed"][:10]:
            print(f"        skipped {p}: {why}")

    # z coverage: how many DISTINCT neurons cover each z
    recs = idx["records"]
    z_neurons: dict[int, set] = {}
    for zk, ris in idx["z_to_recs"].items():
        z_neurons[int(zk)] = {recs[ri]["neuron"] for ri in ris}
    if args.dense_report:
        rows = sorted(z_neurons.items(), key=lambda kv: -len(kv[1]))[:40]
        allz = sorted(z_neurons)
        print(f"[dense] z range {allz[0]}..{allz[-1]} ({len(allz)} distinct z)")
        print("[dense] densest z (z: #neurons):")
        for z, ns in rows:
            print(f"        {z}: {len(ns)}")
        return

    zs = list(range(args.z_start, args.z_end + 1, args.stride))
    lut = build_palette(len(idx["neuron_id"]))
    (out_dir / "_legend.json").write_text(json.dumps(idx["neuron_id"], indent=0))

    stats = []
    for z in zs:
        if z not in z_neurons:
            print(f"[dense] z={z}: no neuron masks; skip"); continue
        s = process_z(z, idx, args.scale, lut, out_dir, save_npy=not args.no_npy,
                      compress=args.compress)
        if s is None:
            print(f"[dense] z={z}: empty after remap; skip"); continue
        stats.append(s)
        print(f"[dense] z={z}: {s['n_neurons']} neurons, {s['labeled_px']} px, "
              f"{s['contested_px']} contested, {s['seconds']}s, "
              f"raw npy {s['raw_npy_bytes']//1024} KiB")

    if stats:
        write_index_html(out_dir, stats, idx, args)
        tot = sum(s["seconds"] for s in stats)
        avg_npy = np.mean([s["raw_npy_bytes"] for s in stats]) / 1024
        print(f"\n[dense] {len(stats)} frames in {tot:.1f}s "
              f"({tot/len(stats):.2f}s/frame), avg raw labelmap {avg_npy:.0f} KiB")
        print(f"[dense] viewer -> {out_dir/'index.html'}")
    else:
        print("[dense] no frames rendered")


if __name__ == "__main__":
    main()
