"""Presentation figure: dense per-frame segmentation and how overlapping neuron masks
get resolved. LOCAL only, no Narval, no commit.

One tight crop from the dense nerve-ring band (z=1456 by default), shown as a set of
panels for a slide:

  1. Raw EM crop.
  2. Sato ridge / membrane map on that crop (sam2_utils.membrane.membrane_map): membranes
     light up as dark ridges, i.e. the walls between neurons.
  3. Dense segmentation: every SAM3 per-slice neuron mask that lands in the crop, merged
     into one labelmap (experiments/dense_overlay), each neuron a distinct colour.
  4. Overlap resolution: the SAME set of overlapping neuron masks resolved two ways,
     resolve_overlaps_argmax vs resolve_overlaps_watershed. Watershed takes the ridge map
     as its elevation, so it splits touching neurons along real membranes; argmax draws an
     arbitrary nearest-seed line. Pixels the two resolvers disagree on are outlined so the
     difference is visible.
  5. (nice-to-have) Automask: SAM2AutomaticMaskGenerator run on the crop, if a model run is
     available. Omitted with --no-amg or if the model cannot be built.

Nothing here trains or evaluates; it reuses the cached dense-overlay index and the CPU
membrane filter, plus one optional short AMG model run on the crop.

    py -3 experiments/dense_overlaps_figure.py            # z=1456, AMG on if GPU/model available
    py -3 experiments/dense_overlaps_figure.py --no-amg   # panels 1-4 only
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
from matplotlib.patches import Patch

import pipeline
from experiments import dense_overlay as do
from sam2_utils import membrane as mb
from sam2_utils.perframe import resolve_overlaps_argmax, resolve_overlaps_watershed

INDEX_CACHE = Path("docs/figures/sam3-bakeoff/dense-overlay/_index.json")
OUT_DIR = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\dense-overlaps")
SCALE = 8


# ---------------------------------------------------------------------------
# small render helpers
# ---------------------------------------------------------------------------
def gray_rgb(em_crop: np.ndarray) -> np.ndarray:
    """Grayscale EM crop as an RGB uint8 backdrop."""
    g = em_crop.mean(axis=2) if em_crop.ndim == 3 else em_crop
    g = g.astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def label_rgb_over_em(em_rgb, label, id_to_rgb, alpha=0.55):
    """Colour a labelmap (id -> rgb) over the EM backdrop."""
    base = em_rgb.astype(np.float32)
    out = base.copy()
    fg = label > 0
    col = np.zeros_like(base)
    for lid, rgb in id_to_rgb.items():
        col[label == lid] = rgb
    out[fg] = (1 - alpha) * base[fg] + alpha * col[fg]
    return out.astype(np.uint8)


def outline(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """`width`-px ring hugging the boundary of a boolean region, for marking where the two
    resolvers disagree without hiding the actual per-neuron assignment underneath."""
    from scipy import ndimage as ndi
    d = ndi.binary_dilation(mask, iterations=width)
    e = ndi.binary_erosion(mask, iterations=width)
    return d & ~e


# ---------------------------------------------------------------------------
# data assembly for the crop
# ---------------------------------------------------------------------------
def build_crop(z: int, r: int, c: int, W: int, idx: dict):
    """Load frame z, crop to [r:r+W, c:c+W], and return the crop EM plus per-neuron masks
    (full-frame bool sliced to the crop). Only neurons with any pixel in the crop kept."""
    em, _full_hw = pipeline.load_frame_sam(z, scale=SCALE)
    h8, w8 = em.shape[:2]
    nm_full = do.neuron_masks_at_z(z, idx, SCALE, h8, w8)
    _raw, _count, order = do.stack_labelmap(nm_full, idx["neuron_id"], h8, w8)

    em_crop = em[r:r + W, c:c + W]
    crop_masks = {}
    for n in order:
        m = nm_full[n][r:r + W, c:c + W]
        if m.any():
            crop_masks[n] = m
    return em_crop, crop_masks


def masks_and_seeds(neurons, crop_masks):
    """Ordered mask list + centroid seeds (x, y) for the given neuron names.

    The seed stands in for the skeleton node: argmax sends a contested pixel to the nearest
    seed, watershed uses each seed as a marker. Both resolvers here get the SAME seeds, so
    any difference in the output comes only from the rule (nearest-seed vs membrane walls),
    not from the seeds."""
    masks, seeds, kept = [], [], []
    for n in neurons:
        m = crop_masks[n]
        ys, xs = np.where(m)
        masks.append(m)
        seeds.append((float(xs.mean()), float(ys.mean())))
        kept.append(n)
    return masks, seeds, kept


# ---------------------------------------------------------------------------
# optional AMG panel
# ---------------------------------------------------------------------------
def try_amg(em_crop: np.ndarray, model_size: str = "tiny"):
    """SAM2AutomaticMaskGenerator on the crop. Returns a list of bool masks, or None if a
    model run is not available (no torch/checkpoint, build failure). Kept short: modest
    points_per_side, no crop layers, so a single small crop finishes in seconds on GPU."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2_utils import setup
        import torch

        t0 = time.time()
        device = setup.setup_device()
        ckpt, cfg = setup.ensure_checkpoint(model_size)
        model = build_sam2(cfg, str(ckpt), device=device)
        amg = SAM2AutomaticMaskGenerator(
            model=model, points_per_side=32, points_per_batch=64,
            pred_iou_thresh=0.7, stability_score_thresh=0.9,
            crop_n_layers=0, min_mask_region_area=15,
        )
        rgb = em_crop if em_crop.ndim == 3 else np.stack([em_crop] * 3, axis=2)
        with torch.inference_mode():
            anns = amg.generate(np.ascontiguousarray(rgb.astype(np.uint8)))
        masks = [np.asarray(a["segmentation"]).astype(bool) for a in anns]
        print(f"[fig] AMG produced {len(masks)} masks in {time.time() - t0:.1f}s "
              f"on {device}")
        return masks
    except Exception as e:  # noqa: BLE001 - AMG is nice-to-have, never blocks the figure
        print(f"[fig] AMG skipped: {e!r}")
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--row", type=int, default=340)
    ap.add_argument("--col", type=int, default=580)
    ap.add_argument("--size", type=int, default=180, help="square crop side, _sam px")
    ap.add_argument("--index", default=str(INDEX_CACHE))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-amg", action="store_true", help="skip the automask panel")
    ap.add_argument("--tau", type=float, default=mb.DEFAULT_TAU,
                    help="membrane threshold for the watershed elevation display")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = json.loads(Path(args.index).read_text())
    z, r, c, W = args.z, args.row, args.col, args.size

    print(f"[fig] z={z} crop rows {r}-{r + W} cols {c}-{c + W} (scale {SCALE})")
    em_crop, crop_masks = build_crop(z, r, c, W, idx)
    em_bg = gray_rgb(em_crop)
    gray = em_crop.mean(axis=2) if em_crop.ndim == 3 else em_crop
    mem = mb.membrane_map(gray)

    # neuron colours, consistent across the dense and resolution panels
    palette = do.build_palette(len(idx["neuron_id"]))
    id_to_rgb = {idx["neuron_id"][n]: tuple(int(v) for v in palette[idx["neuron_id"][n]])
                 for n in crop_masks}

    # panel 3: dense segmentation over every neuron in the crop (raw merge, higher id wins)
    dense_lab = np.zeros((W, W), np.uint16)
    for n in sorted(crop_masks, key=lambda k: idx["neuron_id"][k]):
        dense_lab[crop_masks[n]] = idx["neuron_id"][n]

    # which neurons overlap inside the crop: those are the "same set of overlapping masks"
    stack = np.stack([crop_masks[n] for n in crop_masks], axis=0)
    count = stack.sum(axis=0)
    contested = count >= 2
    overl = [n for n in sorted(crop_masks, key=lambda k: idx["neuron_id"][k])
             if (crop_masks[n] & contested).any()]
    print(f"[fig] {len(crop_masks)} neurons in crop, {len(overl)} overlap: {overl}; "
          f"{int(contested.sum())} contested px")

    o_masks, o_seeds, o_names = masks_and_seeds(overl, crop_masks)
    lab_arg = resolve_overlaps_argmax(o_masks, o_seeds, mem)
    lab_ws = resolve_overlaps_watershed(o_masks, o_seeds, mem)
    # map the resolver's index labels (1..k) back to global neuron ids for consistent colour
    arg_ids = np.zeros((W, W), np.uint16)
    ws_ids = np.zeros((W, W), np.uint16)
    for i, n in enumerate(o_names):
        arg_ids[lab_arg == i + 1] = idx["neuron_id"][n]
        ws_ids[lab_ws == i + 1] = idx["neuron_id"][n]
    union = np.zeros((W, W), bool)
    for m in o_masks:
        union |= m
    disagree = (lab_arg != lab_ws) & union
    # thin contour around the disagreement zone: the fill underneath keeps each resolver's
    # TRUE per-neuron colour, so the moved boundary (argmax's straight cut vs watershed's
    # membrane-following one) stays visible; the ring just draws the eye to it.
    diff_ring = outline(disagree, width=1)
    print(f"[fig] argmax vs watershed differ on {int(disagree.sum())} px "
          f"({100 * disagree.sum() / max(1, union.sum()):.1f}% of the overlapping union)")

    # -----------------------------------------------------------------------
    # individual panels (each also a standalone PNG for slide flexibility)
    # -----------------------------------------------------------------------
    def save_img(arr, name, cmap=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(out_dir / name, dpi=140, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    save_img(em_bg, "panel1_em.png")
    save_img(mem, "panel2_ridge.png", cmap="inferno")
    save_img(label_rgb_over_em(em_bg, dense_lab, id_to_rgb), "panel3_dense.png")

    # overlap raw: the overlapping masks coloured, contested pixels marked white
    raw_over = label_rgb_over_em(em_bg, dense_lab * np.isin(
        dense_lab, [idx["neuron_id"][n] for n in overl]), id_to_rgb)
    raw_over[contested] = (255, 255, 255)
    save_img(raw_over, "panel4a_overlap_raw.png")

    arg_img = label_rgb_over_em(em_bg, arg_ids, id_to_rgb)
    ws_img = label_rgb_over_em(em_bg, ws_ids, id_to_rgb)
    arg_img[diff_ring] = (255, 0, 255)
    ws_img[diff_ring] = (255, 0, 255)
    save_img(arg_img, "panel4b_argmax.png")
    save_img(ws_img, "panel4c_watershed.png")

    amg_masks = None if args.no_amg else try_amg(em_crop)
    if amg_masks:
        amg_pal = do.build_palette(len(amg_masks) + 1)
        amg_lab = np.zeros((W, W), np.uint16)
        for i, m in enumerate(sorted(amg_masks, key=lambda a: -int(a.sum()))):
            amg_lab[m] = i + 1
        amg_id_rgb = {i + 1: tuple(int(v) for v in amg_pal[i + 1])
                      for i in range(len(amg_masks))}
        save_img(label_rgb_over_em(em_bg, amg_lab, amg_id_rgb), "panel5_amg.png")

    # -----------------------------------------------------------------------
    # combined montage for the slide
    # -----------------------------------------------------------------------
    ncols = 3
    nrows = 3 if amg_masks else 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    ax = np.atleast_2d(ax)

    def show(a, arr, title, cmap=None):
        a.imshow(arr, cmap=cmap)
        a.set_title(title, fontsize=11)
        a.set_xticks([]); a.set_yticks([])

    show(ax[0, 0], em_bg, "1. Raw EM crop")
    show(ax[0, 1], mem, "2. Sato ridge / membrane map", cmap="inferno")
    show(ax[0, 2], label_rgb_over_em(em_bg, dense_lab, id_to_rgb),
         f"3. Dense segmentation ({len(crop_masks)} neurons)")
    show(ax[1, 0], raw_over,
         f"4a. Overlapping masks ({len(overl)}), contested = white")
    show(ax[1, 1], arg_img, "4b. Resolved: argmax (nearest seed)")
    show(ax[1, 2], ws_img, "4c. Resolved: watershed on ridge map")
    if amg_masks:
        show(ax[2, 0], label_rgb_over_em(em_bg, amg_lab, amg_id_rgb),
             f"5. Automask (SAM2 AMG, {len(amg_masks)} masks)")
        for j in (1, 2):
            ax[2, j].axis("off")

    fig.suptitle(
        f"Dense per-frame segmentation and overlap resolution  (z={z}, "
        f"crop {W}x{W} px at scale {SCALE}; magenta = argmax vs watershed disagree)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "figure_dense_overlaps.png", dpi=140)
    plt.close(fig)

    write_index_html(out_dir, z, r, c, W, crop_masks, overl, contested, disagree,
                     union, amg_masks)
    print(f"[fig] wrote {out_dir / 'figure_dense_overlaps.png'} and panels + index.html")


def write_index_html(out_dir, z, r, c, W, crop_masks, overl, contested, disagree,
                     union, amg_masks):
    pct = 100 * disagree.sum() / max(1, union.sum())
    amg_row = ("" if not amg_masks else
               f"<h3>5. Automask (SAM2 AMG)</h3>"
               f"<p>SAM2AutomaticMaskGenerator run on the crop, {len(amg_masks)} masks, "
               f"each a distinct colour. Class-agnostic: it segments everything it can find, "
               f"with no notion of which blob is which neuron.</p>"
               f"<img src='panel5_amg.png'>")
    html = f"""<!doctype html><meta charset=utf-8>
<title>Dense per-frame segmentation and overlap resolution</title>
<body style="font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:1200px;margin:auto;padding:1em">
<h2>Dense per-frame segmentation and how overlapping neuron masks get resolved</h2>
<p>One crop from the dense nerve-ring band: <b>z={z}</b>, rows {r}-{r + W}, cols {c}-{c + W}
({W}x{W} px on the scale-8 _sam grid). {len(crop_masks)} neuron masks land in this crop;
{len(overl)} of them overlap ({int(contested.sum())} contested pixels).</p>

<img src="figure_dense_overlaps.png" style="width:100%">

<h3>1. Raw EM crop</h3>
<p>The electron-microscopy image, the input to everything else.</p>
<img src="panel1_em.png">

<h3>2. Sato ridge / membrane map</h3>
<p>sam2_utils.membrane.membrane_map: a Sato dark-ridge filter. Cell membranes are dark walls
on bright cytoplasm, so they light up as bright ridges here. This is the boundary evidence the
watershed resolver uses.</p>
<img src="panel2_ridge.png">

<h3>3. Dense segmentation</h3>
<p>Every SAM3 per-slice neuron mask that lands in the crop, merged into one labelmap
(experiments/dense_overlay), one colour per neuron.</p>
<img src="panel3_dense.png">

<h3>4. Overlap resolution: argmax vs watershed</h3>
<p>The SAME {len(overl)} overlapping masks, resolved two ways. <b>4a</b> shows the raw masks
with contested pixels (claimed by 2+ neurons) in white. <b>4b</b> argmax sends each contested
pixel to the nearest seed, an arbitrary straight-ish line through the overlap. <b>4c</b>
watershed floods from the same seeds but uses the ridge map as elevation, so the split follows
real membranes. Pixels the two resolvers assign differently are outlined in magenta:
{int(disagree.sum())} px, {pct:.1f}% of the overlapping union.</p>
<img src="panel4a_overlap_raw.png">
<img src="panel4b_argmax.png">
<img src="panel4c_watershed.png">
{amg_row}
</body>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
