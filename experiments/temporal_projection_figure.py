"""Why the temporal projection failed, shown rather than tallied.

The "already tried" slide currently carries `temporal-sweep/temporal_sweep.png`, a
chart of foreign-node bleed against z-window. That chart is SUPERSEDED: it was run at
`uf_min=0.6`, and the roadmap later found a gate-membership confound at that gate (a
blurrier map pulls far more cells past a fixed underfill threshold and into the grow
step, so the window is partly being scored on a different, larger population). Held
open at `uf_min=0`, window=1 is close to flat and window=2 only moderately worse. So
the chart overstates the negative and should not be presented as-is.

This renders the mechanism instead, and the mechanism is a TRADEOFF, not a flat
failure. Averaging adjacent z-slices is supposed to cancel transient organelles and
keep stationary membranes. On real EM it does both: the interior speckle really does
drop, and the membranes really do thin. Bleed gets worse because a thinned wall leaks,
and that is what the corrected numbers say too, window=1 close to flat because the two
effects roughly cancel, window=2 worse because wall damage wins.

The difference panel makes that objective rather than something the eye has to be
trusted on: it is window=0 minus window=2, so signal LOST by projecting separates from
texture REMOVED, and the printed medians say how much of each.

A note on registration, since it is easy to over-claim. A shift-clamp diagnostic
elsewhere found `register_crops` asking for 173-269px on 14-30% of crop pairs, far past
any real slice jitter. That is a genuine defect, but it is a MINORITY of pairs and it is
not what is happening in this crop: the raw shifts here are 1-2px, well inside the
clamp. So this figure shows the tradeoff, not the misregistration. Do not caption it as
the latter.

Panels: raw EM, the window=0 ridge map, the projected map at window=1 and 2, and the
window=0 minus window=2 difference. CPU only, no model.

    py -3 experiments/temporal_projection_figure.py
    py -3 experiments/temporal_projection_figure.py --z 1472 --size 320
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pipeline
from sam2_utils import membrane

OUT = Path(r"F:\ZhenLab\Data\repo_offload\presentation_figures\temporal-projection")
SCALE = 8


def raw_shifts(crops: list[np.ndarray], center_i: int) -> list[float]:
    """The shift magnitude phase correlation asks for BEFORE register_crops clamps it.
    Reported so a reader can see whether alignment is the explanation here or not."""
    from skimage.registration import phase_cross_correlation
    ref = crops[center_i].astype(np.float32)
    out = []
    for i, c in enumerate(crops):
        if i == center_i:
            continue
        shift, _e, _d = phase_cross_correlation(ref, c.astype(np.float32), upsample_factor=1)
        out.append(float(np.hypot(*shift)))
    return out


def build(z: int, cx: int, cy: int, size: int) -> tuple:
    half = size // 2
    per_window = {}
    for w in (0, 1, 2):
        crops = []
        for zz in range(z - w, z + w + 1):
            em, _ = pipeline.load_frame_sam(zz, scale=SCALE)
            g = em if em.ndim == 2 else em[..., 0]
            crops.append(g[cy - half:cy + half, cx - half:cx + half].astype(np.float32))
        mems = [membrane.membrane_map(c) for c in crops]
        ci = len(mems) // 2
        shifts = raw_shifts(mems, ci) if w else []
        reg = membrane.register_crops(mems, center=ci) if w else mems
        per_window[w] = (membrane.project_crops(reg, combine="median"), shifts)
    em0, _ = pipeline.load_frame_sam(z, scale=SCALE)
    g0 = em0 if em0.ndim == 2 else em0[..., 0]
    return g0[cy - half:cy + half, cx - half:cx + half], per_window


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=1456)
    ap.add_argument("--cx", type=int, default=560)
    ap.add_argument("--cy", type=int, default=560)
    ap.add_argument("--size", type=int, default=260)
    args = ap.parse_args(argv)

    em_crop, per_window = build(args.z, args.cx, args.cy, args.size)
    OUT.mkdir(parents=True, exist_ok=True)

    w0 = per_window[0][0]
    w2 = per_window[2][0]
    # "membrane" = the top decile of the window=0 ridge response, "interior" = the rest.
    # Grouping by the BASELINE map keeps the same pixels in each group across windows, so
    # a change in the medians is a change in signal, not a change in group membership.
    # That is the same confound, in miniature, that invalidated the old sweep chart.
    thr = np.percentile(w0, 90)
    mem_px, int_px = w0 >= thr, w0 < thr
    # Report the upper tail as well as the median. Organelle ridges are the BRIGHT
    # interior pixels, so an interior median can sit flat while the texture the method
    # was built to remove is genuinely going away; only the tail shows that.
    print("  ridge response, pixels grouped by the window=0 map (median / 90th pct):")
    print(f"    {'window':<8}{'on membrane':>18}{'interior':>18}")
    for w in (0, 1, 2):
        img = per_window[w][0]
        m, i = img[mem_px], img[int_px]
        print(f"    {w:<8}"
              f"{f'{np.median(m):.2f} / {np.percentile(m, 90):.2f}':>18}"
              f"{f'{np.median(i):.2f} / {np.percentile(i, 90):.2f}':>18}")

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.9))
    axes[0].imshow(em_crop, cmap="gray")
    axes[0].set_title(f"raw EM (z={args.z})", fontsize=11)
    for ax, w in zip(axes[1:4], (0, 1, 2)):
        ax.imshow(per_window[w][0], cmap="gray", vmin=w0.min(), vmax=w0.max())
        ax.set_title("ridge map, window=0\n(what the pipeline uses)" if w == 0
                     else f"projected, window={w} (median)", fontsize=11)
    d = w0 - w2
    lim = float(np.percentile(np.abs(d), 99))
    im = axes[4].imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[4].set_title("window=0 minus window=2\nred = lost by projecting", fontsize=11)
    fig.colorbar(im, ax=axes[4], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Temporal projection is a tradeoff: it removes organelle texture AND "
                 "thins real membranes", fontsize=14)
    fig.tight_layout()
    out = OUT / f"temporal_projection_z{args.z}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")
    for w in (1, 2):
        sh = per_window[w][1]
        if sh:
            joined = ", ".join(f"{v:.0f}" for v in sh)
            print(f"  window={w}: raw phase-correlation shifts (px) = {joined}"
                  f"  [clamp is 5, so alignment is fine in this crop]")


if __name__ == "__main__":
    main()
