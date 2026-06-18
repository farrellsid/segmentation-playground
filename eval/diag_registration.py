"""diag_registration.py — structural + visual check of the skel→GT registration.

`registration.py` reports the headline on-mask rate; this answers the *next* question:
**is the leftover residual structured or irreducible?** That decides the fix:

* If a per-section **full affine** (rotation/scale/shear, not just the current per-section
  *translation*) cuts the residual a lot, the realignment carries per-section rotation/scale the
  translation-only model misses → upgrade the registration model.
* If it doesn't, the residual is noise / single-process-centroid ambiguity, and the ~half-of-nodes
  miss is just neurite thinness (~20-40 px wide) vs the residual (~20 px). No amount of global
  re-fitting helps, and the lever is elsewhere (snap-to-neurite seeding, neighborhood label
  sampling for ERL, or full-res which shrinks both in px).

Also writes overlay montages (EM + GT segment + registered skeleton nodes) so the transform can be
eyeballed across many (neuron, z) without the GUI — the fast first cut before wiring gui.py.

Run:  py -3 -u -m eval.diag_registration            # uses config GT + skeletons_p280
Outputs console stats + PNGs under data/groundtruth/reg_diag/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from sam2_utils import config
from sam2_utils.skeletons import normalize_name
from .groundtruth import GroundTruth
from .registration import Registration, _collect_correspondences


def _per_slice_affine_resid(S, G, z):
    """For each slice with >=4 correspondences, fit S->G full affine on THAT slice alone and
    return its residual magnitudes — the 'best a per-section affine could do' lower bound."""
    out = []
    for zz in np.unique(z):
        m = z == zz
        if m.sum() < 4:
            continue
        s, g = S[m], G[m]
        Sa = np.hstack([s, np.ones((len(s), 1))])          # (n,3)
        M, *_ = np.linalg.lstsq(Sa, g, rcond=None)         # (3,2)
        pred = Sa @ M
        out.append(np.linalg.norm(pred - g, axis=1))
    return np.concatenate(out) if out else np.array([])


def residual_analysis(gt: GroundTruth, sk: pd.DataFrame, reg: Registration) -> pd.DataFrame:
    corr = _collect_correspondences(gt, sk, label_col="_label")
    if corr.empty:
        print("[diag] no correspondences collected"); return corr
    S = corr[["sx", "sy"]].to_numpy()
    G = corr[["gx", "gy"]].to_numpy()
    z = corr["z"].to_numpy().astype(int)

    pred = np.vstack([reg.transform(S[i:i + 1], int(z[i]))[0] for i in range(len(S))])
    res = G - pred
    mag = np.linalg.norm(res, axis=1)
    corr = corr.assign(rx=res[:, 0], ry=res[:, 1], rmag=mag)

    print(f"\n=== residual after current model (global A + per-section translation) ===")
    print(f"  correspondences: {len(corr)}  over {len(np.unique(z))} slices")
    print(f"  |residual| px:  mean {mag.mean():.1f}  median {np.median(mag):.1f}  "
          f"p90 {np.percentile(mag,90):.1f}  max {mag.max():.1f}")

    # (1) per-slice coherence: do neurons on the same slice share a residual direction?
    #     high shared/total => an uncorrected per-section *translation* (cheap to fix);
    #     low => residual is per-neuron/local (translation can't fix it).
    shared, total = [], []
    for zz in np.unique(z):
        m = z == zz
        if m.sum() < 2:
            continue
        r = res[m]
        shared.append(np.linalg.norm(r.mean(0)))   # coherent part
        total.append(np.linalg.norm(r, axis=1).mean())
    if total:
        frac = np.array(shared) / np.maximum(np.array(total), 1e-6)
        print(f"  per-slice coherence (|mean res| / mean|res|): "
              f"median {np.median(frac):.2f}  (hi=>missing per-section shift, lo=>local/noise)")

    # (2) does a per-section FULL AFFINE do better? (rotation/scale the translation model omits)
    aff = _per_slice_affine_resid(S, G, z)
    if aff.size:
        # restrict the current-model comparison to the same multi-corr slices for fairness
        multi = np.isin(z, [zz for zz in np.unique(z) if (z == zz).sum() >= 4])
        cur = mag[multi]
        print(f"  on slices with >=4 corr (n={cur.size}):")
        print(f"     current model    median |res| = {np.median(cur):.1f} px")
        print(f"     per-section affine median |res| = {np.median(aff):.1f} px")
        gain = (np.median(cur) - np.median(aff)) / max(np.median(cur), 1e-6)
        verdict = ("PER-SECTION AFFINE WOULD HELP (residual is structured)"
                   if gain > 0.25 else
                   "affine barely helps -> residual is irreducible (noise/thinness), not a model gap")
        print(f"     => {gain:+.0%} median residual change.  {verdict}")

    # (3) drift vs z
    if len(np.unique(z)) > 5:
        cz = np.corrcoef(z, mag)[0, 1]
        print(f"  |residual| vs z correlation: {cz:+.2f} (drift trend across the stack)")
    return corr


def montage(gt: GroundTruth, sk: pd.DataFrame, reg: Registration, corr: pd.DataFrame,
            out_dir: Path, *, n_neurons: int = 6, n_z: int = 4) -> None:
    if corr.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # pick neurons with a GT segment and the widest z-extent (most to show)
    labs = []
    for lab in sk["_label"].unique():
        nrs = gt.nr_for_label(lab)
        if not nrs:
            continue
        s = sk[sk["_label"] == lab]
        zspan = s["z"].astype(int)
        if zspan.nunique() >= n_z:
            labs.append((lab, int(zspan.max() - zspan.min()), nrs))
    labs.sort(key=lambda t: -t[1])
    labs = labs[:n_neurons]
    if not labs:
        print("[diag] no neurons with enough z-extent for montage"); return

    fig, axes = plt.subplots(len(labs), n_z, figsize=(3.2 * n_z, 3.2 * len(labs)))
    axes = np.atleast_2d(axes)
    for r, (lab, _span, nrs) in enumerate(labs):
        s = sk[sk["_label"] == lab].dropna(subset=["x", "y", "z"])
        zs = sorted(set(s["z"].astype(int)) & set(gt.slice_indices))
        picks = [zs[int(i)] for i in np.linspace(0, len(zs) - 1, n_z)] if len(zs) >= n_z else zs
        for c in range(n_z):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(picks):
                continue
            z = picks[c]
            try:
                em = gt.em_slice(z); gtm = np.isin(gt.label_slice(z), nrs)
            except OSError:
                continue
            nodes = s[s["z"].astype(int) == z][["x", "y"]].to_numpy()
            xy = reg.transform(nodes, z) if len(nodes) else np.empty((0, 2))
            # crop around GT segment (fallback to node bbox) for visibility
            ys, xs = np.nonzero(gtm)
            if xs.size:
                cx0, cx1, cy0, cy1 = xs.min(), xs.max(), ys.min(), ys.max()
            elif len(xy):
                cx0, cx1 = xy[:, 0].min(), xy[:, 0].max(); cy0, cy1 = xy[:, 1].min(), xy[:, 1].max()
            else:
                continue
            pad = 60
            H, W = gtm.shape
            x0, x1 = max(0, int(cx0 - pad)), min(W, int(cx1 + pad))
            y0, y1 = max(0, int(cy0 - pad)), min(H, int(cy1 + pad))
            emc = em[y0:y1, x0:x1]
            ax.imshow(emc, cmap="gray")
            # GT segment outline in green
            gm = gtm[y0:y1, x0:x1]
            if gm.any():
                ax.contour(gm, levels=[0.5], colors="lime", linewidths=0.8)
            # registered nodes in red
            if len(xy):
                ax.scatter(xy[:, 0] - x0, xy[:, 1] - y0, s=10, c="red", marker="x", linewidths=0.8)
            on = 0
            if len(xy):
                xc = xy[:, 0].round().astype(int); yc = xy[:, 1].round().astype(int)
                ok = (xc >= 0) & (xc < W) & (yc >= 0) & (yc < H)
                on = int(gtm[yc[ok], xc[ok]].sum()) if ok.any() else 0
            ax.set_title(f"{lab} z{z}  {on}/{len(xy)} on", fontsize=7)
    fig.suptitle("Registered skeleton nodes (red ×) vs GT segment (green) on GT EM", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = out_dir / "registration_montage.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[diag] montage -> {p}")


def main() -> None:
    base = config.DATA_DIR / "groundtruth" / "skeletons_p280"
    ap = argparse.ArgumentParser(description="Structural + visual registration diagnostic.")
    ap.add_argument("--skeleton-csv", type=Path, default=base / "aggregate_data_pv.csv")
    ap.add_argument("--registration", type=Path, default=base / "registration.json")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "groundtruth" / "reg_diag")
    args = ap.parse_args()

    gt = GroundTruth.from_config()
    reg = Registration.from_json(args.registration)
    sk = pd.read_csv(args.skeleton_csv)
    sk["_label"] = sk["cell_name"].map(normalize_name)

    corr = residual_analysis(gt, sk, reg)
    montage(gt, sk, reg, corr, args.out)


if __name__ == "__main__":
    main()
