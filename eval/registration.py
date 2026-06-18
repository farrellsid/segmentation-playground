"""
registration.py — align a worm's CATMAID skeleton to its VAST GT pixel grid.

The cross-worm GT (project 280) was traced on the *pre-realignment* stack, while
the VAST masks/EM live in the ``realigned_with_blur`` stack — re-aligned
**section by section**. So skeleton stack-px → VAST mask-px is NOT one global
affine: the linear part is just the ~¼ downscale (full-res trace px == full-res
VAST px), but each z-slice carries its own small translation from the realignment.

This module fits that model

    mask_xy(z) = skel_xy @ A.T + offset[z]

in two stages, using the GT masks themselves as the alignment target (no manual
landmarks):

1. **Global linear A** — pooled least-squares of *z-centered* correspondences
   (subtract each slice's mean from both sides), so per-section translation can't
   contaminate the linear estimate. Comes out ≈ (1/downscale)·I.
2. **Per-section offset[z]** — for each slice, the robust (median) residual
   ``gt_centroid - skel_centroid @ A.T`` over the neurons present on that slice;
   slices with too few clean correspondences are linearly interpolated from their
   neighbours and the whole curve is lightly smoothed (realignment shifts vary
   smoothly in z).

A "correspondence" is one (neuron, z) where the neuron's GT segment is a single
connected component and its skeleton nodes on that slice are spatially tight (one
process crossing) — so the two centroids refer to the same thing.

Reusable for any worm: pass a :class:`eval.groundtruth.GroundTruth`, the skeleton
``aggregate_data_pv.csv``, and (optionally) a name-normalizer. Torch-free; needs
scipy.ndimage for connected components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage

from sam2_utils.skeletons import normalize_name
from .groundtruth import GroundTruth


@dataclass
class Registration:
    """skel-stack-px → GT-mask-px transform, per z-section.

    Two models share one ``transform(xy, z)`` API and JSON:

    * **per-section affine** (preferred; set ``affines``) — ``mask_xy = xy @ L[z].T
      + t[z]`` with a full 2×2 linear ``L[z]`` *and* translation ``t[z]`` per slice.
      ``affines`` is ``(n_z, 2, 3)``: ``affines[i, :, :2]`` is ``L`` (in the ``A.T``
      convention) and ``affines[i, :, 2]`` is ``t``, indexed by ``z_min + i``. This
      captures the per-section rotation/scale the realignment carries, which a single
      global linear + per-section *translation* misses (diag: median residual
      19.6 px → 5.1 px).
    * **global linear + per-section translation** (legacy / baseline; ``affines is
      None``) — ``mask_xy = xy @ A.T + offset[z]``.

    ``A`` (2,2) and ``offsets`` (n_z,2) are always kept: they are the legacy model,
    the baseline constructor's inputs, and a human-readable summary of the affine
    model's mean linear part.
    """
    A: np.ndarray
    offsets: np.ndarray          # (n_z, 2), row i = offset for z = z_min + i
    z_min: int
    affines: Optional[np.ndarray] = None   # (n_z, 2, 3) per-section affine, if fitted

    def transform(self, xy: np.ndarray, z: int) -> np.ndarray:
        """Map (N,2) skeleton stack-px at slice ``z`` to GT mask px."""
        xy = np.asarray(xy, dtype=float)
        if self.affines is not None:
            i = max(0, min(int(z) - self.z_min, len(self.affines) - 1))
            M = self.affines[i]
            return xy @ M[:, :2].T + M[:, 2]
        i = max(0, min(int(z) - self.z_min, len(self.offsets) - 1))
        return xy @ self.A.T + self.offsets[i]

    # -- persistence -----------------------------------------------------------
    def to_json(self, path: Union[str, Path]) -> None:
        d = {
            "A": self.A.tolist(),
            "z_min": int(self.z_min),
            "offsets": self.offsets.tolist(),
        }
        if self.affines is not None:
            d["model"] = "per_section_affine"
            d["affines"] = self.affines.tolist()
        else:
            d["model"] = "global_linear_plus_translation"
        Path(path).write_text(json.dumps(d))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Registration":
        d = json.loads(Path(path).read_text())
        affines = np.array(d["affines"]) if d.get("affines") is not None else None
        return cls(A=np.array(d["A"]), offsets=np.array(d["offsets"]),
                   z_min=int(d["z_min"]), affines=affines)


def _collect_correspondences(
    gt: GroundTruth,
    sk: pd.DataFrame,
    *,
    label_col: str,
    min_nodes: int = 1,
    max_skel_spread: float = 400.0,   # full-res px; reject multi-process slices
) -> pd.DataFrame:
    """One (z, neuron) centroid pair per clean single-process slice.

    Iterates each GT slice once (reads its labelmap a single time) and matches
    every neuron that has skeleton nodes there. Returns columns ``z, sx, sy, gx,
    gy``: skeleton centroid in raw CATMAID stack-px, GT mask centroid in mask-px.
    A is fit raw→mask, so it comes out ≈ (1/downscale)·I and ``transform``
    consumes raw stack-px directly.
    """
    # neuron -> set of GT segment Nr
    by_label: Dict[str, List[int]] = {}
    for lab in sk[label_col].unique():
        nrs = gt.nr_for_label(lab)
        if nrs:
            by_label[lab] = nrs

    # per-neuron union bbox in MASK px (computed once). Bounding the isin +
    # connected-components below to this local window instead of the full
    # 2432×2304 frame is the difference between minutes and ~an hour: the inner
    # op runs tens of thousands of times (every (neuron, slice) pair).
    bbox_cache: Dict[str, Tuple[int, int, int, int]] = {}
    for lab, nrs in by_label.items():
        x1 = y1 = 1 << 30
        x2 = y2 = -1
        for nr in nrs:
            bx1, by1, bx2, by2 = gt.bbox_in_mask_px(nr)
            x1, y1 = min(x1, bx1), min(y1, by1)
            x2, y2 = max(x2, bx2), max(y2, by2)
        bbox_cache[lab] = (x1, y1, x2, y2)

    # group skeleton nodes by (z, label): centroid + spread, in stack px
    sk = sk.dropna(subset=["x", "y", "z"]).copy()
    sk["zi"] = sk["z"].astype(int)
    rows: List[dict] = []
    g = sk.groupby(["zi", label_col])
    agg = g.agg(sx=("x", "mean"), sy=("y", "mean"),
                spread_x=("x", "std"), spread_y=("y", "std"), n=("x", "size"))
    agg = agg.reset_index()

    H = W = None
    rows_skipped: List[int] = []
    for z, sub in agg.groupby("zi"):
        if not gt.has_slice(z):
            continue
        lab_img = None
        for _, r in sub.iterrows():
            lab = r[label_col]
            if lab not in by_label or r["n"] < min_nodes:
                continue
            spread = float(np.hypot(np.nan_to_num(r["spread_x"]),
                                    np.nan_to_num(r["spread_y"])))
            if spread > max_skel_spread:
                continue                      # multiple processes; centroid is meaningless
            if lab_img is None:
                try:
                    lab_img = gt.label_slice(z)
                except OSError:               # flaky external drive; skip this slice,
                    rows_skipped.append(int(z))   # its offset interpolates from neighbours
                    break
                H, W = lab_img.shape
            # crop to the neuron's bbox (padded for downscale-floor slack, then
            # clipped); centroid is offset back after.
            _PAD = 4
            bx1, by1, bx2, by2 = bbox_cache[lab]
            bx1, by1 = max(0, bx1 - _PAD), max(0, by1 - _PAD)
            bx2, by2 = min(W - 1, bx2 + _PAD), min(H - 1, by2 + _PAD)
            if bx2 < bx1 or by2 < by1:
                continue
            crop = lab_img[by1:by2 + 1, bx1:bx2 + 1]
            gtm = np.isin(crop, by_label[lab])
            if not gtm.any():
                continue
            cc, ncc = ndimage.label(gtm)
            if ncc != 1:
                continue                      # ambiguous which blob the centroid pairs to
            ys, xs = np.nonzero(gtm)
            rows.append({"z": int(z),
                         "sx": r["sx"], "sy": r["sy"],
                         "gx": xs.mean() + bx1, "gy": ys.mean() + by1})
    if rows_skipped:
        print(f"[reg] WARNING: skipped {len(rows_skipped)} unreadable slice(s) "
              f"(drive); offsets there will be interpolated. e.g. {rows_skipped[:8]}")
    return pd.DataFrame(rows)


def fit(
    gt: GroundTruth,
    skeleton_csv: Union[str, Path],
    *,
    normalize: Callable[[str], str] = normalize_name,
    min_corr_per_slice: int = 3,
    smooth_window: int = 5,
) -> Tuple[Registration, dict]:
    """Fit the section-wise registration. Returns ``(Registration, report)``.

    ``report`` holds diagnostics: correspondence count, linear matrix, fraction of
    slices directly fit vs interpolated, and the centroid residual.
    """
    sk = pd.read_csv(skeleton_csv)
    sk["_label"] = sk["cell_name"].map(normalize)
    corr = _collect_correspondences(gt, sk, label_col="_label")
    if len(corr) < 10:
        raise RuntimeError(f"only {len(corr)} correspondences; cannot fit "
                           "(check name normalization / GT paths)")

    S = corr[["sx", "sy"]].to_numpy()
    G = corr[["gx", "gy"]].to_numpy()
    z = corr["z"].to_numpy()

    # --- stage 1: global linear A from z-centered points ---
    Sc = S.copy(); Gc = G.copy()
    for zz in np.unique(z):
        m = z == zz
        Sc[m] -= S[m].mean(0); Gc[m] -= G[m].mean(0)
    A, *_ = np.linalg.lstsq(Sc, Gc, rcond=None)     # Gc ≈ Sc @ A  (so mask = skel @ A.T? see below)
    A = A.T                                          # store so mask_xy = skel_xy @ A.T

    # --- stage 2: per-section offset = median residual after A ---
    resid = G - S @ A.T
    z_lo, z_hi = int(z.min()), int(z.max())
    # extend to the full GT slice range so every slice is transformable
    z_lo = min(z_lo, min(gt.slice_indices))
    z_hi = max(z_hi, max(gt.slice_indices))
    n_z = z_hi - z_lo + 1
    offsets = np.full((n_z, 2), np.nan)
    direct = 0
    for zz in np.unique(z):
        m = z == zz
        if m.sum() >= min_corr_per_slice:
            offsets[int(zz) - z_lo] = np.median(resid[m], axis=0)
            direct += 1

    # interpolate gaps over z, then light rolling-median smooth
    offsets = _interp_fill(offsets)
    offsets = _smooth(offsets, smooth_window)

    # --- stage 3: per-section AFFINE (the preferred model) ---
    # Fit a full 2×3 affine per slice (≥ min_corr_per_slice clean correspondences),
    # then interpolate the 6 params over z and lightly smooth — same gap-handling as
    # the translation offsets, since the realignment varies smoothly in z.
    affines = np.full((n_z, 2, 3), np.nan)
    direct_aff = 0
    for zz in np.unique(z):
        m = z == zz
        if m.sum() >= min_corr_per_slice:
            M = _fit_slice_affine(S[m], G[m])
            if M is not None:
                affines[int(zz) - z_lo] = M
                direct_aff += 1
    flat = affines.reshape(n_z, 6)
    flat = _smooth(_interp_fill(flat), smooth_window)
    affines = flat.reshape(n_z, 2, 3)

    reg = Registration(A=A, offsets=offsets, z_min=z_lo, affines=affines)

    # residuals after each model (sanity / comparison)
    pred_tr = (S @ A.T) + offsets[(z - z_lo)]
    res_tr = np.linalg.norm(pred_tr - G, axis=1)
    Mz = affines[(z - z_lo)]                                   # (N,2,3)
    pred_aff = np.einsum("nij,nj->ni", Mz[:, :, :2], S) + Mz[:, :, 2]
    res_aff = np.linalg.norm(pred_aff - G, axis=1)
    report = {
        "model": "per_section_affine",
        "n_correspondences": int(len(corr)),
        "A": A.tolist(),
        "n_slices_total": int(n_z),
        "n_slices_direct_fit": int(direct),
        "n_slices_affine_direct_fit": int(direct_aff),
        "n_slices_interpolated": int(n_z - direct),
        "centroid_residual_px": {                             # affine model (what transform uses)
            "mean": float(res_aff.mean()), "median": float(np.median(res_aff)),
            "p90": float(np.percentile(res_aff, 90)),
        },
        "centroid_residual_px_translation_model": {           # legacy, for comparison
            "mean": float(res_tr.mean()), "median": float(np.median(res_tr)),
            "p90": float(np.percentile(res_tr, 90)),
        },
    }
    return reg, report


def _interp_fill(offsets: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows over the z index; clamp the ends."""
    out = offsets.copy()
    idx = np.arange(len(out))
    for c in range(out.shape[1]):
        col = out[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            out[:, c] = 0.0
        else:
            out[:, c] = np.interp(idx, idx[good], col[good])
    return out


def _smooth(offsets: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median over z (odd window); robust to residual spikes."""
    if window < 3:
        return offsets
    w = window if window % 2 else window + 1
    pad = w // 2
    out = np.empty_like(offsets)
    for c in range(offsets.shape[1]):
        padded = np.pad(offsets[:, c], pad, mode="edge")
        out[:, c] = np.array([np.median(padded[i:i + w]) for i in range(len(offsets))])
    return out


def _fit_slice_affine(S: np.ndarray, G: np.ndarray, *,
                      max_resid_px: float = 40.0) -> Optional[np.ndarray]:
    """Robust per-slice affine ``skel→GT``. Returns a (2,3) ``[L | t]`` in the
    ``transform`` convention (``mask_xy = xy @ L.T + t``) or ``None`` if unstable.

    One round of outlier rejection guards against bad correspondences (the
    diagnostic saw a 706 px max): least-squares fit, drop points whose residual
    exceeds ``max(max_resid_px, 3·median)``, refit if ≥4 inliers remain. A pure
    least-squares affine (unlike the median-translation model) is not robust on its
    own, so this rejection matters.
    """
    if len(S) < 4:
        return None
    A1 = np.hstack([S, np.ones((len(S), 1))])          # (n,3)
    M, *_ = np.linalg.lstsq(A1, G, rcond=None)         # (3,2): G ≈ A1 @ M
    res = np.linalg.norm(A1 @ M - G, axis=1)
    keep = res <= max(max_resid_px, 3.0 * np.median(res))
    if keep.sum() >= 4 and keep.sum() < len(S):
        M, *_ = np.linalg.lstsq(A1[keep], G[keep], rcond=None)
    L = M[:2, :].T                                      # (2,2) in the xy @ L.T + t convention
    t = M[2, :]                                         # (2,)
    return np.hstack([L, t.reshape(2, 1)])              # (2,3)


def on_mask_rate(
    gt: GroundTruth,
    skeleton_csv: Union[str, Path],
    reg: Registration,
    *,
    normalize: Callable[[str], str] = normalize_name,
    max_slices_per_neuron: int = 8,
    neuron_limit: Optional[int] = None,
) -> Tuple[float, int]:
    """Fraction of skeleton nodes that land on their own GT segment after ``reg``.

    The headline validation number (vs the ~38% a global affine gave). Returns
    ``(rate, n_nodes_checked)``.
    """
    sk = pd.read_csv(skeleton_csv)
    sk["_label"] = sk["cell_name"].map(normalize)
    sk = sk.dropna(subset=["x", "y", "z"])
    labels = [l for l in sk["_label"].unique() if gt.nr_for_label(l)]
    if neuron_limit:
        labels = labels[:neuron_limit]
    hits = tot = 0
    for lab in labels:
        nrs = gt.nr_for_label(lab)
        s = sk[sk["_label"] == lab]
        zs = sorted(set(s["z"].astype(int)) & set(gt.slice_indices))
        for z in zs[:: max(1, len(zs) // max_slices_per_neuron)][:max_slices_per_neuron]:
            try:
                gtm = np.isin(gt.label_slice(z), nrs)
            except OSError:
                continue
            if not gtm.any():
                continue
            H, W = gtm.shape
            xy = reg.transform(s.loc[s["z"].astype(int) == z, ["x", "y"]].to_numpy(), z)
            xc = xy[:, 0].round().astype(int); yc = xy[:, 1].round().astype(int)
            ok = (xc >= 0) & (xc < W) & (yc >= 0) & (yc < H)
            xc, yc = xc[ok], yc[ok]
            if not len(xc):
                continue
            hits += int(gtm[yc, xc].sum()); tot += len(xc)
    return (hits / tot if tot else float("nan")), tot


# =============================================================================
# CLI: fit + validate + save for one worm
# =============================================================================

def _main() -> None:
    import argparse
    from sam2_utils import config

    ap = argparse.ArgumentParser(description="Fit skeleton->GT-mask section registration.")
    ap.add_argument("--skeleton-csv", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "data" / "groundtruth" / "skeletons_p280" / "aggregate_data_pv.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="registration.json path (default: next to the skeleton CSV)")
    args = ap.parse_args()
    out = args.out or args.skeleton_csv.parent / "registration.json"

    gt = GroundTruth.from_config()
    before, n0 = on_mask_rate(gt, args.skeleton_csv, Registration(
        A=np.eye(2) / gt.downscale, offsets=np.zeros((1, 2)), z_min=0), neuron_limit=40)
    print(f"[reg] baseline (pure 1/{gt.downscale} scale) on-mask: {before:.1%} (n={n0})")

    reg, report = fit(gt, args.skeleton_csv)
    print(f"[reg] model: {report['model']}  correspondences: {report['n_correspondences']}")
    print(f"[reg] mean linear A = {np.round(reg.A, 5).tolist()}")
    print(f"[reg] slices: affine direct-fit {report['n_slices_affine_direct_fit']} "
          f"of {report['n_slices_total']} (rest interpolated over z)")
    print(f"[reg] centroid residual px (AFFINE, used):       {report['centroid_residual_px']}")
    print(f"[reg] centroid residual px (translation, legacy): "
          f"{report['centroid_residual_px_translation_model']}")

    # on-mask: legacy translation model vs the new affine model (apples to apples)
    legacy = Registration(A=reg.A, offsets=reg.offsets, z_min=reg.z_min)  # affines=None
    tr_rate, n_tr = on_mask_rate(gt, args.skeleton_csv, legacy, neuron_limit=40)
    after, n1 = on_mask_rate(gt, args.skeleton_csv, reg, neuron_limit=40)
    print(f"[reg] on-mask: translation model {tr_rate:.1%} (n={n_tr})  ->  "
          f"affine model {after:.1%} (n={n1})")

    reg.to_json(out)
    print(f"[reg] saved -> {out}")


if __name__ == "__main__":
    _main()
