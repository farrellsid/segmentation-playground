"""CATMAID-stack <-> tif affine alignment.

Provides:
    catmaid_to_tif(x, y)           — apply the stored affine from config
    apply_affine(xy, M, t)         — apply an arbitrary affine
    fit_affine(landmarks)          — least-squares fit + residuals + decomposition
    sample_nodes_grid(df, n, seed) — evenly spread landmark candidates over xy bbox
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Apply
# =============================================================================

def apply_affine(xy: np.ndarray, M: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map a (N, 2) array of (x, y) points: out = xy @ M.T + t.

    Accepts a single (2,) point too; returns (2,) in that case.
    """
    xy = np.asarray(xy, dtype=float)
    single = xy.ndim == 1
    if single:
        xy = xy[None, :]
    out = xy @ M.T + t
    return out[0] if single else out


def catmaid_to_tif(x, y) -> np.ndarray:
    """Convert CATMAID stack-pixel coords to tif image coords.

    Accepts scalars or arrays. Uses the fitted M_AFFINE/T_AFFINE from config.
    Returns a (..., 2) array (or shape-(2,) for scalar input).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 0:
        pts = np.array([[float(x), float(y)]])
        return apply_affine(pts, config.M_AFFINE, config.T_AFFINE)[0]
    pts = np.column_stack([x, y])
    return apply_affine(pts, config.M_AFFINE, config.T_AFFINE)


# =============================================================================
# Fit
# =============================================================================

Landmark = Tuple[str, float, float, float, float]   # (name, cx, cy, tx, ty)


def fit_affine(landmarks: Iterable[Landmark],
               verbose: bool = True) -> Dict[str, Any]:
    """Fit `tif = M @ catmaid + t` via least squares from N labelled landmarks.

    Parameters
    ----------
    landmarks : iterable of (name, catmaid_x, catmaid_y, tif_x, tif_y)
        Need >=3 points. >=4 recommended; more reduces sensitivity to picking noise.
    verbose : bool
        Print the matrix, translation, decomposition, and per-landmark residuals.

    Returns
    -------
    dict with keys:
        M          : (2, 2) ndarray
        t          : (2,)   ndarray
        scale_x, scale_y, angle_x_deg, angle_y_deg
        residuals  : dict {name -> magnitude px}
        residual_stats : dict {mean, median, max, std}
    """
    lms = list(landmarks)
    if len(lms) < 3:
        raise ValueError(f"Need at least 3 landmarks, got {len(lms)}")

    # Build the LS system. Each landmark contributes 2 rows.
    A_rows: List[List[float]] = []
    b_vals: List[float] = []
    for _, cx, cy, tx, ty in lms:
        A_rows.append([cx, cy, 1, 0, 0, 0])
        A_rows.append([0, 0, 0, cx, cy, 1])
        b_vals.extend([tx, ty])

    A = np.array(A_rows, dtype=float)
    b = np.array(b_vals, dtype=float)
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, tx_off, c, d, ty_off = params

    M = np.array([[a, b_coef], [c, d]])
    t = np.array([tx_off, ty_off])

    scale_x = float(np.sqrt(a ** 2 + b_coef ** 2))
    scale_y = float(np.sqrt(c ** 2 + d ** 2))
    angle_x = float(np.degrees(np.arctan2(-b_coef, a)))
    angle_y = float(np.degrees(np.arctan2(c, d)))

    # Per-landmark residuals
    residuals: Dict[str, float] = {}
    errs: List[float] = []
    rows = []
    for name, cx, cy, tx, ty in lms:
        pred = M @ np.array([cx, cy]) + t
        err = pred - np.array([tx, ty])
        mag = float(np.hypot(*err))
        residuals[name] = mag
        errs.append(mag)
        rows.append((name, pred[0], pred[1], tx, ty, err[0], err[1], mag))
    errs_arr = np.array(errs)

    if verbose:
        print("Affine M:")
        print(M)
        print(f"\nTranslation t: {t}\n")
        print("Decomposition:")
        print(f"  Scale: {scale_x:.5f} (row 1), {scale_y:.5f} (row 2)")
        print(f"  Angle: {angle_x:.4f}° (row 1), {angle_y:.4f}° (row 2)")
        print(f"  a≈d?   {a:.5f} vs {d:.5f}   (diff {abs(a-d):.5f})")
        print(f"  b≈-c?  {b_coef:.5f} vs {-c:.5f}   (diff {abs(b_coef+c):.5f})\n")
        print(f"{'Landmark':<10} {'Predicted':<20} {'Actual':<18} {'Error':<18} {'|err|':>7}")
        print("-" * 80)
        for name, px, py, tx, ty, ex, ey, mag in rows:
            print(f"{name:<10} ({px:7.1f},{py:7.1f})  ({tx:>5},{ty:>5})    "
                  f"({ex:+6.1f},{ey:+6.1f})   {mag:6.1f}")
        print(f"\nResidual stats: mean={errs_arr.mean():.1f} px, "
              f"median={np.median(errs_arr):.1f} px, "
              f"max={errs_arr.max():.1f} px, std={errs_arr.std():.1f} px")

    return {
        "M": M,
        "t": t,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "angle_x_deg": angle_x,
        "angle_y_deg": angle_y,
        "residuals": residuals,
        "residual_stats": {
            "mean": float(errs_arr.mean()),
            "median": float(np.median(errs_arr)),
            "max": float(errs_arr.max()),
            "std": float(errs_arr.std()),
        },
    }


# =============================================================================
# Grid sampler
# =============================================================================

def sample_nodes_grid(df: pd.DataFrame,
                      n_regions: int = 3,
                      seed: int = 42,
                      x_col: str = "x",
                      y_col: str = "y") -> List[pd.Series]:
    """Sample up to n_regions^2 nodes evenly distributed across the xy bbox of df.

    Splits the bbox into an n×n grid; picks one random node per non-empty cell.

    Returns
    -------
    list of pandas.Series (one per selected node).
    """
    rng = np.random.default_rng(seed)
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_edges = np.linspace(x_min, x_max, n_regions + 1)
    y_edges = np.linspace(y_min, y_max, n_regions + 1)

    selected: List[pd.Series] = []
    for i in range(n_regions):
        for j in range(n_regions):
            region = df[
                (df[x_col] >= x_edges[i]) & (df[x_col] < x_edges[i + 1]) &
                (df[y_col] >= y_edges[j]) & (df[y_col] < y_edges[j + 1])
            ]
            if len(region) > 0:
                pick = region.sample(
                    n=1, random_state=int(rng.integers(0, 2**31))
                ).iloc[0]
                selected.append(pick)
    return selected
