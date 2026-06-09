"""Coordinate transforms for the SAM2 pipeline. THE single home for them.

PIPELINE_CONTEXT s4 asks that every coordinate transform live in one place and
that variables be tagged with their space. This module is that place. The spaces
(suffix convention, matching pipeline.py):
    _cm    CATMAID stack-pixel coords (annotate_df x, y)
    _tif   full-resolution tif pixels (annotate_df x_tif, y_tif)
    _sam   a scale-downscaled space = _tif / scale (SAM2 video input AND, under
           the canonical save_downscale == scale rule, the on-disk mask space)
    _crop  high-res anchor-crop space (see CropWindow)
plus the z section maps (catmaid_z vs file_z) and the nm -> stack-px voxel divide.

Provides:
    catmaid_to_tif(x, y)            - _cm -> _tif via the stored affine
    apply_affine(xy, M, t)          - apply an arbitrary affine
    tif_to_sam(xy, scale) /
        sam_to_tif(xy, scale)       - _tif <-> _sam resolution-scale point maps
    catmaid_z_to_file_z(z) /
        file_z_to_catmaid_z(z)      - z section-number <-> tif filename z
    nm_to_stack_px(x, y, z)         - CATMAID nm -> stack-pixel (voxel divide)
    CropWindow                      - _crop <-> _tif <-> _sam (high-res anchor crop)
    fit_affine(landmarks)           - least-squares fit + residuals + decomposition
    sample_nodes_grid(df, n, seed)  - evenly spread landmark candidates over xy bbox
"""

from __future__ import annotations

from dataclasses import dataclass
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
# Resolution-scale point maps:  _tif <-> _sam
# =============================================================================
# A "scale" of s means an image downscaled by s on each axis. The SAM2 video
# input is the full-res tif shrunk by SCALE, and the on-disk masks live at the
# same resolution under the canonical save_downscale == SCALE rule. So a full-res
# coordinate lands in the downscaled space by dividing by s, and back by
# multiplying. These are the module-level twins of CropWindow.tif_to_sam, and
# they replace the bare "/ scale" / "* scale" arithmetic that used to be copied
# across pipeline.build_prompts, anchor_crop_predict, and qc. Point convention is
# (x, y); a single (2,) point or an (N, 2) array is accepted, shape preserved.

def tif_to_sam(xy_tif, scale) -> np.ndarray:
    """Full-res tif px -> a scale-downscaled px space.

    `scale` is any full-res downscale factor. Pass SCALE to land in _sam (the
    video-propagation and canonical mask space); pass `save_downscale` to land in
    the saved-mask px space (identical to _sam under save_downscale == SCALE).
    """
    return np.asarray(xy_tif, dtype=float) / float(scale)


def sam_to_tif(xy_sam, scale) -> np.ndarray:
    """Inverse of tif_to_sam: a scale-downscaled px space -> full-res tif px."""
    return np.asarray(xy_sam, dtype=float) * float(scale)


# =============================================================================
# Z-section maps:  catmaid_z <-> file_z
# =============================================================================
# The tif filename z and the CATMAID section number differ by a fixed offset
# (config.FILE_Z_OFFSET): CATMAID_z = file_z + FILE_Z_OFFSET. This was the other
# transform copied inline (load_frame_sam, prepare_video_frames); centralised here
# so the offset has exactly one definition. Scalar in, scalar out.

def catmaid_z_to_file_z(catmaid_z: int) -> int:
    """CATMAID section number -> tif filename z."""
    return int(catmaid_z) - config.FILE_Z_OFFSET


def file_z_to_catmaid_z(file_z: int) -> int:
    """Tif filename z -> CATMAID section number."""
    return int(file_z) + config.FILE_Z_OFFSET


# =============================================================================
# CATMAID nm -> stack-pixel
# =============================================================================
# CATMAID returns node coords in nm; dividing by the per-axis voxel size
# (config.STACK_RESOLUTION_NM) gives stack-pixel coords. Used by
# catmaid.fetch_all_annotations; kept here so every coordinate transform has one
# home (PIPELINE_CONTEXT s4).

def nm_to_stack_px(x_nm, y_nm, z_nm):
    """Convert CATMAID nm coords to stack-pixel coords (per-axis voxel divide).

    Returns (x_px, y_px, z_px) as float arrays (floats for scalar input).
    """
    rx, ry, rz = config.STACK_RESOLUTION_NM
    return (np.asarray(x_nm, dtype=float) / rx,
            np.asarray(y_nm, dtype=float) / ry,
            np.asarray(z_nm, dtype=float) / rz)


# =============================================================================
# Crop window  (the ONE place crop<->tif<->sam mapping lives)
# =============================================================================
# PIPELINE_CONTEXT §4/§5: centralize coordinate transforms; tag every coord with
# its space. The local high-res crop (M3.5) introduces a *new* space, _crop, and
# the prior art (Bader Lab sam2maskpropagator) shows the trap — tangled x/y swaps
# when crop<->full mapping is done ad hoc. So all of it goes here, behind one
# tested object, and nothing else does crop arithmetic by hand.
#
# Spaces (suffix convention matches pipeline.py):
#   _tif   full-resolution tif pixels
#   _sam   SAM2 input space = _tif / sam_scale
#   _crop  pixels inside this crop's image = (_tif - origin_tif) / crop_scale
#
# Point convention is (x, y); box convention is xyxy = (x0, y0, x1, y1).
# numpy arrays are [row, col] = [y, x], so the array slice swaps the order — that
# swap happens in exactly one method (slice_tif) and nowhere else.

@dataclass(frozen=True)
class CropWindow:
    """A high-res crop around an anchor node + its coordinate maps.

    Build with `CropWindow.around_node(...)`, which centers a window on the node
    and clips it to the image. A node near an edge yields a smaller/shifted
    window, so the realized `origin_tif`/`size_tif` are authoritative — never
    assume the node sits at the window centre.
    """
    origin_tif: Tuple[float, float]   # (x, y) top-left in full-res tif px, post-clip
    size_tif: Tuple[int, int]         # (w, h) realized extent in tif px, post-clip
    crop_scale: int                   # downscale when the crop is read (1 = full-res)
    sam_scale: int                    # pipeline SCALE, to map crop results -> _sam

    @classmethod
    def around_node(cls, node_xy_tif, *, size_tif, image_hw_tif,
                    crop_scale: int, sam_scale: int) -> "CropWindow":
        """Center a `size_tif` window on `node_xy_tif` (full-res tif px), clip to image.

        size_tif : int (square) or (w, h) in tif px.
        image_hw_tif : (H, W) of the full-res frame.
        """
        if np.isscalar(size_tif):
            w = h = int(size_tif)
        else:
            w, h = int(size_tif[0]), int(size_tif[1])
        H_tif, W_tif = int(image_hw_tif[0]), int(image_hw_tif[1])
        w, h = min(w, W_tif), min(h, H_tif)              # window can't exceed image
        nx, ny = float(node_xy_tif[0]), float(node_xy_tif[1])
        x0 = int(round(nx - w / 2.0))
        y0 = int(round(ny - h / 2.0))
        x0 = max(0, min(x0, W_tif - w))                  # slide inside the image
        y0 = max(0, min(y0, H_tif - h))
        return cls(origin_tif=(float(x0), float(y0)), size_tif=(int(w), int(h)),
                   crop_scale=int(crop_scale), sam_scale=int(sam_scale))

    @classmethod
    def around_box(cls, box_tif, *, pad_tif, image_hw_tif,
                   crop_scale: int, sam_scale: int) -> "CropWindow":
        """Window covering a `_tif` bbox (xyxy), expanded by `pad_tif`, clipped to image.

        Unlike `around_node` (a fixed-size window slid inside the frame), this is the
        *intersection* of the padded bbox with the image — so it is exactly the box's
        extent, only smaller at an edge. This is the tier-2 per-chain window: pass the
        bbox of a chain's whole skeleton xy-extent so the entire propagation runs in
        one high-res crop (PIPELINE_CONTEXT §7 "Local high-res cropping" tier 2).

        box_tif : (x0, y0, x1, y1) in full-res tif px.
        image_hw_tif : (H, W) of the full-res frame.
        """
        x0, y0, x1, y1 = (float(v) for v in box_tif)
        H_tif, W_tif = int(image_hw_tif[0]), int(image_hw_tif[1])
        x0 = max(0, int(np.floor(x0 - pad_tif)))
        y0 = max(0, int(np.floor(y0 - pad_tif)))
        x1 = min(W_tif, int(np.ceil(x1 + pad_tif)))
        y1 = min(H_tif, int(np.ceil(y1 + pad_tif)))
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        return cls(origin_tif=(float(x0), float(y0)), size_tif=(int(w), int(h)),
                   crop_scale=int(crop_scale), sam_scale=int(sam_scale))

    # --- array slice: numpy is [row, col] = [y, x]. THE only x/y swap. ---
    def slice_tif(self) -> Tuple[slice, slice]:
        """(row_slice, col_slice) to crop a full-res _tif array: img[slice_tif()]."""
        x0, y0 = self.origin_tif
        w, h = self.size_tif
        x0i, y0i = int(round(x0)), int(round(y0))
        return (slice(y0i, y0i + h), slice(x0i, x0i + w))

    @property
    def crop_hw(self) -> Tuple[int, int]:
        """(H, W) of the crop image after the crop_scale downscale."""
        w, h = self.size_tif
        return (int(round(h / self.crop_scale)), int(round(w / self.crop_scale)))

    # --- point maps. points are (x, y); accept (2,) or (N, 2). ---
    def tif_to_crop(self, xy_tif) -> np.ndarray:
        xy = np.asarray(xy_tif, dtype=float)
        return (xy - np.asarray(self.origin_tif, dtype=float)) / self.crop_scale

    def crop_to_tif(self, xy_crop) -> np.ndarray:
        xy = np.asarray(xy_crop, dtype=float)
        return xy * self.crop_scale + np.asarray(self.origin_tif, dtype=float)

    def crop_to_sam(self, xy_crop) -> np.ndarray:
        return self.crop_to_tif(xy_crop) / self.sam_scale

    def tif_to_sam(self, xy_tif) -> np.ndarray:
        return np.asarray(xy_tif, dtype=float) / self.sam_scale

    def sam_to_crop(self, xy_sam) -> np.ndarray:
        """_sam px -> _crop px (via _tif). The map the tier-2 path uses to land the
        _sam-built prompts/skeleton into the per-chain crop the propagation runs in."""
        return self.tif_to_crop(np.asarray(xy_sam, dtype=float) * self.sam_scale)

    # --- box maps. boxes are xyxy. axis-aligned + positive scale, so corners
    #     map to corners and order is preserved. ---
    def box_crop_to_sam(self, box_crop) -> np.ndarray:
        b = np.asarray(box_crop, dtype=float).reshape(2, 2)   # [[x0,y0],[x1,y1]]
        return self.crop_to_sam(b).reshape(4)

    def box_crop_to_tif(self, box_crop) -> np.ndarray:
        b = np.asarray(box_crop, dtype=float).reshape(2, 2)
        return self.crop_to_tif(b).reshape(4)

    # --- (de)serialize: persisted in state.json so QC/review/GUI can rebuild the
    #     crop space a tier-2 chain was propagated/saved in. ---
    def to_dict(self) -> Dict[str, Any]:
        return {"origin_tif": [float(v) for v in self.origin_tif],
                "size_tif": [int(v) for v in self.size_tif],
                "crop_scale": int(self.crop_scale), "sam_scale": int(self.sam_scale)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CropWindow":
        return cls(origin_tif=(float(d["origin_tif"][0]), float(d["origin_tif"][1])),
                   size_tif=(int(d["size_tif"][0]), int(d["size_tif"][1])),
                   crop_scale=int(d["crop_scale"]), sam_scale=int(d["sam_scale"]))


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