"""
qc.py
-----
Post-hoc QC for SAM2 video-propagated mask stacks.

Drop into ``sam2_utils/`` and use as::

    from sam2_utils import qc

    # Skeleton can be the full multi-cell df + cell_name, a pre-filtered df,
    # a {z: (x, y)} dict, or None to skip skeleton-containment checks.
    df = qc.compute_metrics(
        mask_dir=OUT_DIR,
        skeleton=aggregate_data_pv,     # the full CATMAID df in memory
        cell_name=TARGET_CELL_NAME,     # e.g. "AVAL"
        scale=SCALE,                    # 8 = SAM2 input downscale
        save_downscale=SAVE_DOWNSCALE,  # = scale for pipeline masks (canonical)
    )

Coordinate transforms (the _tif -> mask-px node lookup) go through
``sam2_utils.alignment`` so QC shares one definition with the rest of the
pipeline. Pipeline masks are written at _sam (``save_downscale == scale``, no
resample), so pass ``save_downscale = scale``; a different value only makes sense
for resampled masks written by this module's own ``save_masks``.
    qc.plot_traces(df)                  # line plots of area / centroid / etc.
    qc.show_flagged(df, OUT_DIR,        # thumbnail strip; lazy-loads EM only for flagged frames
        em_loader=lambda z: tifffile.imread(f"slice_{z:04d}.tif"),
        skeleton=aggregate_data_pv, cell_name=TARGET_CELL_NAME)

Design notes
~~~~~~~~~~~~
- **No skeleton reloading.** Pass the in-memory ``annotate`` df straight in.
- **No mask double-read.** ``compute_metrics`` reads each PNG exactly once and
  computes every per-frame and frame-to-frame signal in a single pass.
- **No EM reloading by default.** Thumbnails are only generated for flagged
  frames, and you supply your own ``em_loader`` callable so this module
  doesn't need to know your tif layout.
- Reuses ``sam2_utils.viz.show_mask`` / ``show_points`` for overlays. If you
  haven't refactored those into the package yet, fallback inline plotting
  kicks in.

The five signals computed per frame (see proposed architecture doc, §3):
    area               - mask pixel count
    centroid_y,x       - mask center-of-mass
    n_components       - connected components (after thresholding)
    skeleton_contained - bool: does mask cover the known CATMAID node for z?
    pred_iou           - logged at propagation time if you saved it (NaN if not)

Plus three frame-to-frame signals:
    area_ratio         - area[z] / area[z-1]
    centroid_jump      - L2 distance between consecutive centroids
    temporal_iou       - IoU(mask_z, mask_{z-1})

Composite flag rule:
    flag_count = sum of {pred_iou<0.5, area_ratio∉[0.5,2.0],
                         skeleton_contained==False, temporal_iou<0.3}
    flag = (flag_count >= 1); intervene = (flag_count >= 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndi
from skimage.measure import label
import matplotlib.pyplot as plt

from sam2_utils import alignment   # the one home for coordinate transforms (s4)

# Optional: reuse the project's existing viz module if present.
try:
    from sam2_utils import viz as _viz  # type: ignore
except Exception:
    _viz = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_mask_paths(mask_dir: Path) -> list[tuple[int, Path]]:
    """Return [(z, path), ...] sorted by z, parsing 'mask_NNNN.png' filenames."""
    out: list[tuple[int, Path]] = []
    for p in Path(mask_dir).glob("mask_*.png"):
        stem = p.stem.split("_")[-1]
        if stem.isdigit():
            out.append((int(stem), p))
    out.sort(key=lambda t: t[0])
    return out


def _load_binary(path: Path) -> np.ndarray:
    """Load a uint8/uint16 PNG mask into a boolean array."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:                          # RGBA — collapse
        arr = arr[..., 0]
    return arr > 0

def _node_contained(mask: np.ndarray, sx_i: int, sy_i: int, r: int) -> bool:
    """True iff foreground lies within a (2r+1)x(2r+1) window of node pixel (sx_i, sy_i).

    Single definition of the per-frame containment window — shared by
    compute_metrics and sweep_dilation.py (item 0) so the sensitivity sweep
    scores containment identically to the run. Assumes the node maps inside the
    frame and the mask is non-empty; the no-node / empty-mask / out-of-frame
    tri-state branches stay with the caller.
    """
    H, W = mask.shape
    y0, y1 = max(0, sy_i - r), min(H, sy_i + r + 1)
    x0, x1 = max(0, sx_i - r), min(W, sx_i + r + 1)
    return bool(mask[y0:y1, x0:x1].any())


def _skeleton_xy_for_z(annotate: pd.DataFrame, z: int) -> Optional[tuple[float, float]]:
    """Return the (x_tif, y_tif) of the CATMAID node on frame z, or None."""
    sub = annotate.loc[annotate["z"] == z, ["x_tif", "y_tif"]]
    if sub.empty:
        return None
    # If multiple nodes on this z (branch point), return the centroid — good
    # enough as a containment probe.
    return float(sub["x_tif"].mean()), float(sub["y_tif"].mean())


# Anything the caller might plausibly hand us for "the per-frame skeleton".
SkelInput = Union[
    pd.DataFrame,                              # already filtered to one cell
    Mapping[int, tuple[float, float]],         # {z: (x_tif, y_tif)}
    None,                                      # skip skeleton checks
]


def _resolve_skeleton(
    skeleton: SkelInput,
    cell_name: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Normalize skeleton input into a DataFrame with z / x_tif / y_tif, or None.

    Accepts:
      - A DataFrame already filtered to one cell (must have z, x_tif, y_tif).
      - A DataFrame holding multiple cells + a ``cell_name`` to filter on
        (must additionally have a ``cell_name`` column).
      - A dict {z: (x_tif, y_tif)}.
      - None — skeleton-based checks are skipped.
    """
    if skeleton is None:
        return None
    if isinstance(skeleton, Mapping):
        return pd.DataFrame(
            [(int(z), float(xy[0]), float(xy[1])) for z, xy in skeleton.items()],
            columns=["z", "x_tif", "y_tif"],
        )
    if isinstance(skeleton, pd.DataFrame):
        df = skeleton
        if cell_name is not None and "cell_name" in df.columns:
            df = df[df["cell_name"] == cell_name]
        missing = {"z", "x_tif", "y_tif"} - set(df.columns)
        if missing:
            raise ValueError(f"skeleton df missing columns: {missing}")
        return df[["z", "x_tif", "y_tif"]].reset_index(drop=True)
    raise TypeError(f"unsupported skeleton type: {type(skeleton)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    mask_dir: str | Path,
    skeleton: SkelInput = None,
    *,
    cell_name: Optional[str] = None,
    scale: int = 8,
    save_downscale: int = 4,
    pred_iou_csv: Optional[str | Path] = None,
    pred_iou: Optional[Mapping[int, float]] = None,
    skeleton_dilation_px: int = 3,
    area_ratio_bounds: tuple[float, float] = (0.5, 2.0),
    temporal_iou_min: float = 0.3,
    pred_iou_min: float = 0.5,
    crop_window: object = None,
) -> pd.DataFrame:
    """
    One-pass QC metric collection over a directory of SAM2-output masks.

    Parameters
    ----------
    mask_dir : path
        Directory containing ``mask_NNNN.png`` files from the video predictor.
    skeleton : DataFrame | Mapping[int, (x,y)] | None
        Per-frame CATMAID skeleton positions in full-res tif coords. Accepts:
          - A df already filtered to the target cell (columns ``z``,
            ``x_tif``, ``y_tif``).
          - The full ``aggregate_data_pv`` df + a ``cell_name`` kwarg to
            filter on (must also have a ``cell_name`` column).
          - A ``{z: (x_tif, y_tif)}`` dict.
          - ``None`` — skeleton-containment checks are skipped (composite
            flag falls back to the other three signals).
    cell_name : str, optional
        Used only when ``skeleton`` is a multi-cell df. Equivalent to
        pre-filtering ``aggregate_data_pv[aggregate_data_pv.cell_name == cell_name]``.
    scale : int
        SAM2 input downscale (e.g. 8). Used only for sanity logging.
    save_downscale : int
        Downscale of saved masks relative to the full-res tif (e.g. 4).
        Skeleton (x_tif, y_tif) coords are full-res, so they're divided by
        this to land in mask-pixel space.
    pred_iou_csv : path, optional
        CSV with columns ``z,pred_iou`` (and optionally ``occlusion_score``)
        that you logged during propagation. Joined onto the output df if
        provided; otherwise those columns are NaN.
    pred_iou : Mapping[int, float], optional
        In-memory ``{z: pred_iou}`` (or anything dict-like) — the same data as
        ``pred_iou_csv`` without the disk round-trip. Takes precedence over
        ``pred_iou_csv``. This is what the pipeline passes from
        ``PropagationSession.pred_iou`` (mapped frame_idx -> z).
    skeleton_dilation_px : int
        Tolerance for the "skeleton node is inside the mask" check, in
        mask-pixel units. 3 px is roughly one neurite radius at the canonical
        SCALE == SAVE_DOWNSCALE == 8.
    area_ratio_bounds : (float, float)
        (lo, hi) for the area-ratio signal: a frame fires when
        area[z]/area[z-1] is outside [lo, hi]. Default (0.5, 2.0).
    temporal_iou_min : float
        A frame fires when IoU(z, z-1) < this. Default 0.3.
    pred_iou_min : float
        A frame fires when pred_iou < this. Default 0.5. Live once ``pred_iou``
        (or ``pred_iou_csv``) is supplied; inert only when pred_iou stays NaN
        (no mapping/CSV joined). Set <= 0 to record pred_iou without flagging on it.
    crop_window : alignment.CropWindow, optional
        Tier-2 per-chain crop (`_pcrop`). When given, the masks live in this crop
        space rather than _sam, so a skeleton node's (x_tif, y_tif) is mapped to
        mask px via ``crop_window.sam_to... ``->`` tif_to_crop`` instead of being
        divided by ``save_downscale``. ``None`` keeps the _sam ``/ save_downscale``
        mapping (the default full-frame path).

    Returns
    -------
    df : DataFrame indexed by z, with columns:
        area, centroid_y, centroid_x, n_components, skeleton_contained,
        pred_iou, area_ratio, centroid_jump, temporal_iou,
        flag_count, flag, intervene
    """
    mask_dir = Path(mask_dir)
    paths = _iter_mask_paths(mask_dir)
    if not paths:
        raise FileNotFoundError(f"No mask_*.png files in {mask_dir}")

    skel_df = _resolve_skeleton(skeleton, cell_name)

    rows = []
    prev_mask: Optional[np.ndarray] = None
    prev_centroid: Optional[tuple[float, float]] = None

    for z, p in paths:
        m = _load_binary(p)
        area = int(m.sum())

        # skeleton-node containment (cheapest, most informative signal).
        # Tri-state:
        #   True  — a chain node exists at this z and the mask covers it
        #   False — a chain node exists but the mask does NOT cover it (a flag)
        #   NaN   — no chain node at this z (non-monotonic neurite leaves this
        #           section); NOT assessable, so it must not flag. The area /
        #           temporal signals still guard these frames.
        xy = _skeleton_xy_for_z(skel_df, z) if skel_df is not None else None
        if skel_df is None or xy is None:
            contained: object = np.nan
        elif area == 0:
            contained = False                        # node exists, mask empty
        else:
            # _tif skeleton node -> saved-mask px. Tier-2: map _tif->_pcrop via the
            # crop window (the node lookup must land in the crop the masks were saved
            # in). Else _tif->_sam (== /save_downscale under the canonical rule).
            if crop_window is not None:
                sx, sy = np.asarray(crop_window.tif_to_crop(xy), dtype=float).ravel()[:2]
            else:
                sx, sy = alignment.tif_to_sam(xy, save_downscale)
            sx_i, sy_i = int(round(sx)), int(round(sy))
            if 0 <= sy_i < m.shape[0] and 0 <= sx_i < m.shape[1]:
                contained = _node_contained(m, sx_i, sy_i, skeleton_dilation_px)
            else:
                contained = False                    # node maps outside the frame                  # node maps outside the frame

        if area == 0:
            rows.append(dict(
                z=z, area=0, centroid_y=np.nan, centroid_x=np.nan,
                n_components=0, skeleton_contained=contained,
                area_ratio=np.nan, centroid_jump=np.nan, temporal_iou=np.nan,
            ))
            prev_mask, prev_centroid = m, None
            continue

        cy, cx = ndi.center_of_mass(m)
        n_cc = int(label(m, connectivity=2).max())

        # frame-to-frame signals
        if prev_mask is not None and prev_mask.any():
            inter = int((m & prev_mask).sum())
            union = int((m | prev_mask).sum())
            t_iou = inter / union if union else np.nan
            a_ratio = area / max(int(prev_mask.sum()), 1)
        else:
            t_iou, a_ratio = np.nan, np.nan

        if prev_centroid is not None:
            jump = float(np.hypot(cy - prev_centroid[0], cx - prev_centroid[1]))
        else:
            jump = np.nan

        rows.append(dict(
            z=z, area=area, centroid_y=cy, centroid_x=cx,
            n_components=n_cc, skeleton_contained=contained,
            area_ratio=a_ratio, centroid_jump=jump, temporal_iou=t_iou,
        ))
        prev_mask, prev_centroid = m, (cy, cx)

    df = pd.DataFrame(rows).set_index("z").sort_index()

    # pred_iou join. Prefer an in-memory {z: pred_iou} mapping (what the pipeline
    # passes now that propagate() captures SAM2's mask-decoder IoU head — see
    # pipeline.PropagationSession / _attach_iou_hook); fall back to a CSV; else NaN.
    if pred_iou is not None:
        s = pd.Series(dict(pred_iou), dtype=float)
        s.index = s.index.astype(int)
        df["pred_iou"] = df.index.to_series().map(s)
    elif pred_iou_csv is not None:
        pi = pd.read_csv(pred_iou_csv).set_index("z")
        df = df.join(pi, how="left")
    else:
        df["pred_iou"] = np.nan

    # Composite flag — count how many signals fire. When no skeleton was
    # provided, the containment signal is uninformative so we drop it from
    # the count (otherwise every frame would trip it). Thresholds are
    # parameters (defaults preserve the original hardcoded behavior) so the
    # pipeline / a tuning sweep can adjust them in one call — see PIPELINE_CONTEXT §7.
    ar_lo, ar_hi = area_ratio_bounds
    fc = (
        (df["pred_iou"].fillna(1.0) < pred_iou_min).astype(int)
        + ((df["area_ratio"] < ar_lo) | (df["area_ratio"] > ar_hi)).astype(int)
        + (df["temporal_iou"].fillna(1.0) < temporal_iou_min).astype(int)
    )
    if skel_df is not None:
        fc = fc + (df["skeleton_contained"] == False).astype(int)  # noqa: E712 — NaN must not count
    df["flag_count"] = fc
    df["flag"] = fc >= 1
    df["intervene"] = fc >= 2

    # Friendly summary print
    n = len(df)
    n_flag = int(df["flag"].sum())
    n_int = int(df["intervene"].sum())
    print(f"[qc] {n} frames | flagged: {n_flag} ({n_flag/n:.0%}) "
          f"| intervene: {n_int} ({n_int/n:.0%}) "
          f"| skel miss: {(df['skeleton_contained'] == False).sum()} "  # noqa: E712
          f"| skel n/a: {df['skeleton_contained'].isna().sum()}")
    return df


def plot_traces(df: pd.DataFrame, figsize: tuple[float, float] = (12, 8)) -> plt.Figure:
    """Five-panel diagnostic plot of QC signals over z. Flagged z are shaded."""
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    z = df.index.values

    panels = [
        ("area",          "mask area (px)",  None),
        ("area_ratio",    "area[z]/area[z-1]", (0.5, 2.0)),
        ("temporal_iou",  "IoU(z, z-1)",     (0.3, None)),
        ("centroid_jump", "centroid jump (px)", None),
        ("pred_iou",      "predicted IoU",   (0.5, None)),
    ]

    for ax, (col, label_, band) in zip(axes, panels):
        ax.plot(z, df[col].values, lw=1.0)
        ax.set_ylabel(label_)
        if band is not None:
            lo, hi = band
            if lo is not None:
                ax.axhline(lo, color="red", lw=0.5, ls=":")
            if hi is not None:
                ax.axhline(hi, color="red", lw=0.5, ls=":")
        # shade flagged frames
        for z_flag in df.index[df["flag"]]:
            ax.axvspan(z_flag - 0.5, z_flag + 0.5, color="red", alpha=0.08, lw=0)
        # shade skeleton-missing frames a bit darker
        for z_miss in df.index[df["skeleton_contained"] == False]:  # noqa: E712
            ax.axvspan(z_miss - 0.5, z_miss + 0.5, color="orange", alpha=0.10, lw=0)

    axes[-1].set_xlabel("z (frame index)")
    fig.suptitle(f"QC traces ({len(df)} frames; "
                 f"{int(df['flag'].sum())} flagged, "
                 f"{int(df['intervene'].sum())} intervene)")
    fig.tight_layout()
    return fig


def show_flagged(
    df: pd.DataFrame,
    mask_dir: str | Path,
    em_loader: Optional[Callable[[int], np.ndarray]] = None,
    *,
    n_max: int = 20,
    skeleton: SkelInput = None,
    cell_name: Optional[str] = None,
    save_downscale: int = 4,
    cols: int = 5,
    crop_radius: int = 200,
) -> plt.Figure:
    """
    Thumbnail strip of flagged frames.

    Lazy: only the flagged z's are touched. Each panel shows the mask overlaid
    on the EM crop centered on the CATMAID skeleton node for that frame
    (or on the mask centroid if no skeleton is available). If
    ``em_loader`` is None, masks are shown without EM context.

    ``skeleton`` / ``cell_name`` follow the same conventions as
    ``compute_metrics``.
    """
    mask_dir = Path(mask_dir)
    skel_df = _resolve_skeleton(skeleton, cell_name)
    flagged = df.index[df["flag"]].tolist()[:n_max]
    if not flagged:
        print("[qc] no flagged frames — nothing to show")
        return plt.figure()

    rows = (len(flagged) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows),
                             squeeze=False)

    for ax_idx, z in enumerate(flagged):
        ax = axes[ax_idx // cols][ax_idx % cols]
        m_path = mask_dir / f"mask_{z:04d}.png"
        if not m_path.exists():
            ax.set_visible(False)
            continue
        mask = _load_binary(m_path)
        H, W = mask.shape

        # Decide crop center.
        cx = cy = None
        if skel_df is not None:
            xy = _skeleton_xy_for_z(skel_df, z)
            if xy is not None:
                cx, cy = alignment.tif_to_sam(xy, save_downscale)   # _tif -> mask px
        if cx is None:
            cy, cx = df.loc[z, ["centroid_y", "centroid_x"]]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cy, cx = H / 2, W / 2

        r = crop_radius
        y0, y1 = max(0, int(cy - r)), min(H, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(W, int(cx + r))

        mask_crop = mask[y0:y1, x0:x1]

        # Draw EM background if available — em_loader returns FULL-RES
        if em_loader is not None:
            try:
                em = em_loader(z)
                # downscale-aware crop in full-res coords
                ey0, ey1 = y0 * save_downscale, y1 * save_downscale
                ex0, ex1 = x0 * save_downscale, x1 * save_downscale
                em_crop = em[ey0:ey1, ex0:ex1]
                ax.imshow(em_crop, cmap="gray",
                          extent=(x0, x1, y1, y0))
            except Exception as e:
                ax.text(0.02, 0.02, f"em err: {e}", color="red",
                        transform=ax.transAxes, fontsize=7)

        # Overlay mask
        if _viz is not None and hasattr(_viz, "show_mask"):
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            # _viz.show_mask expects ax + mask in image coords — reproject
            full = np.zeros_like(mask, dtype=bool)
            full[y0:y1, x0:x1] = mask_crop
            _viz.show_mask(full, ax, borders=True)
        else:
            ax.imshow(np.ma.masked_where(~mask_crop, mask_crop),
                      cmap="autumn", alpha=0.45,
                      extent=(x0, x1, y1, y0))

        # Skeleton node marker
        if skel_df is not None:
            xy = _skeleton_xy_for_z(skel_df, z)
            if xy is not None:
                sx, sy = alignment.tif_to_sam(xy, save_downscale)   # _tif -> mask px
                ax.scatter([sx], [sy], s=40, c="yellow",
                           edgecolors="black", linewidths=0.8, zorder=5)

        # Title: which signals fired
        r_row = df.loc[z]
        reasons = []
        if r_row["skeleton_contained"] == False: reasons.append("noskel")  # noqa: E712
        if pd.notna(r_row["area_ratio"]) and not (0.5 <= r_row["area_ratio"] <= 2.0):
            reasons.append(f"area×{r_row['area_ratio']:.1f}")
        if pd.notna(r_row["temporal_iou"]) and r_row["temporal_iou"] < 0.3:
            reasons.append(f"tIoU{r_row['temporal_iou']:.2f}")
        if pd.notna(r_row["pred_iou"]) and r_row["pred_iou"] < 0.5:
            reasons.append(f"pIoU{r_row['pred_iou']:.2f}")
        ax.set_title(f"z={z}  " + " ".join(reasons), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused axes
    for k in range(len(flagged), rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    fig.tight_layout()
    return fig


def export_triage(df: pd.DataFrame, out_csv: str | Path,
                  cell_name: str = "") -> None:
    """Write flagged frames to CSV for the proofreading queue."""
    out = df[df["flag"]].copy()
    if cell_name:
        out.insert(0, "cell_name", cell_name)
    out.to_csv(out_csv)
    print(f"[qc] wrote {len(out)} flagged rows -> {out_csv}")


# ---------------------------------------------------------------------------
# Mask I/O — bridge from in-memory video_segments to disk
# ---------------------------------------------------------------------------

def save_masks(
    video_segments: dict[int, dict[int, np.ndarray]],
    out_dir: str | Path,
    *,
    frame_to_z: Optional[Mapping[int, int]] = None,
    scale: int = 8,
    save_downscale: int = 4,
    obj_ids: Optional[Sequence[int]] = None,
) -> None:
    """
    Write a SAM2 video_segments dict to ``out_dir/mask_NNNN.png``.

    Parameters
    ----------
    video_segments : dict
        ``{frame_idx: {obj_id: 2-D bool mask in SCALE-downscaled space}}``
        — the dict your propagation loop builds in RAM.
    out_dir : path
        Created if missing. Existing mask files are overwritten.
    frame_to_z : Mapping[int, int], optional
        ``{frame_idx: catmaid_z}``. If provided, files are named by z so the
        QC parser and skeleton lookup line up. If omitted, files are named by
        the raw video frame index. Build it like::

            frame_to_z = {i: alignment.file_z_to_catmaid_z(parse_file_z(tif))
                          for i, tif in enumerate(subset_tifs)}

    scale : int
        The SAM2 input downscale the masks were produced at (e.g. 8).
    save_downscale : int
        Target on-disk downscale relative to the full-res tif (e.g. 4).
        Masks are resampled by ``scale / save_downscale`` (nearest-neighbour).
        If they're equal, no resampling happens.
    obj_ids : Sequence[int], optional
        Which obj_ids in video_segments to write. Defaults to all. For
        multi-MLC merging, pass a single id per call and increment a label
        offset, or call once with all ids and they'll be union'd via
        ``np.maximum``.

    Output format: uint16 PNG, instance labels (obj_id as pixel value), one
    file per frame named ``mask_NNNN.png`` (matches the liver pipeline).
    """
    import cv2
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resize_ratio = scale / save_downscale  # >1 means upscale, <1 means downscale

    written = 0
    for frame_idx, per_obj in video_segments.items():
        if not per_obj:
            continue
        key = frame_to_z[frame_idx] if frame_to_z is not None else frame_idx

        # Compose obj_ids into a single instance-label image
        combined: Optional[np.ndarray] = None
        for oid, m in per_obj.items():
            if obj_ids is not None and oid not in obj_ids:
                continue
            m = np.squeeze(m).astype(bool)
            if resize_ratio != 1.0:
                H, W = m.shape
                new_hw = (int(round(W * resize_ratio)),
                          int(round(H * resize_ratio)))
                m = cv2.resize(m.astype(np.uint8), new_hw,
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            label_img = (m.astype(np.uint16) * int(oid))
            combined = (label_img if combined is None
                        else np.maximum(combined, label_img))

        if combined is None:
            continue
        Image.fromarray(combined).save(out_dir / f"mask_{key:04d}.png")
        written += 1

    print(f"[qc] wrote {written} masks -> {out_dir} "
          f"(named by {'z' if frame_to_z is not None else 'frame_idx'}, "
          f"resize={resize_ratio:g}x)")