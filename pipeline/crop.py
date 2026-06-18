"""Video frame prep (shared cache view) and the tier-2 per-chain crop machinery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from sam2_utils import alignment

if TYPE_CHECKING:
    from .config import PipelineConfig

from .frames import (
    FrameStore,
    TifFrameStore,
    _downscale_image,
    _ensure_cached_frames,
    _link_frame,
    _read_tif_window,
)


def prepare_video_frames(chain: dict, annotate_df: pd.DataFrame, *, scale: int,
                         frames_root: Optional[Path],
                         anchor_catmaid_z: int,
                         neuron: str, chain_idx: int,
                         frame_store: Optional[FrameStore] = None
                         ) -> tuple[str, dict[int, int], int, int]:
    """Give SAM2 the 0-indexed downscaled JPEG sequence it needs -> with reuse.

    Two-tier layout under frames_root:
      * a shared cache  ``frames_cache_s{scale}/z{file_z}.jpg`` -> each frame
        decoded+downscaled ONCE ever (see _ensure_cached_frames);
      * a per-chain view ``chain_views/{neuron}_chain{idx:02d}_s{scale}/{i:05d}.jpg``
        of links into that cache, contiguous and 0-indexed as init_state requires.

    Overlapping chains share the cache, so the expensive decode happens once per z
    across the whole dataset instead of once per chain -> this is the fix for the
    frame-prep bottleneck. The cached JPEG bytes are identical to the old per-range
    writer (same imread -> downscale -> imwrite), so masks still reproduce
    pixel-for-pixel.

    The view is namespaced by `neuron`+`chain_idx` so a batch over many neurons
    can't collide (AVAL chain0 vs AVAR chain0), and is rebuilt from scratch each
    call -> links are free, so this sidesteps stale-link risk if a chain's z-range
    changed between runs.

    Returns (view_dir str, frame_to_z, anchor_frame_idx, n_frames).

    Lift from: 'Video Input Setup' frame-prep cell.
    """
    import shutil

    if frames_root is None:
        raise ValueError("PipelineConfig.frames_root must be set for video frame prep")

    # z-extent over ALL chain nodes (non-monotonic in z -> can't use nodes[0]/[-1])
    chain_z = [
        int(annotate_df.loc[
            annotate_df["node_id"].astype(str) == str(n), "z"
        ].item())
        for n in chain["nodes"]
    ]
    start_z, end_z = min(chain_z), max(chain_z)

    fs = frame_store or TifFrameStore()
    anchor_key = fs.key_of_z(anchor_catmaid_z)
    subset = fs.files_in_z_range(start_z, end_z)     # [(key, src_path), ...] sorted by key

    frames_root = Path(frames_root)

    # 1. shared decode cache (write-once, keyed by frame key + scale)
    cache_dir = frames_root / f"frames_cache_s{scale}"
    _ensure_cached_frames(subset, cache_dir, scale)

    # 2. per-chain 0-indexed link view (namespaced; rebuilt fresh each call)
    view_dir = frames_root / "chain_views" / f"{neuron}_chain{chain_idx:02d}_s{scale}"
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)

    anchor_frame_idx: Optional[int] = None
    for i, (key, _src) in enumerate(subset):
        if key == anchor_key:
            anchor_frame_idx = i                 # anchor, in 0-based video index
        _link_frame(cache_dir / f"z{key}.jpg", view_dir / f"{i:05d}.jpg")

    if anchor_frame_idx is None:
        raise AssertionError(
            f"anchor key={anchor_key} not in z-range [{start_z}, {end_z}]"
        )

    frame_to_z = {i: fs.z_of_key(key) for i, (key, _src) in enumerate(subset)}

    # init_state needs a STRING path, not a Path, or it raises
    # "Only MP4 video and JPEG folder are supported".
    return str(view_dir), frame_to_z, anchor_frame_idx, len(subset)


def _chain_skeleton_box_tif(chain: dict, annotate_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) bbox of a chain's whole skeleton in _tif px."""
    ids = {str(n) for n in chain["nodes"]}
    sub = annotate_df[annotate_df["node_id"].astype(str).isin(ids)]
    xs = sub["x_tif"].to_numpy(dtype=float)
    ys = sub["y_tif"].to_numpy(dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _prior_queued_z(qc_csv_path: Path) -> set[int]:
    """The CATMAID-z of the queued (least-trustworthy) frames from a prior qc.csv,
    or empty if absent. Used by chain_crop_from_mask to exclude flagged frames when
    sizing the crop from the _sam masks (a drifted/merged frame would inflate the box
    toward the error). Prefers the `queue` column (intervene-gated), falls back to
    `flag`."""
    qc_csv_path = Path(qc_csv_path)
    if not qc_csv_path.exists():
        return set()
    try:
        df = pd.read_csv(qc_csv_path)
    except Exception:
        return set()
    col = next((c for c in ("queue", "flag") if c in df.columns), None)
    if col is None or "z" not in df.columns:
        return set()
    return {int(z) for z in df.loc[df[col].astype(bool), "z"]}


def mask_union_box_px(masks_dir: Path, *, exclude_z: Optional[set[int]] = None
                      ) -> Optional[tuple[float, float, float, float]]:
    """Union foreground bbox (x0, y0, x1, y1) over a chain's saved mask PNGs, in MASK
    px (the space the PNGs are stored in, _sam for a normal run). Returns None if no
    usable foreground. ``exclude_z`` drops those CATMAID-z (the queued frames); if the
    exclusion would empty the set, it retries over ALL frames so a fully-queued chain
    still yields a box. Reuses qc's single mask-reading definition (_iter_mask_paths /
    _load_binary), so "how a mask is read" stays one place."""
    from sam2_utils import qc   # lazy: keeps pipeline import free of qc's heavy deps
    masks_dir = Path(masks_dir)

    def _union(skip: Optional[set[int]]):
        x0 = y0 = np.inf
        x1 = y1 = -np.inf
        found = False
        for z, path in qc._iter_mask_paths(masks_dir):
            if skip and int(z) in skip:
                continue
            m = qc._load_binary(path)                # bool, mask space
            ys, xs = np.where(m)
            if not xs.size:
                continue
            found = True
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        return (float(x0), float(y0), float(x1), float(y1)) if found else None

    box = _union(exclude_z)
    if box is None and exclude_z:
        box = _union(None)                            # every frame was queued -> use all
    return box


def chain_masks_in_sam(chain_dir: Path, *, verbose: bool = False
                       ) -> dict[int, tuple[np.ndarray, int, int]]:
    """Aggregation prep: read a finished chain's saved masks back onto the canonical
    _sam grid WITH their placement in the full _sam frame, so a per-neuron z-union can
    paste every chain (legacy _sam AND tier-2 _pcrop) onto one common grid.

    Returns ``{catmaid_z: (mask_sam: bool ndarray, x0_sam: int, y0_sam: int)}``:
      * legacy _sam chain (state.json ``crop_window`` is null): the saved mask IS already
        _sam and full-frame, so the entry is ``(mask, 0, 0)``.
      * tier-2 _pcrop chain: the _pcrop mask is downscaled crop_scale -> sam_scale and its
        top-left offset is ``origin_tif / sam_scale``, so it drops into the full _sam frame
        at ``frame[y0:y0+h, x0:x0+w] |= mask_sam``.

    THIS is the single home for the _pcrop -> _sam remap downstream merge needs:
    crop_window is persisted to state.json precisely so aggregation can rebuild the crop
    space (the mask PNG + filename alone do NOT encode their space). The aggregator must
    consume it; globbing ``masks/*.png`` without it would mis-place every tier-2 chain.

    Resolution note: a tier-2 mask carries MORE spatial detail than _sam; here it is
    *downsampled* to the shared _sam grid for the union. Keep the _pcrop original on disk
    if a higher-res mesh is wanted later (the anisotropy / Blender path). Mask reading
    reuses qc's _iter_mask_paths / _load_binary so "how a mask is read" stays one place.
    """
    from sam2_utils import qc   # lazy: keeps pipeline import free of qc's heavy deps
    import cv2

    chain_dir = Path(chain_dir)
    sp = chain_dir / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}
    cwd = state.get("crop_window")
    cw = alignment.CropWindow.from_dict(cwd) if cwd else None

    out: dict[int, tuple[np.ndarray, int, int]] = {}
    for z, path in qc._iter_mask_paths(chain_dir / "masks"):
        m = qc._load_binary(path)                          # bool, mask space (_sam or _pcrop)
        if cw is None:
            out[int(z)] = (m, 0, 0)                         # legacy _sam: full-frame, no offset
            continue
        # _pcrop -> _sam: the window's _sam footprint is size_tif / sam_scale, and a
        # _pcrop pixel (crop_scale tif px) is coarser than... no: crop_scale (2) is FINER
        # than sam_scale (8), so this downscales by sam_scale/crop_scale (e.g. 4x).
        w_sam = max(1, int(round(cw.size_tif[0] / cw.sam_scale)))
        h_sam = max(1, int(round(cw.size_tif[1] / cw.sam_scale)))
        m_sam = cv2.resize(m.astype(np.uint8), (w_sam, h_sam),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        x0 = int(round(cw.origin_tif[0] / cw.sam_scale))
        y0 = int(round(cw.origin_tif[1] / cw.sam_scale))
        out[int(z)] = (m_sam, x0, y0)

    if verbose:
        space = "_pcrop (tier-2)" if cw is not None else "_sam (legacy)"
        print(f"[aggregate] {chain_dir.name}: {len(out)} masks, space={space}")
    return out


def chain_crop_window(chain: dict, annotate_df: pd.DataFrame, *, cfg: "PipelineConfig",
                      image_hw_tif: tuple[int, int],
                      extra_box_tif: Optional[tuple[float, float, float, float]] = None,
                      ) -> "alignment.CropWindow":
    """Tier-2 per-chain CropWindow: the chain's whole-skeleton xy-bbox in _tif,
    padded by cfg.chain_crop_pad_tif, at an adaptive crop_scale.

    crop_scale starts at cfg.chain_crop_scale and is bumped coarser if the padded
    extent's longest edge would exceed cfg.chain_crop_max_px at that scale, so the
    SAM2 input stays bounded for a chain that wanders far across the section. The
    realized window is clipped to the frame (alignment.CropWindow.around_box).

    ``extra_box_tif`` (xyxy, _tif) is UNIONED with the skeleton bbox before padding, the chain_crop_from_mask path passes the _sam mask's bbox here so the window grows
    to contain the segmented cell, not just the centerline (a strict superset of the
    skeleton-only window). None reproduces the skeleton-only sizing exactly.
    """
    box_tif = _chain_skeleton_box_tif(chain, annotate_df)
    if extra_box_tif is not None:
        box_tif = (min(box_tif[0], extra_box_tif[0]), min(box_tif[1], extra_box_tif[1]),
                   max(box_tif[2], extra_box_tif[2]), max(box_tif[3], extra_box_tif[3]))
    H_tif, W_tif = int(image_hw_tif[0]), int(image_hw_tif[1])
    pad = int(cfg.chain_crop_pad_tif)
    min_tif = int(cfg.chain_crop_min_tif)
    # desired extent = skeleton bbox + pad, floored to chain_crop_min_tif (context for
    # a low-motion chain), capped at the image. Expand symmetrically about the bbox
    # centre, then hand a 0-pad box to around_box (which clips to the frame).
    cx = 0.5 * (box_tif[0] + box_tif[2])
    cy = 0.5 * (box_tif[1] + box_tif[3])
    w_tif = min(W_tif, max((box_tif[2] - box_tif[0]) + 2 * pad, min_tif))
    h_tif = min(H_tif, max((box_tif[3] - box_tif[1]) + 2 * pad, min_tif))
    exp_box = (cx - w_tif / 2.0, cy - h_tif / 2.0, cx + w_tif / 2.0, cy + h_tif / 2.0)
    longest = max(w_tif, h_tif, 1.0)
    crop_scale = max(int(cfg.chain_crop_scale),
                     int(np.ceil(longest / float(cfg.chain_crop_max_px))))
    return alignment.CropWindow.around_box(
        exp_box, pad_tif=0, image_hw_tif=image_hw_tif,
        crop_scale=crop_scale, sam_scale=cfg.scale)


def prepare_chain_crop_frames(chain: dict, annotate_df: pd.DataFrame,
                              cw: "alignment.CropWindow", *,
                              frames_root: Optional[Path],
                              anchor_catmaid_z: int,
                              neuron: str, chain_idx: int,
                              frame_store: Optional[FrameStore] = None
                              ) -> tuple[str, dict[int, int], int, int]:
    """Tier-2 video frames: the chain's frames cropped to `cw` and saved as a
    0-indexed JPEG view in `_pcrop` space.

    Each frame is the full-res tif cropped to ``cw.slice_tif()`` then downscaled by
    ``cw.crop_scale``, the SAME crop-then-downscale as ``anchor_crop_predict``, so
    the anchor seed (computed in the crop) and the propagated frames share EXACT
    `_pcrop` pixels. Unlike ``prepare_video_frames`` there is no cross-chain decode
    cache (every chain's window is unique). The per-frame read goes through
    ``_read_tif_window``: a windowed memmap slice that pages in only the
    window's rows instead of decoding the whole ~85 MB frame, which is where this
    function's wall-time lived. View dir is namespaced by neuron+chain+crop_scale and
    rebuilt fresh. Returns (view_dir str, frame_to_z, anchor_frame_idx, n_frames).
    """
    import cv2
    import shutil
    from tqdm import tqdm

    if frames_root is None:
        raise ValueError("PipelineConfig.frames_root must be set for video frame prep")

    chain_z = [
        int(annotate_df.loc[
            annotate_df["node_id"].astype(str) == str(n), "z"
        ].item())
        for n in chain["nodes"]
    ]
    start_z, end_z = min(chain_z), max(chain_z)

    fs = frame_store or TifFrameStore()
    anchor_key = fs.key_of_z(anchor_catmaid_z)
    subset = fs.files_in_z_range(start_z, end_z)     # [(key, src_path), ...] sorted by key

    frames_root = Path(frames_root)
    view_dir = (frames_root / "chain_views"
                / f"{neuron}_chain{chain_idx:02d}_pcrop_s{cw.crop_scale}")
    if view_dir.exists():
        shutil.rmtree(view_dir)
    view_dir.mkdir(parents=True)

    sl = cw.slice_tif()
    frame_to_z: dict[int, int] = {}
    anchor_frame_idx: Optional[int] = None
    for i, (key, src_path) in enumerate(tqdm(subset, desc="caching _pcrop frames", unit="frame")):
        crop = _read_tif_window(src_path, sl)       # windowed read; == cv2.imread(src)[sl]
        crop = _downscale_image(crop, cw.crop_scale)
        cv2.imwrite(str(view_dir / f"{i:05d}.jpg"), crop)
        frame_to_z[i] = fs.z_of_key(key)
        if key == anchor_key:
            anchor_frame_idx = i

    if anchor_frame_idx is None:
        raise AssertionError(
            f"anchor key={anchor_key} not in z-range [{start_z}, {end_z}]"
        )
    return str(view_dir), frame_to_z, anchor_frame_idx, len(subset)
