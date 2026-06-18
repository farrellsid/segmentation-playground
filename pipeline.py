"""
pipeline.py -> phase functions + per-chain orchestration.

This is the "library of phase functions" the notebook lifts into. Each function
is a near-mechanical extraction of a notebook cell-group. `run_chain` at the
bottom is the thin driver -> reproducing the current AVAL masks is the regression
baseline.

State machine, inline QC, and the GUI are not implemented yet.

Coordinate-space convention
---------------------------
Tag every coordinate/array with the space it lives in, via a suffix:
    _cm   CATMAID stack-pixel space      (annotate_df x, y)
    _tif  full-resolution tif-pixel      (annotate_df x_tif, y_tif)
    _sam  SAM2 *video* input space = full / SCALE (the propagation frames + the
          canonical on-disk mask space; the video predictor + saved masks live here)
    _crop high-res anchor-crop space = (full - crop_origin) / crop_scale. Only the
          *image/anchor* phase uses it (default crop); alignment.CropWindow is
          the one place _crop <-> _tif <-> _sam mapping lives. The crop's box is
          mapped back to _sam before the video seed, so _sam stays the spine.
z is even more error-prone, so name it explicitly too:
    catmaid_z   CATMAID section number   (annotate_df z)
    file_z      tif filename z           (catmaid_z - config.FILE_Z_OFFSET)
    frame_idx   0-based video frame index

Canonical on-disk mask space
----------------------------
Masks are *computed* at _sam (SCALE). Store them there too: set
save_downscale == SCALE so there's no resample and no 2x skeleton bug. Filenames
are `mask_<catmaid_z:04d>.png` (no 'z' prefix) so qc._iter_mask_paths finds them.
Only diverge from this if you decide you want interpolated higher-res masks for
Blender meshing -> and if so, make that one decision in PipelineConfig, in one place.

A note on node_id matching
--------------------------
Every node lookup uses `annotate_df["node_id"].astype(str) == str(node)`. The
notebook's "Prompt Construction" and "Video Input Setup" cells already do this;
its "Load Image" cell used a bare `== str(...)`, which only works when node_id
happens to be object-dtype and silently returns an empty match otherwise. The
.astype(str) form is dtype-agnostic and reproduces the intended result, so it's
used everywhere here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

from time import perf_counter

from sam2_utils import config, alignment


# =============================================================================
# Module-local helpers (shared across phases -> defined once, used twice)
# =============================================================================

def _parse_file_z(p) -> int:
    """'.../1301____z1300.0.tif' -> 1300. Matches the notebook's parse_file_z."""
    token = Path(p).stem.split("z")[-1]
    return int(float(token))


def _downscale_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Area-downsample by `scale`. scale==1 is a no-op copy (notebook helper)."""
    import cv2
    if scale == 1:
        return img.copy()
    return cv2.resize(img, None, fx=1 / scale, fy=1 / scale,
                      interpolation=cv2.INTER_AREA)


def _read_tif_window(tif_path, sl) -> np.ndarray:
    """Return the [y0:y1, x0:x1] window `sl` of a tif as BGR HxWx3 uint8, EXACTLY what
    ``cv2.imread(str(tif_path))[sl]`` returns, but read lazily so only the window's rows
    page in instead of decoding the whole frame (the tier-2 perf optimisation).

    The Zhen EM tifs are uncompressed, single-strip (row-contiguous), 8-bit grayscale, so a
    ``tifffile.memmap`` slice touches only the sliced rows: ~(y1-y0)*W bytes vs the full
    ~85 MB frame. The window is COPIED out (np.array) so the file mapping is released and the
    returned array is plain in-memory. cv2.imread loads grayscale as 3-channel BGR by
    replication, so GRAY2BGR reproduces it bit-for-bit (and BGR==RGB here anyway since the
    source is grayscale). Any tif that can't be windowed this way (compressed, tiled,
    multi-page/3-channel, or tifffile missing) falls through to a full cv2.imread+slice, so
    the output is invariant to the read path, only the wall-time differs."""
    import cv2
    try:
        import tifffile
        mm = tifffile.memmap(str(tif_path), mode="r")
        try:
            if mm.ndim != 2:                         # only 2D grayscale is windowable here
                raise ValueError("not a 2D grayscale tif")
            win = np.array(mm[sl])                    # copy of just the window (pages in its rows)
        finally:
            del mm                                    # release the mapping
        return cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)  # -> BGR 3-ch, matching cv2.imread
    except Exception:
        return cv2.imread(str(tif_path))[sl]          # safe full-read fallback


def _ensure_cached_frames(subset, cache_dir: Path, scale: int) -> None:
    """Decode+downscale any source frames not yet in the shared cache.

    `subset` is a list of ``(key, src_path)`` from a FrameStore — one JPEG per `key`,
    written once ever at this `scale`, named ``z{key}.jpg`` and reused by every chain
    whose z-range overlaps it. This is where the prep cost actually lives (a ~9k x 9k
    imread + resize); overlapping chains now pay it once across the whole dataset
    instead of once per chain. (`key` == file_z for the tif store, == slice z for the
    GT png store; the cache name scheme is unchanged for the target worm.)
    """
    import cv2
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [(k, p) for (k, p) in subset
               if not (cache_dir / f"z{k}.jpg").exists()]
    if not missing:
        return
    for key, src_path in tqdm(missing, desc="caching JPEG frames", unit="frame"):
        img = cv2.imread(str(src_path))          # BGR, fine for grayscale EM (tif or png)
        img = _downscale_image(img, scale)       # match image-mode coord space
        cv2.imwrite(str(cache_dir / f"z{key}.jpg"), img)


def _link_frame(src: Path, dst: Path) -> None:
    """Expose cache frame `src` at 0-indexed view path `dst`.

    Tries symlink, then hard-link, then a plain copy. On Windows bare symlinks
    need Developer Mode or admin, so the hard-link branch is the usual one -> it
    requires src and dst on the same volume (both live under frames_root, so OK).
    """
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            import os
            os.link(src, dst)                    # hard-link fallback (no privilege)
        except OSError:
            import shutil
            shutil.copy2(src, dst)               # last resort


# =============================================================================
# Frame source, the ONE worm-coupled seam
# =============================================================================
# run_chain's phase functions are otherwise worm-agnostic (they consume annotate_df's
# x_tif/y_tif + a logical z), but the EM frames are read straight from the target worm's
# tif stack by file_z. A FrameStore abstracts "logical z -> source EM file + an integer
# cache/order key", so the same pipeline can run on a different worm (e.g. SEM-Dauer 1's
# per-slice PNG export) by passing a different store to run_chain. The default
# (TifFrameStore) reproduces the original target-worm behavior byte-for-byte.

class FrameStore:
    """Maps a *logical z* (catmaid_z on the target worm; VAST slice z on SEM-Dauer 1)
    to its source EM file, and assigns each frame a stable integer ``key`` used for the
    shared JPEG cache name and the frame ordering. Subclass to retarget the pipeline's
    EM source without touching run_chain or any phase function."""

    def key_of_z(self, z: int) -> int:
        raise NotImplementedError

    def z_of_key(self, key: int) -> int:
        raise NotImplementedError

    def file_for_z(self, z: int) -> Path:
        """The single source file for logical z (the anchor-frame load)."""
        raise NotImplementedError

    def files_in_z_range(self, z0: int, z1: int) -> "list[tuple[int, Path]]":
        """``[(key, path), ...]`` for every frame with logical z in [z0, z1], sorted by key."""
        raise NotImplementedError


class TifFrameStore(FrameStore):
    """Target worm: a ``.tif`` stack under ``worm_path`` named ``..z{file_z}.tif``.
    key == file_z, logical z == catmaid_z (related by config.FILE_Z_OFFSET via alignment).
    Reproduces the original glob/parse/z-map exactly, the reproduction path."""

    def __init__(self, worm_path: Optional[Path] = None):
        self.worm_path = Path(worm_path) if worm_path is not None else config.WORM_PATH

    def key_of_z(self, z: int) -> int:
        return alignment.catmaid_z_to_file_z(int(z))

    def z_of_key(self, key: int) -> int:
        return alignment.file_z_to_catmaid_z(int(key))

    def file_for_z(self, z: int) -> Path:
        k = self.key_of_z(z)
        matches = [f for f in self.worm_path.glob("*.tif") if _parse_file_z(f) == k]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected 1 tif for file_z={k} (z={z}), got {len(matches)}: {matches}")
        return matches[0]

    def files_in_z_range(self, z0: int, z1: int) -> "list[tuple[int, Path]]":
        k0, k1 = self.key_of_z(z0), self.key_of_z(z1)
        lo, hi = (k0, k1) if k0 <= k1 else (k1, k0)
        out = [(_parse_file_z(f), f) for f in self.worm_path.glob("*.tif")]
        return sorted([(k, f) for (k, f) in out if lo <= k <= hi], key=lambda kf: kf[0])


# =============================================================================
# Run settings (the notebook's top-level knobs) and small structs
# =============================================================================

@dataclass
class PipelineConfig:
    """Tunable run settings -> the notebook's top-level knobs, in one place.

    These are the fields a future settings GUI binds to. Kept deliberately as a
    plain dataclass: no file format, no loader, no validation yet -> that pairs
    with the GUI later. Static project facts (WORM_PATH, checkpoint
    registry, affine, CATMAID) stay in sam2_utils.config; this is per-run tuning
    only.

    A copy lives on each ChainState (see below) so a resumed/re-opened chain
    reproduces with the settings it actually ran under, even if these defaults
    drift later.
    """
    # model / resolution
    model_size: str = "large"          # tiny / small / base_plus / large; read at predictor-build
    scale: int = 8                     # SAM2 input downscale (1 = full-res)
    save_downscale: int = 8            # on-disk mask downscale; == scale is canonical.
                                       # Diverge only if you want interpolated
                                       # higher-res Blender masks.

    # prompt construction
    k_max_neg: int = 7                 # max negative points per object
    neg_radius: int = 150              # neg-point exclusion radius, _sam pixels
    box_margin: int = 10               # anchor-box padding, _sam pixels

    # QC thresholds. Forwarded to qc.compute_metrics; defaults match its
    # original hardcoded rule. These need tuning on AVAL, and this is the one
    # place to turn the knobs.
    qc_area_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    qc_temporal_iou_min: float = 0.3
    qc_pred_iou_min: float = 0.3       # pred_iou is now populated (propagate captures
                                       # SAM2's mask-decoder IoU head); a frame fires when
                                       # pred_iou < this. Set <= 0 to record but not flag.
    qc_skeleton_dilation_px: int = 3
    # chain-level verdict: mark the whole chain "flagged" when this many frames
    # hit `intervene` (>=2 signals). 1 = flag the chain if any frame needs a human.
    qc_intervene_to_flag_chain: int = 1

    # anchor-quality gate. The raw image-mode anchor mask is scored
    # in _sam space *before* propagation (see score_anchor). Deliberately loose
    # first-pass values: levers are judged by *relative* queue deltas at fixed
    # thresholds, not absolute correctness, so these start permissive and get
    # tuned, not trusted. The containment probe reuses qc_skeleton_dilation_px (no
    # separate knob) so anchor- and per-frame containment mean the same thing, and
    # the dilation sweep informs both.
    gate_min_area_frac: float = 1e-5       # area floor (frac of frame): catch empty/near-empty
    gate_max_area_frac: float = 0.4       # area ceiling: catch a runaway background grab
    gate_min_largest_cc_frac: float = 0.8  # single-CC: >= this share of fg in the largest blob

    # anchor crop (DEFAULT). The image/anchor phase runs image mode
    # on a high-res crop around the node (space _crop, via alignment.CropWindow)
    # instead of the scale-8 full frame, then maps the resulting box back to _sam
    # for the video seed. `scale` is UNCHANGED by this: it still governs video
    # propagation + the canonical mask space; the crop only changes the *anchor*
    # resolution. Set crop_anchor=False to fall back to the legacy scale-8 image
    # phase (the pixel-for-pixel regression baseline).
    # NB the gate's contain radius and area_frac are space-relative, so under the
    # crop the radius is rescaled (x scale/crop_scale) and the area_frac thresholds
    # are measured against the crop, not the full frame -> re-tune on the next run.
    crop_anchor: bool = True           # False -> legacy scale-8 full-frame image phase
    crop_size_tif: int = 1200          # crop window edge in full-res tif px
    crop_scale: int = 2                # crop read downscale (1 = full-res); input edge = size/scale px

    # tier-2 per-chain crop. The resolution lever that actually moves
    # downstream propagation drift ("Local high-res cropping",
    # tier 2): instead of propagating the scale-8 full frame, crop ONE window sized
    # to the chain's whole skeleton xy-extent (+ chain_crop_pad_tif) and run the
    # *entire* image+propagation in that crop at chain_crop_scale. Masks are then
    # stored in this per-chain crop space (`_pcrop`), NOT _sam, and the CropWindow is
    # persisted to state.json so QC/review/GUI can interpret them (alignment.CropWindow).
    # Default OFF -> the _sam full-frame path above is unchanged (and the baseline
    # holds). When on it SUPERSEDES crop_anchor (the chain window is the anchor window
    # too). chain_crop_scale is a *target*: a chain whose padded extent would exceed
    # chain_crop_max_px on its longest edge is read coarser (scale bumped up) so the
    # SAM2 input stays bounded. NB tier-2 loses the cross-chain decode cache (each
    # window is unique) -> one full-res imread per frame per chain; a windowed/memmap
    # read is the documented later optimisation.
    chain_crop: bool = False
    chain_crop_pad_tif: int = 64       # padding around the skeleton xy-extent, _tif px
    chain_crop_scale: int = 2          # target read downscale (1 = full-res)
    chain_crop_max_px: int = 1536      # cap on the crop's longest input edge (bounds VRAM)
    # FLOOR on the crop's _tif extent. A low-motion chain (neurite barely moves in xy)
    # otherwise gets a tiny over-zoomed window where SAM2 loses inter-frame context and
    # the mask collapses to empty (the AIYL chain_02 over-zoom failure). This pads
    # the window out (centred) so the crop always carries enough surrounding context to
    # track. 1024 _tif px -> ~512 px input at crop_scale 2, still ~4x the neurite
    # resolution of the scale-8 full frame.
    chain_crop_min_tif: int = 1024

    # Tier-2 crop sizing from the _sam mask, not the skeleton (default OFF).
    # The skeleton-bbox window (chain_crop_window/_chain_skeleton_box_tif)
    # is sized to the centerline NODES; a cell whose membrane bulges past the nodes +
    # chain_crop_pad_tif gets CLIPPED at the window edge (measured: AIAL/chain_00 clips
    # 24/113 frames). With radius dead (placeholder), there's no per-node extent to pad
    # by. When this is on AND chain_crop is on, the window is grown to the UNION of the
    # skeleton bbox and the bbox of the chain's already-saved _sam masks (the
    # "generate normally -> bbox -> crop" idea): a strict SUPERSET of the skeleton
    # window, so it can only grow to contain the segmented cell, never clip worse. The
    # mask bbox is taken over the NON-queued frames only (the flagged frames are the
    # least-trustworthy masks, drift/merge would inflate the box toward the error);
    # if every frame is queued it falls back to all frames, and if no usable _sam mask
    # exists (or the prior masks are themselves _pcrop) it falls back to the skeleton
    # bbox. chain_crop_max_px still caps the result (bumps crop_scale coarser), trading
    # resolution for not clipping (the accuracy-over-everything call). The natural
    # home is the auto second-pass (batch.tier2_on_flagged), where the _sam masks the
    # first pass just wrote are exactly the bbox source.
    chain_crop_from_mask: bool = False

    # Tier-2 SAFETY. When the per-chain crop yields a POOR anchor, do
    # not propagate a bad crop (or flag the chain); re-run the image phase + the whole
    # propagation in the plain _sam full-frame path instead. "Poor" = empty anchor mask
    # in the crop, OR the anchor gate fires (area/frag/noskel), OR (if a floor is set)
    # image_score below chain_crop_min_image_score. This is what makes chain_crop safe
    # to enable broadly: a chain only KEEPS the crop when its anchor is trustworthy
    # there, else it degrades to the full-frame path rather than to a collapsed _pcrop mask
    # (the AIYL chain_02 over-zoom failure). Only active when chain_crop is on;
    # records ChainState.fell_back_to_sam for the future P(error) features.
    chain_crop_fallback: bool = True
    # image_score floor that triggers the fallback. THIS, not the geometry gate, is what
    # catches the over-zoom: the fallback behavior showed the over-zoomed
    # _pcrop anchor PASSES the geometry gate (clean single blob, contains the node) yet
    # collapses during propagation, the failure is a tracking effect, invisible at the
    # anchor frame. SAM2's own anchor confidence IS the pre-propagation tell: over-zoom
    # scored 0.516 vs 0.848 / 0.879 for healthy crops, so a 0.7 floor cleanly separates
    # them (over-zoom -> fall back to _sam and recover the clean baseline; healthy ->
    # keep tier-2). First-pass value per the "permissive, tune don't trust" rule;
    # widen the sweep before trusting it. 0 disables the floor (gate-only, which
    # proved insufficient on its own).
    chain_crop_min_image_score: float = 0.7

    # multimask anchor auto-select. Ask SAM2 for its 3 candidate
    # masks and auto-pick (node-containment -> plausible-area -> single-CC -> IoU;
    # see _select_anchor_mask) instead of taking the single-mask output. Near-free:
    # SAM2's decoder computes all 3 either way, set_image runs once, only CPU scoring
    # of 3 masks is added. Default OFF to preserve the pixel-for-pixel baseline
    # (it changes which anchor mask is chosen); flip on to compare. Reuses the gate's
    # contain radius + area_frac bounds, so it scores in the same space as the gate.
    multimask_anchor: bool = False

    # paths (project-static paths like WORM_PATH stay in sam2_utils.config)
    output_root: Optional[Path] = None     # e.g. .../output_masks; per-chain subdir is derived
    frames_root: Optional[Path] = None     # parent dir for SAM2 JPEG frame folders


    # video seed (seed ablation). What conditioning to put on the
    # anchor frame for video propagation. SAM2 treats MASK and POINTS/BOX as mutually
    # exclusive per frame (add_new_mask pops point/box inputs and vice-versa), so the
    # valid space is: seed_mask alone, OR any combination of {box, positive, negative}.
    # Defaults reproduce the current seed exactly (fixed-margin box + positive point).
    # The ablation sweeps these to find the seeding sweet spot:
    # more prompts is NOT always better (over-constraining the anchor can hurt tracking).
    seed_box: str = "fixed"            # "none" | "fixed" (box_margin px) | "frac" (box_margin_frac)
    seed_points: bool = True           # include the positive (anchor skeleton) point
    seed_negatives: bool = False       # include build_prompts' neighbour-node negatives
    seed_mask: bool = False            # seed add_new_mask with the anchor mask instead of box/points
                                       # (requires the anchor mask to be in the propagation space:
                                       # legacy _sam or tier-2 _pcrop, NOT tier-1 crop_anchor _crop)
    box_margin_frac: float = 0.0       # %-of-bbox-size box pad when seed_box == "frac" (underfill fix)

    # mask post-processing -> deterministic, no model. Runs before
    # save+QC so QC scores the delivered mask. Off = baseline. Kernels are in
    # scale-8 _sam px; keep <= the neurite half-width.
    postprocess_masks: bool = False
    postproc_open_px: int = 1
    postproc_close_px: int = 1
    postproc_keep_largest_cc: bool = True
    postproc_fill_holes: bool = True

    # which per-frame severity enters the human triage queue. A frame is
    # queued when flag_count >= this. 2 = intervene-level (>=2 corroborating signals),
    # the default: single-signal flags are dominated by
    # dilation-sensitive `noskel` noise (flag_rate moved 0.33->0.19 over a 0..10px
    # dilation sweep) while the intervene set is dilation-robust (rate moved <0.005).
    # Set to 1 to restore the legacy "queue every flag" behaviour.
    qc_triage_min_signals: int = 2

@dataclass
class Prompts:
    """SAM2-space prompts for one chain's anchor frame."""
    points_sam: np.ndarray            # (N, 2) float, _sam space
    labels: np.ndarray                # (N,) int, 1 = positive / 0 = negative
    box_sam: Optional[np.ndarray] = None   # (4,) xyxy float, _sam space; None until box_from_mask


@dataclass
class ChainState:
    """
    Everything needed to run, pause, resume, or re-open one chain.

    Persist this to <neuron>/chain_<idx>/state.json. It holds *references and
    metadata*, never the mask arrays themselves -> those live on disk under
    masks/. video_segments stays in RAM during a run and is reconstructed from
    PNGs if you re-open the chain.
    """
    neuron: str                       # = the notebook's TARGET_CELL_NAME (identity, not a knob)
    chain_idx: int
    status: str = "pending"           # pending / running / done / flagged / failed

    # anchor (filled by select_anchor)
    anchor_node_id: Optional[int] = None
    anchor_catmaid_z: Optional[int] = None
    anchor_frame_idx: Optional[int] = None     # filled once video frames are prepped

    # prompts (filled by build_prompts, updated by box_from_mask / GUI edits)
    prompts: Optional[Prompts] = None

    # image-phase result summary (mask itself goes to disk)
    image_score: Optional[float] = None

    # anchor-quality gate verdict, as a plain JSON-ready dict ->
    # parallel to qc_summary. Filled by score_anchor in run_chain. Logs this as
    # the per-chain "anchor verdict" feature for the learned P(error) detector.
    anchor_score: Optional[dict] = None

    # video input metadata (filled by prepare_video_frames)
    frames_dir: Optional[str] = None
    frame_to_z: Optional[dict[int, int]] = None
    n_frames: Optional[int] = None

    # tier-2 per-chain crop window (alignment.CropWindow.to_dict()), or None for the
    # _sam full-frame path. When set, this chain's masks/frames/prompts all live in
    # `_pcrop` (the crop space) rather than _sam, and QC/review/GUI rebuild the
    # CropWindow from this to map skeleton nodes + clicks. Filled by run_chain when
    # cfg.chain_crop is on.
    crop_window: Optional[dict] = None

    # tier-2 SAFETY: True when cfg.chain_crop was requested but the
    # per-chain crop anchor was poor, so this chain was re-run in the plain _sam path
    # (crop_window is then None and masks/frames are _sam, exactly like a full-frame run).
    # Recorded for how often the fallback fires, and as a P(error) feature.
    fell_back_to_sam: bool = False

    # tier-2 fallback DIAGNOSTICS (captured from the CROP pass before the _sam recovery
    # pass overwrites image_score/anchor_score, otherwise the failing crop-pass values
    # are lost and the final state.json only shows the healthy _sam recovery, making it
    # impossible to tell WHY a chain fell back). None unless fell_back_to_sam is True.
    #   fellback_reason   : which trigger fired, "empty-mask" / "gate(...)" / "score<0.7"
    #   crop_image_score  : the crop-pass anchor image_score (the over-zoom tell)
    #   crop_anchor_score : the crop-pass anchor gate verdict (score_anchor dict)
    fellback_reason: Optional[str] = None
    crop_image_score: Optional[float] = None
    crop_anchor_score: Optional[dict] = None

    # qc summary + triage
    qc_summary: Optional[dict] = None          # flag counts, worst frames, etc.
    triage_frames: list[int] = field(default_factory=list)

    obj_id: int = 1                            # per-chain; increments for multi-obj merge

    # snapshot of the run settings this chain was processed under (reproducibility):
    # a resumed/re-opened chain replays with the knobs it actually ran under, even
    # if the global defaults have since drifted.
    config: PipelineConfig = field(default_factory=PipelineConfig)

    # runtime telemetry, filled by run_chain's per-phase timer (_step/_finish).
    # Declared as real fields (not stamped-on attributes) so they serialise with
    # the rest of the state: batch.py reads them right after a run to write
    # _timing.csv, and persisting them keeps a resumed/re-opened chain's timing.
    phase_seconds: dict = field(default_factory=dict)        # {phase label: seconds}
    phase_subseconds: dict = field(default_factory=dict)     # {sub-step label: seconds}


# =============================================================================
# Phase functions  (each ~= one notebook cell-group)
# =============================================================================

def select_anchor(chain: dict, annotate_df: pd.DataFrame) -> tuple[int, int]:
    """Pick the anchor node for a chain and resolve its CATMAID z.

    Currently the mid-node heuristic. Returns (anchor_node_id, anchor_catmaid_z).

    Lift from: 'Load Image' cell (midnode / TARGET_Z).
    not implemented: failed-anchor auto re-pick policy lives here later, not in the driver.
    """
    nodes = chain["nodes"]
    midnode = nodes[len(nodes) // 2]
    z_series = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(midnode), "z"
    ]
    anchor_catmaid_z = int(z_series.item())   # .item() asserts exactly one match
    return midnode, anchor_catmaid_z


def load_frame_sam(catmaid_z: int, *, scale: int,
                   frame_store: Optional[FrameStore] = None
                   ) -> tuple[np.ndarray, tuple[int, int]]:
    """Find the EM frame for logical `catmaid_z`, read it, downscale by `scale`.

    Returns (image_sam RGB uint8, full_hw) -> full_hw is the pre-downscale (H, W),
    kept only so later steps can map back to full-res if ever needed.

    `frame_store` selects the EM source; the default (TifFrameStore) is the original
    tif-stack path. Lift from: parse_file_z + tif glob + cv2.imread + downscale_image.
    """
    import cv2

    fs = frame_store or TifFrameStore()
    src_path = fs.file_for_z(catmaid_z)

    image_full = cv2.cvtColor(cv2.imread(str(src_path)), cv2.COLOR_BGR2RGB)
    H_full, W_full = image_full.shape[:2]
    image_sam = _downscale_image(image_full, scale)
    return image_sam, (H_full, W_full)


def build_prompts(anchor_node_id: int, catmaid_z: int, annotate_df: pd.DataFrame,
                  *, scale: int, k_max_neg: int, neg_radius: int) -> Prompts:
    """Anchor skeleton node (positive) + K nearest same-z nodes (negative), in _sam.

    Returns a Prompts with box_sam still None.

    Lift from: 'Prompt Construction' cell. Note the x_tif/y_tif -> _sam division
    by `scale` -> that division is exactly the kind of thing the space-suffix
    convention is meant to make un-loseable.

    `neg_radius` is accepted for signature stability but is intentionally NOT
    applied: the notebook's prompt-construction cell never filtered negatives by
    radius (it only capped count via k_max_neg). Applying it now would change the
    masks and break the regression match. Wire the radius gate in later when QC
    thresholds are being tuned, not here.
    """
    # --- positive: the anchor (mid) node, _tif -> _sam ---
    cell_node = annotate_df.loc[
        annotate_df["node_id"].astype(str) == str(anchor_node_id)
    ]
    pos_sam = alignment.tif_to_sam(
        cell_node[["x_tif", "y_tif"]].to_numpy(dtype=float), scale)   # (1, 2)

    points: list[list[float]] = [pos_sam[0].tolist()]
    labels: list[int] = [1]

    # --- negatives: nearest same-z nodes by CATMAID (_cm) distance ---
    cell_x_cm = float(cell_node["x"].iloc[0])
    cell_y_cm = float(cell_node["y"].iloc[0])

    z_points = annotate_df[annotate_df["z"] == catmaid_z].copy()
    z_points["x"] = pd.to_numeric(z_points["x"], errors="coerce")
    z_points["y"] = pd.to_numeric(z_points["y"], errors="coerce")
    z_points["distance"] = np.sqrt(
        (z_points["x"] - cell_x_cm) ** 2 + (z_points["y"] - cell_y_cm) ** 2
    )
    z_points = z_points.sort_values(by="distance").reset_index(drop=True)
    if len(z_points) and z_points.iloc[0]["distance"] == 0:
        z_points = z_points.drop(0).reset_index(drop=True)   # drop the anchor itself

    negnodes_sam = alignment.tif_to_sam(
        z_points[["x_tif", "y_tif"]].to_numpy(dtype=float), scale)   # (M, 2)
    n_neg = min(len(z_points), k_max_neg)
    for i in range(n_neg):
        points.append([float(negnodes_sam[i, 0]), float(negnodes_sam[i, 1])])
        labels.append(0)

    return Prompts(points_sam=np.array(points, dtype=float),
                   labels=np.array(labels, dtype=int))


def _point_in_mask(mask: np.ndarray, x: float, y: float, radius: int) -> bool:
    """True if any foreground pixel lies within `radius` of (x, y).

    Point and mask must share a space (no transform here). Out-of-frame -> False.
    The single neighbourhood-containment test reused by score_anchor (anchor gate)
    and _select_anchor_mask (multimask pick) and matching qc's per-frame probe, so
    "node is inside the mask" means one thing everywhere.
    """
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= yi < h and 0 <= xi < w):
        return False
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    return bool(mask[y0:y1, x0:x1].any())


def _largest_cc_frac(mask: np.ndarray) -> tuple[int, float]:
    """(n_components, largest-CC fraction of foreground) for a bool mask.

    (0, 0.0) when empty. The single-CC health measure generalises the
    largest-component pick already in box_from_mask; reused by score_anchor and
    _select_anchor_mask so the gate and the multimask pick agree.
    """
    from skimage.measure import label as cc_label

    m = np.asarray(mask).astype(bool)
    area = int(m.sum())
    if area == 0:
        return 0, 0.0
    lbl = cc_label(m, connectivity=2)
    sizes = np.bincount(lbl.ravel())[1:]            # drop background (label 0)
    return int(sizes.size), (float(sizes.max() / area) if sizes.size else 0.0)


def _positive_point(prompts: Optional[Prompts]) -> Optional[np.ndarray]:
    """The first positive (anchor/skeleton) prompt point, or None. Mask space."""
    if prompts is None or prompts.points_sam is None:
        return None
    pts = np.asarray(prompts.points_sam, dtype=float)
    pos = pts[np.asarray(prompts.labels) == 1]
    return pos[0] if len(pos) else None


def _select_anchor_mask(masks: np.ndarray, scores: np.ndarray, prompts: Optional[Prompts],
                        image_hw: tuple[int, int], *, contain_radius_px: int,
                        area_bounds: tuple[float, float]) -> tuple[int, np.ndarray, float]:
    """Pick the best of SAM2's multimask candidates for an anchor seed.

    Ranking is lexicographic and *graceful*, it always returns one candidate, so a
    chain with no clean mask still produces a box (and is then caught by the gate /
    empty-mask flag downstream) rather than crashing. Priority order is
    node-containment / plausible-area / single-CC:
      1. contains the positive node      : domain anchor, the mask must sit on the neurite
      2. plausible area (in area_bounds)  : reject runaway background grabs / empty masks
                                            *before* single-CC, since a runaway grab is
                                            usually one huge clean blob (lcc ~ 1.0) that
                                            would otherwise win on step 3
      3. single-CC health (largest_cc_frac) : one clean blob over fragmented membrane
      4. SAM predicted IoU (scores)       : final tiebreak among otherwise-equal masks

    Everything is judged in the space the masks live in (the caller passes matching
    `prompts`, `image_hw`, and `contain_radius_px`), so this is transform-free like
    score_anchor. The chosen mask still only sources the video-seed *box*; the
    positive seed point is unchanged, so a multimask pick never moves the seed point.
    Returns (best_idx, mask_bool, score).
    """
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float).ravel()
    H, W = int(image_hw[0]), int(image_hw[1])
    frame_px = H * W
    min_af, max_af = area_bounds
    pos = _positive_point(prompts)

    best_idx, best_key = 0, None
    for i in range(masks.shape[0]):
        m = masks[i]
        area_frac = (int(m.sum()) / frame_px) if frame_px else 0.0
        contained = pos is not None and _point_in_mask(m, float(pos[0]), float(pos[1]), contain_radius_px)
        _, lcc = _largest_cc_frac(m)
        score = float(scores[i]) if i < scores.size else 0.0
        key = (int(contained), int(min_af <= area_frac <= max_af), lcc, score)
        if best_key is None or key > best_key:
            best_idx, best_key = i, key
    return best_idx, masks[best_idx], (float(scores[best_idx]) if best_idx < scores.size else 0.0)


def image_predict(image_predictor, image_sam: np.ndarray, prompts: Prompts, *,
                  multimask: bool = False, select_contain_radius_px: int = 0,
                  select_area_bounds: tuple[float, float] = (0.0, 1.0),
                  ) -> tuple[np.ndarray, float, np.ndarray]:
    """Run image-mode SAM2 on the anchor frame.

    Returns (mask bool HxW, score, logits) in whatever space `image_sam` is.

    `multimask=False` (default) is the single-mask path, `multimask_output=False`,
    the regression baseline, exactly reproduces the notebook. `multimask=True`
    asks SAM2 for its 3 candidate masks and auto-selects one via `_select_anchor_mask`.
    This is near-free: SAM2's mask decoder
    *always* computes all 3 candidates regardless of the flag (it only slices the
    output — see sam2/modeling/sam/mask_decoder.py), and the heavy image-encoder
    `set_image` runs once either way; the only added work is scoring 3 masks on CPU.
    The selection params are only consulted when `multimask=True`.

    Lift from: 'Image Prediction' cell.
    not implemented: the GUI refinement loop wraps this call (re-predict on each point edit).
    """
    import torch

    with torch.inference_mode():
        image_predictor.set_image(image_sam)
        masks, scores, logits = image_predictor.predict(
            point_coords=np.asarray(prompts.points_sam, dtype=float),
            point_labels=np.asarray(prompts.labels, dtype=int),
            multimask_output=multimask,
        )
    if not multimask:
        return masks[0].astype(bool), float(scores[0]), logits
    best, mask_b, score = _select_anchor_mask(
        masks, scores, prompts, masks.shape[1:],
        contain_radius_px=select_contain_radius_px, area_bounds=select_area_bounds)
    return mask_b.astype(bool), score, logits[best:best + 1]


def box_from_mask(mask_sam: np.ndarray, *, margin: int, margin_frac: float = 0.0,
                  image_hw_sam: tuple[int, int]) -> Optional[np.ndarray]:
    """Largest connected component -> xyxy box (+margin), clipped to image, _sam space.

    Returns the box, or None if the mask is empty -> None is the signal to flag the
    chain for human review rather than feed garbage into propagation.

    ``margin`` is a fixed pad in mask-space px (the historical behaviour). ``margin_frac``
    (>0) adds a pad scaled to the box's own size (``round(margin_frac * max(w, h))`` of
    the largest-CC bbox, applied per side) and the effective pad is the LARGER of the two.
    Rationale (seed ablation): when the anchor mask *under*-fills the cell, a
    fixed 10px box doesn't enclose the whole process, so propagation can't recover the
    missing extent; a size-relative pad widens the box in proportion to the object, which
    is the cheap way to keep the box seed competitive with the (curated) mask seed.

    Lift from: 'Bounding Box Generation' cell.
    """
    from skimage.measure import label as cc_label, regionprops

    m = np.asarray(mask_sam).astype(bool)
    if not m.any():
        return None

    # largest blob only (suppresses stray membrane fragments)
    lbl = cc_label(m, connectivity=2)
    m = lbl == (1 + int(np.argmax([r.area for r in regionprops(lbl)])))

    H_sam, W_sam = image_hw_sam
    ys, xs = np.where(m)
    w = int(xs.max()) - int(xs.min()) + 1
    h = int(ys.max()) - int(ys.min()) + 1
    pad = max(int(margin), int(round(margin_frac * max(w, h))))   # frac scales with object size
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, W_sam - 1)
    y1 = min(int(ys.max()) + pad, H_sam - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


@dataclass
class AnchorScore:
    """Threshold-light quality verdict for one chain's anchor (image-phase) mask.

    The geometry here is judged entirely in _sam space -> the space image_predict
    works in, and the space prompts.points_sam already lives in -> so there is *no*
    coordinate transform in this function (deliberately: the anchor mask and the
    positive prompt point share one frame). That keeps it off the bug-prone
    transform path.

    Three sub-checks, mirroring the gate:
      contained        -> does the mask cover the positive (skeleton) prompt point,
                          within a small radius? Tri-state, same meaning as
                          qc.skeleton_contained but encoded JSON-clean:
                          True / False / None(no positive point -> abstain).
      n_components,
      largest_cc_frac   -> single-CC health: fraction of foreground in the largest
                          connected component (a clean anchor is ~one blob).
      area_frac         -> foreground as a fraction of the frame: floored to catch an
                          empty/near-empty mask, ceiled to catch a runaway grab of
                          background.

    `passed` is the AND of the enabled checks; an abstaining (None) containment does
    not fail. `reasons` lists the checks that fired, reusing the qc vocabulary
    ('noskel' / 'area' / 'frag') so the gate and the per-frame QC speak the same
    language downstream.
    """
    contained: Optional[bool]
    n_components: int
    largest_cc_frac: float
    area_frac: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _anchor_score_to_dict(s: AnchorScore) -> dict:
    """JSON-ready plain dict (no numpy types) for ChainState.anchor_score."""
    return {
        "contained": None if s.contained is None else bool(s.contained),
        "n_components": int(s.n_components),
        "largest_cc_frac": float(s.largest_cc_frac),
        "area_frac": float(s.area_frac),
        "passed": bool(s.passed),
        "reasons": list(s.reasons),
    }


def score_anchor(mask_sam: np.ndarray, prompts: Prompts, *,
                 image_hw_sam: tuple[int, int],
                 contain_radius_px: int,
                 min_area_frac: float,
                 max_area_frac: float,
                 min_largest_cc_frac: float) -> AnchorScore:
    """Score the raw image-mode anchor mask for propagation-readiness, in _sam space.

    Called *before* box_from_mask (it judges the raw multi-blob mask, not the
    largest-CC box) and before propagation, so a bad anchor costs one frame's
    compute instead of a wasted ~300-frame propagate. This is the
    *scoring* half only: it is pure (reads the mask + prompts, writes nothing) and
    decides nothing -> the gate that escalates prompts / re-picks the node / blocks
    propagation consumes this verdict in the next increment.

    Lift/parallel: the containment probe is the same neighbourhood test as
    qc.compute_metrics (so anchor- and per-frame containment agree); the single-CC
    measure generalises the largest-component pick already in box_from_mask.
    """
    H_sam, W_sam = image_hw_sam
    frame_px = int(H_sam) * int(W_sam)
    m = np.asarray(mask_sam).astype(bool)
    area = int(m.sum())
    area_frac = (area / frame_px) if frame_px else 0.0

    # --- containment: does the mask cover the positive (anchor) prompt point? ---
    # Tri-state, matching qc.skeleton_contained: an empty mask with a node present
    # is an explicit miss (False); no positive point at all is an abstain (None).
    # (_point_in_mask returns False for a node that maps outside the frame.)
    pos = _positive_point(prompts)                 # the anchor (skeleton) node, mask space
    if pos is None:
        contained: Optional[bool] = None
    elif area == 0:
        contained = False
    else:
        contained = _point_in_mask(m, float(pos[0]), float(pos[1]), contain_radius_px)

    # --- single-CC health ---
    n_cc, largest_cc_frac = _largest_cc_frac(m)

    # --- compose the verdict ---
    reasons: list[str] = []
    if not (min_area_frac <= area_frac <= max_area_frac):
        reasons.append("area")
    if largest_cc_frac < min_largest_cc_frac:
        reasons.append("frag")
    if contained is False:                         # None abstains, must not fail
        reasons.append("noskel")

    return AnchorScore(
        contained=contained,
        n_components=n_cc,
        largest_cc_frac=largest_cc_frac,
        area_frac=area_frac,
        passed=(len(reasons) == 0),
        reasons=reasons,
    )


def anchor_crop_predict(image_predictor, image_full: np.ndarray, full_hw: tuple[int, int],
                        anchor_node_id: int, prompts_sam: "Prompts", annotate_df: pd.DataFrame,
                        *, scale: int, crop_size_tif: int, crop_scale: int,
                        cw: Optional["alignment.CropWindow"] = None,
                        multimask: bool = False, select_contain_radius_px: int = 0,
                        select_area_bounds: tuple[float, float] = (0.0, 1.0),
                        ) -> tuple[np.ndarray, float, "alignment.CropWindow", "Prompts"]:
    """Image-mode anchor prediction on a high-res crop (default path).

    Crops a `crop_size_tif` window around the anchor node (alignment.CropWindow ->
    the single home of _crop<->_tif<->_sam mapping), runs image mode in _crop at
    `crop_scale`, and returns the mask + the CropWindow so the caller can map the
    box back to _sam for the video seed.

    The prompt POINTS are the already-built _sam prompts remapped into _crop
    (_sam -> _tif via *scale, then CropWindow.tif_to_crop); negatives that fall
    outside the window are dropped (the positive anchor is inside by construction).
    The returned Prompts is in _crop, so the gate (score_anchor) can score in the
    same space the mask lives in. The original _sam prompts are untouched -> they
    still seed the video positive point; only the box comes from the crop.

    `multimask` + the `select_*` params forward straight to image_predict's
    multimask auto-select. They must already be in _crop space: the
    caller rescales `select_contain_radius_px` by scale/crop_scale, and
    `select_area_bounds` are frame-fraction bounds the crop config already tunes,
    so selection scores in the same _crop space as the mask.

    A prebuilt ``cw`` (tier-2 per-chain window) is used as-is: the image phase then
    runs in the SAME crop the whole chain propagates in, so the seed needs no
    _crop->_sam remap. When ``cw`` is None (tier-1 default) a fresh ``crop_size_tif``
    window is centred on the anchor node.

    Returns (mask_crop bool HxW, score, cw, prompts_crop).
    """
    from sam2_utils import alignment

    if cw is None:
        # tier-1: a fresh window centred on the anchor node (_tif).
        node = annotate_df.loc[annotate_df["node_id"].astype(str) == str(anchor_node_id)]
        node_xy_tif = node[["x_tif", "y_tif"]].to_numpy(dtype=float)[0]
        cw = alignment.CropWindow.around_node(
            node_xy_tif, size_tif=crop_size_tif, image_hw_tif=full_hw,
            crop_scale=crop_scale, sam_scale=scale)

    crop_full = image_full[cw.slice_tif()]                  # _tif window
    crop_img = _downscale_image(crop_full, cw.crop_scale)   # _crop input image (window governs scale)
    H_crop, W_crop = crop_img.shape[:2]

    # _sam prompt points -> _tif -> _crop. Keep all positives; drop out-of-window negatives.
    pts_sam = np.asarray(prompts_sam.points_sam, dtype=float)
    labels = np.asarray(prompts_sam.labels, dtype=int)
    pts_crop = cw.tif_to_crop(alignment.sam_to_tif(pts_sam, scale))   # _sam -> _tif -> _crop
    in_bounds = ((pts_crop[:, 0] >= 0) & (pts_crop[:, 0] < W_crop) &
                 (pts_crop[:, 1] >= 0) & (pts_crop[:, 1] < H_crop))
    keep = in_bounds | (labels == 1)
    prompts_crop = Prompts(points_sam=pts_crop[keep], labels=labels[keep])  # NB: _crop coords

    mask_crop, score, _logits = image_predict(
        image_predictor, crop_img, prompts_crop, multimask=multimask,
        select_contain_radius_px=select_contain_radius_px,
        select_area_bounds=select_area_bounds)
    return mask_crop, score, cw, prompts_crop


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

    ``extra_box_tif`` (xyxy, _tif) is UNIONED with the skeleton bbox before padding —
    the chain_crop_from_mask path passes the _sam mask's bbox here so the window grows
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
    ``cw.crop_scale`` — the SAME crop-then-downscale as ``anchor_crop_predict``, so
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


@dataclass
class FrameResult:
    """One propagated frame, as the interruptible loop yields it.

    frame_idx  : 0-based video frame index.
    masks      : {obj_id: bool HxW in _sam} for this frame.
    logit_conf : mean-foreground-sigmoid proxy for the session's obj_id
                 (NaN if the mask is empty). The "stop discarding logits" proxy.
    pred_iou   : SAM2 mask-decoder predicted-IoU for the session's obj_id, captured
                 from the decoder's IoU head (see _attach_iou_hook). NaN if the hook
                 couldn't read it (e.g. a SAM2 refactor) -> the QC flag rule treats
                 NaN as inert, so that degrades to the pre-capture behaviour.
    """
    frame_idx: int
    masks: dict[int, np.ndarray]
    logit_conf: float
    pred_iou: float


def _attach_iou_hook(video_predictor, sink: dict[int, float]) -> Callable[[], None]:
    """Wrap the predictor's ``_track_step`` to record the mask-decoder predicted IoU
    per frame into ``sink`` (frame_idx -> float). Returns a ``restore()`` callable.

    Why a hook and not the public API: SAM2 *computes* the IoU head output (``ious``)
    in the mask decoder, but ``track_step`` unpacks the decoder tuple and **discards**
    it — ``(_, _, _, low_res_masks, ...) = sam_outputs`` — so it never reaches
    ``current_out``, ``inference_state``, or the ``propagate_in_video`` yield (trace:
    sam2/modeling/sam2_base.py ``_forward_sam_heads`` -> ``track_step``). ``_track_step``
    is the last point the value is in hand: it returns ``(current_out, sam_outputs, ...)``
    and ``sam_outputs[2]`` is ``ious`` ([B, M]; M=1 on propagated frames, M=3 on a
    multimask anchor where SAM2 then argmax-selects). We read it **read-only** and pass
    the call through untouched, so masks are bit-identical to an unhooked run.

    Best-effort by design: if ``_track_step`` is absent or its return shape changes, we
    leave pred_iou unpopulated (NaN) rather than raise — same principle as the timing
    wrapper that must never kill a chain. Scope the hook to one propagation run and
    always ``restore()`` (PropagationSession does this in close()).
    """
    orig = getattr(video_predictor, "_track_step", None)
    if orig is None or not callable(orig):
        return lambda: None

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            frame_idx = args[0] if args else kwargs["frame_idx"]
            ious = out[1][2]                       # sam_outputs[2] == decoder IoU head
            # selected mask's predicted IoU; max() == the argmax SAM2 itself selects
            # when multimask (anchor), and the lone value when single-mask (tracking).
            sink[int(frame_idx)] = float(ious.max())
        except Exception:
            pass                                   # leave NaN; never break tracking
        return out

    video_predictor._track_step = wrapped
    return lambda: setattr(video_predictor, "_track_step", orig)


class PropagationSession:
    """Interruptible, resumable SAM2 video propagation for ONE chain.

    Holds the live ``inference_state`` so propagation can be **stopped** at a degrading
    frame, **corrected** (``add_points`` / ``add_mask``), and **resumed**: the
    interruptible-propagation primitive. The headless ``propagate()``
    function below drives it straight through (and is the only thing run_chain uses); a
    future auto-intervention loop or napari GUI drives the *same* methods. No GUI, napari,
    or torch import lives here — the session only calls the predictor's public API plus the
    one read-only ``_track_step`` IoU hook.

    Continuity lives in ``inference_state``, NOT in any generator object. Consequences the
    caller must respect:
      * ``propagate(...)`` yields frames lazily; the caller may ``break`` at any frame and
        every frame seen so far is already recorded in the accumulators.
      * ``add_points`` / ``add_mask`` mutate ``inference_state`` at a chosen frame.
      * Resuming is a **fresh** ``propagate(start_frame_idx=f)`` over the *same* mutated
        state — never ``reset_state`` to resume (that wipes all prompts; it's a
        fresh-chain call, done once in __init__).
      * A re-propagation after a correction re-tracks from the corrected frame onward and
        **overwrites** the stale frames it revisits in the accumulators (last write wins).

    GUI/auto-intervention sketch (not built here):
        sess = PropagationSession(vp, frames_dir, obj_id=1)
        sess.seed(prompts, anchor_idx)
        for fr in sess.propagate(reverse=False):      # fr: FrameResult (mask, conf, pred_iou)
            if looks_bad(fr): break                   # a QC predicate / human stop
        sess.add_points(fr.frame_idx, pts, labels)    # or sess.add_mask(fr.frame_idx, painted)
        for fr in sess.propagate(reverse=False, start_frame_idx=fr.frame_idx):
            ...                                        # resumes over the corrected state

    Accumulators (``video_segments`` / ``frame_conf`` / ``pred_iou``) are frame_idx-keyed
    and persist across calls, so the final dicts reflect the last mask written per frame.
    """

    def __init__(self, video_predictor, frames_dir: str, *, obj_id: int,
                 offload_video_to_cpu: bool = True):
        self.vp = video_predictor
        self.obj_id = obj_id
        # offload_video_to_cpu keeps VRAM bounded for long (~340-frame) chains.
        self.inference_state = video_predictor.init_state(
            video_path=frames_dir,                 # already a str (see prepare_video_frames)
            offload_video_to_cpu=offload_video_to_cpu,
            # offload_state_to_cpu=True,            # enable if OOM on very long chains
        )
        video_predictor.reset_state(self.inference_state)   # per-object scoping (liver pattern)

        self.video_segments: dict[int, dict[int, np.ndarray]] = {}
        self.frame_conf: dict[int, float] = {}
        self.pred_iou: dict[int, float] = {}
        self._iou_sink: dict[int, float] = {}
        self._restore_hook = _attach_iou_hook(video_predictor, self._iou_sink)
        self._closed = False

    # -- seeding / corrections -----------------------------------------------------
    def seed(self, prompts: Prompts, anchor_frame_idx: int, *,
             seed_box: bool = True, seed_points: bool = True,
             seed_negatives: bool = False, seed_mask: bool = False,
             mask_anchor: Optional[np.ndarray] = None) -> None:
        """Seed the anchor frame. Defaults (box + positive point) mirror the notebook's
        anchor seed exactly so masks reproduce.

        SAM2 treats MASK and POINTS/BOX as mutually-exclusive conditioning per frame
        (add_new_mask pops point_inputs and add_new_points_or_box pops mask_inputs — see
        sam2_video_predictor.py), so seed_mask=True takes the add_new_mask path and ignores
        box/points on the anchor frame. The box/points path composes any subset of
        {box, positive, negative} in a single add_new_points_or_box call.

        Negatives are the same same-z neighbour nodes build_prompts placed in _sam (valid in
        both crop and legacy paths — state.prompts stays in _sam). The mask seed needs the
        anchor mask in the PROPAGATION space (legacy _sam or tier-2 _pcrop); run_chain passes
        the right-space mask and guards the tier-1 crop_anchor case.
        """
        if seed_mask:
            if mask_anchor is None:
                raise ValueError("seed_mask=True requires mask_anchor (in the propagation space)")
            self.add_mask(int(anchor_frame_idx), mask_anchor)
            return

        pts = np.asarray(prompts.points_sam, dtype=np.float32)
        labels = np.asarray(prompts.labels, dtype=np.int32)
        keep = np.ones(len(labels), dtype=bool)
        if not seed_points:
            keep &= labels != 1            # drop positives
        if not seed_negatives:
            keep &= labels != 0            # drop negatives
        pts, labels = pts[keep], labels[keep]
        box = (np.asarray(prompts.box_sam, dtype=np.float32)
               if (seed_box and prompts.box_sam is not None) else None)
        if box is None and len(pts) == 0:
            raise ValueError("empty anchor seed: enable at least one of box / points / mask")
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=int(anchor_frame_idx),
            obj_id=self.obj_id,
            box=box,
            points=pts,
            labels=labels,
        )

    def add_points(self, frame_idx: int, points_sam, labels, *,
                   clear_old_points: bool = False) -> None:
        """Inject a point correction at ``frame_idx`` (refinement click(s); incl. negative
        labels). ``clear_old_points=False`` *adds* to the frame's existing prompts (the
        usual refine-click behaviour); pass True to replace them. Mutates inference_state —
        resume with ``propagate(start_frame_idx=frame_idx)``."""
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=int(frame_idx),
            obj_id=self.obj_id,
            points=np.asarray(points_sam, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int32),
            clear_old_points=clear_old_points,
        )

    def add_mask(self, frame_idx: int, mask_sam) -> None:
        """Inject a painted-mask correction at ``frame_idx`` (human-painted anchor or a
        mid-propagation fix). Mutates inference_state — resume with
        ``propagate(start_frame_idx=frame_idx)``."""
        self.vp.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=int(frame_idx),
            obj_id=self.obj_id,
            mask=np.asarray(mask_sam, dtype=bool),
        )

    # -- propagation ---------------------------------------------------------------
    def propagate(self, *, reverse: bool = False, start_frame_idx: Optional[int] = None,
                  max_frames: Optional[int] = None) -> Iterator[FrameResult]:
        """Lazy generator over propagated frames. Yields a FrameResult per frame and
        records it into the accumulators *before* yielding, so a caller that ``break``s
        early keeps every frame it saw. ``start_frame_idx`` resumes from a corrected
        frame; ``max_frames`` caps how far to track (both forwarded to SAM2)."""
        kw: dict = {"reverse": reverse}
        if start_frame_idx is not None:
            kw["start_frame_idx"] = int(start_frame_idx)
        if max_frames is not None:
            kw["max_frame_num_to_track"] = int(max_frames)
        for f, obj_ids, mask_logits in self.vp.propagate_in_video(self.inference_state, **kw):
            yield self._collect(f, obj_ids, mask_logits)

    def run_bidirectional(self) -> None:
        """Headless convenience: drain forward from the anchor, then reverse. Same call
        order as the pre-refactor two-loop drain, so masks are unchanged."""
        for _ in self.propagate(reverse=False):
            pass
        for _ in self.propagate(reverse=True):
            pass

    # -- internals -----------------------------------------------------------------
    def _collect(self, frame_idx, obj_ids, mask_logits) -> FrameResult:
        per_obj: dict[int, np.ndarray] = {}
        conf = float("nan")
        for i, oid in enumerate(obj_ids):
            lg = mask_logits[i].cpu().numpy()
            m = lg > 0.0
            per_obj[oid] = m
            if oid == self.obj_id:                 # confidence proxy for THIS chain's obj
                fg = lg[m]
                conf = float((1.0 / (1.0 + np.exp(-fg))).mean()) if fg.size else float("nan")
        fi = int(frame_idx)
        self.video_segments[fi] = per_obj
        self.frame_conf[fi] = conf
        piou = self._iou_sink.get(fi, float("nan"))   # set by the hook during this frame's inference
        self.pred_iou[fi] = piou
        return FrameResult(frame_idx=fi, masks=per_obj, logit_conf=conf, pred_iou=piou)

    def close(self) -> None:
        """Restore the patched ``_track_step``. Idempotent. Call when done (or use the
        session as a context manager)."""
        if not self._closed:
            try:
                self._restore_hook()
            finally:
                self._closed = True

    def __enter__(self) -> "PropagationSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def propagate(video_predictor, frames_dir: str, prompts: Prompts,
              anchor_frame_idx: int, *, obj_id: int, seed_negatives: bool = False,
              seed_box: bool = True, seed_points: bool = True, seed_mask: bool = False,
              mask_anchor: Optional[np.ndarray] = None,
              subtimings: Optional[dict] = None
              ) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, float], dict[int, float]]:
    """Seed the anchor, propagate bidirectionally, collect masks. Headless straight-through
    driver over PropagationSession (the interruptible primitive); this is what run_chain
    uses. Mid-propagation halt/correct/resume is the session's job, not this function's.

    Returns
    -------
    (video_segments, frame_conf, pred_iou)
        video_segments : {frame_idx: {obj_id: mask_sam bool}}
        frame_conf     : {frame_idx: float}, mean-foreground-sigmoid proxy (the
                         `logit_conf` diagnostic column).
        pred_iou       : {frame_idx: float}, SAM2's mask-decoder predicted IoU, now
                         actually captured. NaN per
                         frame only if the hook couldn't read it. run_chain maps this
                         frame_idx -> catmaid_z and hands it to run_qc, where it populates
                         the `pred_iou` QC column and becomes the 4th flag-rule signal
                         (was inert while NaN). NOTE: enabling this signal changes the
                         flag/queue distribution, clear/re-score the manifest after the
                         switch (the mixed-threshold discipline), and set
                         cfg.qc_pred_iou_min <= 0 to record-but-not-flag.

    Lift from: 'init_state' cell + 'Anchor and propagate bidirectionally' cell.
    """
    _t = perf_counter()
    session = PropagationSession(video_predictor, frames_dir, obj_id=obj_id)
    if subtimings is not None:
        subtimings["jpeg_load"] = perf_counter() - _t      # dominated by SAM2's frame decode
    try:
        session.seed(prompts, anchor_frame_idx, seed_box=seed_box,
                     seed_points=seed_points, seed_negatives=seed_negatives,
                     seed_mask=seed_mask, mask_anchor=mask_anchor)
        _t = perf_counter()
        session.run_bidirectional()
        if subtimings is not None:
            subtimings["propagate_only"] = perf_counter() - _t
        return session.video_segments, session.frame_conf, session.pred_iou
    finally:
        session.close()


def save_masks(video_segments: dict[int, dict[int, np.ndarray]],
               frame_to_z: dict[int, int], out_dir: Path, *,
               obj_id: int, mask_space_downscale: int) -> int:
    """Write one 0/255 uint8 PNG per frame at the canonical mask space.

    Returns count written. Files are `mask_<catmaid_z:04d>.png`, single-channel,
    255 = inside the neurite, 0 = background -> the notebook's exact format, so the
    masks are directly viewable AND pixel-comparable to the notebook output (the
    done-check). qc._load_binary reads `arr > 0`, so this stays fully
    compatible with compute_metrics.

    Why NOT qc.save_masks here: that writer stores uint16 *instance labels*
    (foreground pixel value == obj_id). For a single object obj_id is 1, and value
    1 in a 16-bit image is visually indistinguishable from black -> it looks empty
    and is destroyed by any 16->8-bit conversion, which is exactly the "empty
    masks" confusion. Instance-label encoding is a multi-object concern; adopt it
    later when aggregating several objects per neuron, not now.

    No resample happens: masks already live at _sam, and under the canonical rule
    (scale == save_downscale) that IS the on-disk space. `mask_space_downscale` is
    accepted for signature stability / future divergence but isn't applied here;
    if you ever set save_downscale != scale you'll need the source scale too, and
    the resample should live in this one function.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    for frame_idx in sorted(video_segments):
        seg = video_segments[frame_idx]
        if obj_id not in seg:
            skipped += 1
            continue
        raw = np.asarray(seg[obj_id])
        mask_hw = raw[0] if raw.ndim == 3 else raw     # (H_sam, W_sam) bool
        png = (mask_hw.astype("uint8") * 255)          # 0 / 255, viewable
        catmaid_z = frame_to_z[frame_idx]
        cv2.imwrite(str(out_dir / f"mask_{catmaid_z:04d}.png"), png)
        written += 1

    print(f"[save_masks] wrote {written} masks ({skipped} frames skipped, "
          f"obj_id {obj_id} absent) -> {out_dir}  (0/255 uint8, named by catmaid_z)")
    return written

def postprocess_mask(mask_sam: np.ndarray, *, open_px: int = 1, close_px: int = 1,
                     keep_largest_cc: bool = True, fill_holes: bool = True) -> np.ndarray:
    """Deterministic, model-free cleanup of one propagated mask, in _sam space.

    Masks are downscaled then NN-upscaled, so they come out blocky / speckled /
    holey while true neurite borders are smooth -> cleanup priors are safe.
    Order: open (despeckle) -> close (bridge grid gaps) -> largest-CC (drop
    detached fragments, generalising box_from_mask's pick) -> fill holes. Keep
    kernels small or thin neurites erode away. Empty in -> empty out.

    NB keep_largest_cc drops genuinely-split components too -> fine for a box, but
    it can silently erase a real second process in the *saved* mask, and it can
    paper over a propagation failure QC is meant to flag. Run QC *after* this so
    QC scores the delivered mask, and watch the flag distribution when enabling.
    """
    from scipy import ndimage
    from skimage.measure import label as cc_label

    m = np.asarray(mask_sam).astype(bool)
    if not m.any():
        return m
    if open_px > 0:
        m = ndimage.binary_opening(m, iterations=open_px)
    if close_px > 0:
        m = ndimage.binary_closing(m, iterations=close_px)
    if keep_largest_cc and m.any():
        lbl = cc_label(m, connectivity=2)
        sizes = np.bincount(lbl.ravel())[1:]          # drop background
        if sizes.size:
            m = lbl == (1 + int(np.argmax(sizes)))
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    return m

def run_qc(masks_dir: Path, skeleton: pd.DataFrame, *,
           frame_to_z: dict[int, int],
           frame_conf: Optional[dict[int, float]],
           pred_iou: Optional[dict[int, float]] = None,
           cfg: PipelineConfig,
           qc_csv_path: Optional[Path] = None,
           crop_window: Optional["alignment.CropWindow"] = None,
           ) -> tuple[dict, list[int], str]:
    """Compute QC over the saved masks, write qc.csv, return (summary, triage_z, status).

    QC runs over the
    just-saved chain (joining the inline-captured confidence), produces flags, and
    drives the chain's verdict -> all headless, no human. It still reads the PNGs
    back off disk rather than scoring inside the propagate loop; that fully-inline,
    interleaved form is only required for *halt-and-re-prompt*, which is not built
    yet. So this is "QC moved into the run," not yet "QC moved into the propagation loop."

    Signals and the composite flag/intervene rule come straight from
    ``qc.compute_metrics`` (single source of truth); thresholds come from ``cfg``.

    Parameters
    ----------
    skeleton : DataFrame
        The skeleton of *this chain only* (columns z, x_tif, y_tif), NOT the whole
        neuron. This matters: a neuron like AVAL is many chains, so its nodes cross
        a given z at several xy positions and their centroid lands off any single
        process -> using it makes containment fail on every frame (the AVAL 100%-flag
        bug). Filtering to the chain's own nodes gives a meaningful per-z probe.

    Returns
    -------
    qc_summary : dict (json-safe)   -> counts + worst frames, for ChainState
    triage_z   : list[int]          -> CATMAID-z of every flagged frame (the queue;
                                      z-keyed to match qc, mask filenames, and
                                      review.load_chain's triage_is_z default)
    status     : "done" | "flagged"
    """
    from sam2_utils import qc   # lazy: keeps pipeline import free of qc's heavy deps

    # Invariant: pipeline.save_masks writes masks at _sam (scale) and never
    # resamples, so the on-disk mask space IS `scale`. qc.compute_metrics divides
    # the _tif skeleton by `save_downscale` to land in mask space, so the two must
    # be equal or QC silently mis-locates every node. The
    # canonical rule already enforces this; the guard turns a future divergence
    # from a silent wrong-QC run into a loud failure. If you ever want resampled,
    # higher-res Blender masks, make save_masks resample first, then relax this.
    # The scale==save_downscale guard protects the _sam node lookup (skeleton / scale).
    # Tier-2 masks live in _pcrop, where the node lookup goes through crop_window
    # instead of / save_downscale, so the guard does not apply, skip it when a
    # crop_window is supplied (and the node mapping is overridden below).
    if crop_window is None and cfg.scale != cfg.save_downscale:
        raise ValueError(
            f"run_qc: scale ({cfg.scale}) != save_downscale ({cfg.save_downscale}), "
            "but pipeline.save_masks does not resample, so the on-disk masks are at "
            "`scale`. QC would divide the skeleton by save_downscale and mis-locate "
            "every node. Keep save_downscale == scale (canonical), or make save_masks "
            "resample to save_downscale before changing this."
        )

    # pred_iou comes in frame_idx-keyed (from PropagationSession); compute_metrics is
    # z-indexed, so remap. Once joined, the `pred_iou` column becomes the 4th flag-rule
    # signal (cfg.qc_pred_iou_min); it was inert while NaN. See propagate()'s note re:
    # clearing the manifest after enabling it.
    pred_iou_z = None
    if pred_iou:
        pred_iou_z = {frame_to_z[fi]: v for fi, v in pred_iou.items()
                      if fi in frame_to_z}

    # In _pcrop the node-containment radius is rescaled by scale/crop_scale (same
    # space_ratio run_chain applies to the anchor gate), so the physical tolerance
    # matches the _sam path; compute_metrics maps nodes _tif->_pcrop via crop_window.
    dilation_px = cfg.qc_skeleton_dilation_px
    if crop_window is not None:
        dilation_px = int(round(cfg.qc_skeleton_dilation_px
                                * crop_window.sam_scale / crop_window.crop_scale))

    df = qc.compute_metrics(
        masks_dir,
        skeleton=skeleton,
        scale=cfg.scale,
        save_downscale=cfg.save_downscale,
        pred_iou=pred_iou_z,
        skeleton_dilation_px=dilation_px,
        area_ratio_bounds=cfg.qc_area_ratio_bounds,
        temporal_iou_min=cfg.qc_temporal_iou_min,
        pred_iou_min=cfg.qc_pred_iou_min,
        crop_window=crop_window,
    )

    # Attach the inline confidence proxy as a *diagnostic* column (z-keyed).
    # Deliberately NOT named pred_iou and NOT in the flag rule -> see propagate().
    if frame_conf:
        z_conf = {frame_to_z[fi]: c for fi, c in frame_conf.items()
                  if fi in frame_to_z}
        df["logit_conf"] = df.index.map(lambda z: z_conf.get(int(z), float("nan")))

    # The human triage queue is the frames at/above the configured severity
    # (flag_count >= qc_triage_min_signals; default 2 = intervene-level). `flag`
    # (>=1 signal) stays in the row as a diagnostic: single-signal flags are kept on
    # disk for labels, just not surfaced to a human. Persisted to qc.csv so the
    # cross-chain rollup (batch.build_triage_queue) can filter on the artifact alone.
    df["queue"] = df["flag_count"] >= cfg.qc_triage_min_signals

    if qc_csv_path is not None:
        qc_csv_path = Path(qc_csv_path)
        qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(qc_csv_path)   # index = z

    n = int(len(df))
    n_flag = int(df["flag"].sum())
    n_int = int(df["intervene"].sum())
    n_queue = int(df["queue"].sum())
    n_noskel = int((df["skeleton_contained"] == False).sum())   # noqa: E712
    n_skel_na = int(df["skeleton_contained"].isna().sum())
    triage_z = sorted(int(z) for z in df.index[df["queue"]])

    # worst queue frames first, for a quick human glance from state.json alone
    worst = (df[df["queue"]].sort_values("flag_count", ascending=False).head(10))
    worst_frames = [
        {
            "z": int(z),
            "flag_count": int(r["flag_count"]),
            "area_ratio": (None if pd.isna(r["area_ratio"]) else round(float(r["area_ratio"]), 3)),
            "temporal_iou": (None if pd.isna(r["temporal_iou"]) else round(float(r["temporal_iou"]), 3)),
            "skeleton_contained": bool(r["skeleton_contained"]),
        }
        for z, r in worst.iterrows()
    ]

    qc_summary = {
        "n_frames": n,
        "n_flagged": n_flag,          # all >=1-signal flags (diagnostic; kept for labels)
        "n_queue": n_queue,           # frames surfaced to a human (the triage gate)
        "n_intervene": n_int,
        "n_missing_skel": n_noskel,
        "n_skel_not_assessable": n_skel_na,
        "flag_rate": (round(n_flag / n, 4) if n else 0.0),
        "queue_rate": (round(n_queue / n, 4) if n else 0.0),
        "thresholds": {
            "area_ratio_bounds": list(cfg.qc_area_ratio_bounds),
            "temporal_iou_min": cfg.qc_temporal_iou_min,
            "pred_iou_min": cfg.qc_pred_iou_min,
            "skeleton_dilation_px": cfg.qc_skeleton_dilation_px,
            "triage_min_signals": cfg.qc_triage_min_signals,
        },
        "worst_frames": worst_frames,
    }

    # chain verdict keyed on the SAME queue definition as the frame queue, so the two
    # never disagree. Behaviour-preserving at defaults: qc_triage_min_signals=2 makes
    # n_queue == n_intervene, so this is identical to the prior
    # `n_int >= qc_intervene_to_flag_chain` rule.
    status = "flagged" if n_queue >= cfg.qc_intervene_to_flag_chain else "done"
    return qc_summary, triage_z, status


# =============================================================================
# Serialization  (ChainState <-> state.json)
# =============================================================================
# Three things plain json won't handle on its own:
#   - numpy arrays in Prompts (points / labels / box) -> lists on dump, arrays on load
#   - frame_to_z keys come back as strings           -> cast to int on load
#   - Path fields in PipelineConfig                   -> str on dump, Path on load

def _prompts_to_dict(p: Optional[Prompts]) -> Optional[dict]:
    if p is None:
        return None
    return {
        "points_sam": np.asarray(p.points_sam).tolist(),
        "labels": np.asarray(p.labels).tolist(),
        "box_sam": None if p.box_sam is None else np.asarray(p.box_sam).tolist(),
    }


def _prompts_from_dict(d: Optional[dict]) -> Optional[Prompts]:
    if d is None:
        return None
    box = d.get("box_sam")
    return Prompts(
        points_sam=np.array(d["points_sam"], dtype=float),
        labels=np.array(d["labels"], dtype=int),
        box_sam=None if box is None else np.array(box, dtype=np.float32),
    )


def _config_to_dict(c: PipelineConfig) -> dict:
    d = asdict(c)
    d["output_root"] = None if c.output_root is None else str(c.output_root)
    d["frames_root"] = None if c.frames_root is None else str(c.frames_root)
    return d


def _config_from_dict(d: Optional[dict]) -> PipelineConfig:
    d = dict(d or {})
    if d.get("output_root") is not None:
        d["output_root"] = Path(d["output_root"])
    if d.get("frames_root") is not None:
        d["frames_root"] = Path(d["frames_root"])
    return PipelineConfig(**d)


def state_to_dict(state: ChainState) -> dict:
    """Plain-json-safe dict view of a ChainState."""
    ftz = state.frame_to_z
    return {
        "neuron": state.neuron,
        "chain_idx": state.chain_idx,
        "status": state.status,
        "anchor_node_id": state.anchor_node_id,
        "anchor_catmaid_z": state.anchor_catmaid_z,
        "anchor_frame_idx": state.anchor_frame_idx,
        "prompts": _prompts_to_dict(state.prompts),
        "image_score": None if state.image_score is None else float(state.image_score),
        "anchor_score": state.anchor_score,
        "frames_dir": state.frames_dir,
        "frame_to_z": None if ftz is None else {str(k): int(v) for k, v in ftz.items()},
        "n_frames": state.n_frames,
        "crop_window": state.crop_window,
        "fell_back_to_sam": bool(state.fell_back_to_sam),
        "fellback_reason": state.fellback_reason,
        "crop_image_score": (None if state.crop_image_score is None
                             else float(state.crop_image_score)),
        "crop_anchor_score": state.crop_anchor_score,
        "qc_summary": state.qc_summary,
        "triage_frames": list(state.triage_frames),
        "obj_id": state.obj_id,
        "config": _config_to_dict(state.config),
        "phase_seconds": dict(state.phase_seconds),
        "phase_subseconds": dict(state.phase_subseconds),
    }


def state_from_dict(d: dict) -> ChainState:
    ftz = d.get("frame_to_z")
    return ChainState(
        neuron=d["neuron"],
        chain_idx=d["chain_idx"],
        status=d.get("status", "pending"),
        anchor_node_id=d.get("anchor_node_id"),
        anchor_catmaid_z=d.get("anchor_catmaid_z"),
        anchor_frame_idx=d.get("anchor_frame_idx"),
        prompts=_prompts_from_dict(d.get("prompts")),
        image_score=d.get("image_score"),
        anchor_score=d.get("anchor_score"),
        frames_dir=d.get("frames_dir"),
        frame_to_z=None if ftz is None else {int(k): int(v) for k, v in ftz.items()},
        n_frames=d.get("n_frames"),
        crop_window=d.get("crop_window"),
        fell_back_to_sam=bool(d.get("fell_back_to_sam", False)),
        fellback_reason=d.get("fellback_reason"),
        crop_image_score=d.get("crop_image_score"),
        crop_anchor_score=d.get("crop_anchor_score"),
        qc_summary=d.get("qc_summary"),
        triage_frames=list(d.get("triage_frames", [])),
        obj_id=d.get("obj_id", 1),
        config=_config_from_dict(d.get("config")),
        phase_seconds=dict(d.get("phase_seconds", {}) or {}),
        phase_subseconds=dict(d.get("phase_subseconds", {}) or {}),
    )


def save_state(state: ChainState, path: str | Path) -> Path:
    """Serialize a ChainState to state.json (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_to_dict(state), indent=2))
    return path


def load_state(path: str | Path) -> ChainState:
    """Reload a ChainState from state.json."""
    return state_from_dict(json.loads(Path(path).read_text()))


# =============================================================================
# Thin driver  (the ~20-line regression target)
# =============================================================================

def run_chain(state: ChainState, *, image_predictor, video_predictor,
              annotate_df: pd.DataFrame, chain: dict,
              on_video_phase: Optional[Callable[[], None]] = None,
              frame_store: Optional[FrameStore] = None) -> ChainState:
    """Run one chain end-to-end by composing the phases above.

    No new behavior vs. the notebook -> this just makes the call order explicit and
    threads ChainState through, so you can see exactly which field each phase fills.
    All tunables come from state.config; the state carries its own settings so it
    stays self-contained for serialize / resume. Getting this to reproduce the
    AVAL masks is the done condition.

    `on_video_phase` is an optional callback fired once, after the image phase and
    before video propagation. The driver passes diagnostics.cleanup_vram here so
    VRAM is reclaimed between phases (the notebook does this; skipping it leaks
    VRAM). Keeping it a callback lets pipeline.py stay free of a torch/diagnostics
    import -> the library doesn't decide *how* to clean up, the driver does.

    `frame_store` selects the EM source (the worm-coupled seam): None = the target
    worm's tif stack (TifFrameStore, the default path); pass a different store (e.g. the GT
    per-slice PNG store) to run the same pipeline on another worm. The skeleton->image
    transform is *not* threaded here, it's baked into annotate_df's x_tif/y_tif by the
    caller (catmaid_to_tif for the target worm; the per-section registration for GT).
    """
    cfg = state.config
    fs = frame_store or TifFrameStore()
    state.status = "running"

    tag = f"{state.neuron} chain {state.chain_idx:02d}"

    timings: dict[str, float] = {}
    subtimings: dict[str, float] = {}
    _clk = {"t": perf_counter(), "label": None}

    def _step(n: int, label: str) -> None:
        now = perf_counter()
        if _clk["label"] is not None:
            timings[_clk["label"]] = now - _clk["t"]
        _clk["t"], _clk["label"] = now, label
        print(f"\n[{tag}] step {n}/9 -> {label}", flush=True)
        
    def _finish() -> None:
        """Close out the in-flight phase and stamp timings onto the state.
        Call before every return so phase_seconds is set on all paths."""
        if _clk["label"] is not None:
            timings[_clk["label"]] = perf_counter() - _clk["t"]
            _clk["label"] = None
        state.phase_seconds = timings
        state.phase_subseconds = subtimings

    # 1. anchor
    _step(1, "select anchor")
    state.anchor_node_id, state.anchor_catmaid_z = select_anchor(chain, annotate_df)
    print(f"    anchor node {state.anchor_node_id}  (CATMAID z={state.anchor_catmaid_z})")

    # 2-5. Anchor phase (load frame -> prompts -> image-mode predict -> gate + box),
    #    factored into a closure so the tier-2 SAFETY fallback can re-run it in
    #    the plain _sam path when the per-chain crop yields a poor anchor (the c02
    #    over-zoom mode: a tiny/over-zoomed window where SAM2 loses inter-frame context
    #    and the anchor mask collapses). use_chain_crop=False reproduces the legacy /
    #    tier-1 _sam path EXACTLY (build_prompts in _sam, crop_anchor honoured), so the
    #    baseline and the gate's observational role are unchanged on non-tier-2 runs.
    #    Returns (cw_chain|None, anchor verdict, box_present); sets state.prompts /
    #    image_score / anchor_score / crop_window. (build_prompts is inside so the _sam
    #    rerun rebuilds the seed in its own space; running it twice is deterministic.)
    def _anchor_phase(use_chain_crop: bool):
        cw_chain_l = None
        # 2. anchor frame.
        _step(2, "load anchor frame")
        state.crop_window = None
        if use_chain_crop:
            image_full, full_hw = load_frame_sam(state.anchor_catmaid_z, scale=1, frame_store=fs)   # _tif
            image_sam = None
            # chain_crop_from_mask: grow the window to contain the already-saved _sam
            # mask, not just the skeleton centerline (fixes the cell-edge clip). Only
            # when the prior masks are _sam — a prior tier-2 (state.json has a
            # crop_window) stores _pcrop masks whose bbox is the wrong space, so decline.
            extra_box_tif = None
            if cfg.chain_crop_from_mask:
                chain_dir = Path(cfg.output_root) / state.neuron / f"chain_{state.chain_idx:02d}"
                sp = chain_dir / "state.json"
                prior = load_state(sp) if sp.exists() else None
                if prior is not None and getattr(prior, "crop_window", None):
                    print("    [chain_crop_from_mask] prior masks are _pcrop (tier-2); "
                          "sizing from skeleton bbox")
                else:
                    queued_z = _prior_queued_z(chain_dir / "qc.csv")
                    box_px = (mask_union_box_px((chain_dir / "masks"), exclude_z=queued_z)
                              if (chain_dir / "masks").exists() else None)
                    if box_px is None:
                        print("    [chain_crop_from_mask] no usable _sam mask; "
                              "sizing from skeleton bbox")
                    else:
                        x0, y0, x1, y1 = box_px       # _sam px -> _tif (+1 far corner = pixel extent)
                        extra_box_tif = (x0 * cfg.scale, y0 * cfg.scale,
                                         (x1 + 1) * cfg.scale, (y1 + 1) * cfg.scale)
                        print(f"    [chain_crop_from_mask] _sam mask bbox -> _tif "
                              f"{tuple(int(v) for v in extra_box_tif)} "
                              f"(union w/ skeleton, excl {len(queued_z)} queued frame(s))")
            cw_chain_l = chain_crop_window(chain, annotate_df, cfg=cfg, image_hw_tif=full_hw,
                                           extra_box_tif=extra_box_tif)
            state.crop_window = cw_chain_l.to_dict()
            print(f"    full-res {full_hw[1]}x{full_hw[0]} -> _pcrop window "
                  f"{cw_chain_l.size_tif[0]}x{cw_chain_l.size_tif[1]}px @ crop_scale "
                  f"{cw_chain_l.crop_scale} -> {cw_chain_l.crop_hw[1]}x{cw_chain_l.crop_hw[0]} input")
        elif cfg.crop_anchor:
            image_full, full_hw = load_frame_sam(state.anchor_catmaid_z, scale=1, frame_store=fs)   # _tif
            image_sam = None
            print(f"    full-res frame {full_hw[1]}x{full_hw[0]} -> {cfg.crop_size_tif}px "
                  f"_tif crop @ crop_scale {cfg.crop_scale}")
        else:
            image_sam, full_hw = load_frame_sam(state.anchor_catmaid_z, scale=cfg.scale, frame_store=fs)
            image_full = None
            print(f"    _sam frame {image_sam.shape[1]}x{image_sam.shape[0]} "
                  f"(full {full_hw[1]}x{full_hw[0]}, scale {cfg.scale})")

        # 3. prompts (always built in _sam -> they seed the video positive point; the
        #    crop path remaps a copy into _crop for the anchor prediction).
        _step(3, "build prompts")
        state.prompts = build_prompts(state.anchor_node_id, state.anchor_catmaid_z,
                                      annotate_df, scale=cfg.scale,
                                      k_max_neg=cfg.k_max_neg, neg_radius=cfg.neg_radius)
        n_pos = int((state.prompts.labels == 1).sum())
        n_neg = int((state.prompts.labels == 0).sum())
        print(f"    {n_pos} positive + {n_neg} negative point(s)")

        # 4. image mode -> on a high-res crop by default, else scale-8.
        _step(4, "image-mode prediction")
        # Space-relative tolerances, computed ONCE up front so the multimask pick (step 4)
        # and the anchor gate (step 5) score with the same radius/area bounds in the same
        # space as the mask: 1 _sam px = scale/crop_scale _crop px, so the contain radius
        # + box margin rescale under the crop. area_frac bounds are frame-fractions the
        # crop config already tunes, so they pass through unscaled.
        # tier-2 uses the chain window's (possibly bumped) crop_scale; tier-1 uses crop_scale.
        crop_active = use_chain_crop or cfg.crop_anchor
        eff_crop_scale = (cw_chain_l.crop_scale if use_chain_crop else cfg.crop_scale)
        space_ratio = (cfg.scale / eff_crop_scale) if crop_active else 1.0
        contain_r = int(round(cfg.qc_skeleton_dilation_px * space_ratio))
        margin_local = int(round(cfg.box_margin * space_ratio))
        area_bounds = (cfg.gate_min_area_frac, cfg.gate_max_area_frac)
        if crop_active:
            # cw=cw_chain_l -> tier-2 (image phase runs in the SAME window the chain
            # propagates in); cw=None -> tier-1 (a fresh window centred on the node).
            mask_anchor, state.image_score, cw, prompts_anchor = anchor_crop_predict(
                image_predictor, image_full, full_hw, state.anchor_node_id,
                state.prompts, annotate_df, scale=cfg.scale,
                crop_size_tif=cfg.crop_size_tif, crop_scale=eff_crop_scale, cw=cw_chain_l,
                multimask=cfg.multimask_anchor, select_contain_radius_px=contain_r,
                select_area_bounds=area_bounds)
        else:
            mask_anchor, state.image_score, _logits = image_predict(
                image_predictor, image_sam, state.prompts,
                multimask=cfg.multimask_anchor, select_contain_radius_px=contain_r,
                select_area_bounds=area_bounds)
            cw, prompts_anchor = None, state.prompts
        image_hw_anchor = mask_anchor.shape[:2]
        if cfg.multimask_anchor:
            print(f"    multimask auto-select on (3 candidates -> 1)")
        print(f"    mask {int(mask_anchor.sum())} px  |  score {state.image_score:.4f}"
              + (f"  | {'_pcrop' if use_chain_crop else '_crop'} "
                 f"{image_hw_anchor[1]}x{image_hw_anchor[0]}" if crop_active else ""))

        # 5. anchor gate + box (empty mask -> box_present False). The gate scores the
        #    anchor mask in WHATEVER space it lives in (_crop under the crop path, _sam
        #    in legacy); score_anchor is space-agnostic, we just feed it matching coords
        #    + a space-correct contain radius. Still OBSERVATIONAL for flagging -> the
        #    only branch it drives is the tier-2->_sam fallback, decided below.
        _step(5, "box from mask")
        anchor = score_anchor(
            mask_anchor, prompts_anchor, image_hw_sam=image_hw_anchor,
            contain_radius_px=contain_r,
            min_area_frac=cfg.gate_min_area_frac,
            max_area_frac=cfg.gate_max_area_frac,
            min_largest_cc_frac=cfg.gate_min_largest_cc_frac,
        )
        state.anchor_score = _anchor_score_to_dict(anchor)
        _contained = "n/a" if anchor.contained is None else anchor.contained
        print(f"    anchor gate: {'PASS' if anchor.passed else 'FAIL'}  "
              f"(cc={anchor.n_components} lcc={anchor.largest_cc_frac:.2f} "
              f"area_frac={anchor.area_frac:.5f} contained={_contained})"
              + (f"  reasons: {', '.join(anchor.reasons)}" if anchor.reasons else ""))

        # %-of-bbox box pad (seed ablation) when seed_box=="frac"; frac scales with the
        # object so it needs no space rescale (margin_local, the fixed px pad, already does).
        box_frac = cfg.box_margin_frac if cfg.seed_box == "frac" else 0.0
        box_local = box_from_mask(mask_anchor, margin=margin_local, margin_frac=box_frac,
                                  image_hw_sam=image_hw_anchor)
        box_present = box_local is not None
        if not box_present:
            print("    empty anchor mask")
        elif use_chain_crop:
            # tier-2: the WHOLE chain propagates in _pcrop, so the seed stays there — the
            # points (prompts_anchor, already mapped _sam->_pcrop) and box are crop coords,
            # NOT mapped back to _sam. state.prompts is thus the _pcrop seed (box_sam/points_sam
            # hold _pcrop px; the names are legacy). propagate() feeds them onto the _pcrop frames.
            state.prompts = Prompts(
                points_sam=np.asarray(prompts_anchor.points_sam, dtype=float),
                labels=np.asarray(prompts_anchor.labels, dtype=int),
                box_sam=np.asarray(box_local, dtype=np.float32))
            print(f"    box (xyxy, _pcrop): {state.prompts.box_sam.astype(int).tolist()}")
        else:
            # tier-1 / legacy: map the crop box back to _sam for the video seed (or use as-is).
            box = box_local if cw is None else cw.box_crop_to_sam(box_local)
            state.prompts.box_sam = np.asarray(box, dtype=np.float32)
            print(f"    box (xyxy, _sam): {state.prompts.box_sam.astype(int).tolist()}")
        # mask_anchor is in the anchor's image space: _pcrop (tier-2) or _sam (legacy) ==
        # the propagation space, but _crop (tier-1 crop_anchor) != _sam. Only return it as a
        # seedable mask when it matches the frames the chain will propagate over.
        mask_seedable = mask_anchor if (use_chain_crop or not crop_active) else None
        return cw_chain_l, anchor, box_present, mask_seedable

    def _anchor_poor(anchor, box_present: bool) -> bool:
        """The crop anchor is untrustworthy -> tier-2 should fall back to _sam."""
        if not box_present:                        # empty mask in the crop
            return True
        if not anchor.passed:                      # gate fired (area / frag / noskel)
            return True
        floor = cfg.chain_crop_min_image_score
        if floor > 0 and (state.image_score or 0.0) < floor:
            return True
        return False

    cw_chain, anchor, box_present, mask_anchor_seed = _anchor_phase(use_chain_crop=cfg.chain_crop)
    if cfg.chain_crop and cfg.chain_crop_fallback and _anchor_poor(anchor, box_present):
        reasons = []
        if not box_present: reasons.append("empty-mask")
        if not anchor.passed: reasons.append(f"gate({','.join(anchor.reasons) or 'fail'})")
        if cfg.chain_crop_min_image_score > 0 and (state.image_score or 0.0) < cfg.chain_crop_min_image_score:
            reasons.append(f"score<{cfg.chain_crop_min_image_score}")
        print(f"    [tier-2 fallback] poor _pcrop anchor [{', '.join(reasons)}] "
              f"-> re-running this chain in the plain _sam path")
        state.fell_back_to_sam = True
        # capture the CROP-pass diagnostics BEFORE the _sam recovery pass below overwrites
        # state.image_score / state.anchor_score — else the failing reason is lost (only the
        # run log had it), and the final state.json shows the healthy _sam recovery instead.
        state.fellback_reason = ", ".join(reasons)
        state.crop_image_score = state.image_score
        state.crop_anchor_score = _anchor_score_to_dict(anchor)
        cw_chain, anchor, box_present, mask_anchor_seed = _anchor_phase(use_chain_crop=False)

    # Effective space for the rest of the run: tier-2 only if we didn't fall back.
    eff_chain_crop = cfg.chain_crop and not state.fell_back_to_sam

    if not box_present:
        print("    empty anchor mask -> flagging chain for human review")
        state.status = "flagged"
        _finish()
        return state                          # later: re-pick anchor before flagging

    # Mask seed (seed ablation): valid only if the anchor mask is in the propagation space.
    # _anchor_phase returns None for the tier-1 crop_anchor case (anchor in _crop != _sam).
    if cfg.seed_mask and mask_anchor_seed is None:
        raise ValueError(
            "seed_mask=True but the anchor mask is not in the propagation space "
            "(tier-1 crop_anchor produces a _crop mask). Use crop_anchor=False or chain_crop=True.")

    # free the image embedding before video propagation (notebook does this).
    image_predictor.reset_predictor()
    if on_video_phase is not None:
        on_video_phase()

    # 6. video frames -> tier-2 crops each frame to the chain window (_pcrop); else
    #    the shared scale-8 cache + per-chain link view (_sam). eff_chain_crop (not
    #    cfg.chain_crop) so a chain that fell back to _sam preps _sam frames.
    _step(6, "prepare video frames")
    if eff_chain_crop:
        (state.frames_dir, state.frame_to_z,
         state.anchor_frame_idx, state.n_frames) = prepare_chain_crop_frames(
            chain, annotate_df, cw_chain, frames_root=cfg.frames_root,
            anchor_catmaid_z=state.anchor_catmaid_z,
            neuron=state.neuron, chain_idx=state.chain_idx, frame_store=fs)
    else:
        (state.frames_dir, state.frame_to_z,
         state.anchor_frame_idx, state.n_frames) = prepare_video_frames(
            chain, annotate_df, scale=cfg.scale, frames_root=cfg.frames_root,
            anchor_catmaid_z=state.anchor_catmaid_z,
            neuron=state.neuron, chain_idx=state.chain_idx, frame_store=fs)
    print(f"    {state.n_frames} frames  (anchor frame_idx={state.anchor_frame_idx})")

    # 7. propagate (seed per the ablation spec; defaults = box+positive = baseline seed)
    _step(7, "propagate (bidirectional)")
    video_segments, frame_conf, pred_iou = propagate(
        video_predictor, state.frames_dir, state.prompts,
        state.anchor_frame_idx, obj_id=state.obj_id,
        seed_box=(cfg.seed_box != "none"), seed_points=cfg.seed_points,
        seed_negatives=cfg.seed_negatives, seed_mask=cfg.seed_mask,
        mask_anchor=(mask_anchor_seed if cfg.seed_mask else None),
        subtimings=subtimings)

    # 8. (post-process, then) save at canonical space. Cleanup is a phase
    #    folded in here so it lands BEFORE QC (step 9) reads the masks back.
    _step(8, "save masks")
    if cfg.postprocess_masks:
        n_changed = 0
        for seg in video_segments.values():
            if state.obj_id in seg:
                cleaned = postprocess_mask(
                    seg[state.obj_id],
                    open_px=cfg.postproc_open_px, close_px=cfg.postproc_close_px,
                    keep_largest_cc=cfg.postproc_keep_largest_cc,
                    fill_holes=cfg.postproc_fill_holes)
                n_changed += int(not np.array_equal(cleaned, seg[state.obj_id]))
                seg[state.obj_id] = cleaned
        print(f"    post-processed {n_changed}/{len(video_segments)} masks")
    chain_dir = Path(cfg.output_root) / state.neuron / f"chain_{state.chain_idx:02d}"
    out_dir = chain_dir / "masks"
    save_masks(video_segments, state.frame_to_z, out_dir,
               obj_id=state.obj_id, mask_space_downscale=cfg.save_downscale)

    # 9. QC + flagging: score the run, write qc.csv, set the chain verdict.
    _step(9, "qc + flag")
    # this chain's own skeleton (NOT the whole neuron -> see run_qc docstring)
    chain_node_ids = {str(n) for n in chain["nodes"]}
    skel_chain = annotate_df[
        annotate_df["node_id"].astype(str).isin(chain_node_ids)
    ][["z", "x_tif", "y_tif"]]
    state.qc_summary, state.triage_frames, state.status = run_qc(
        out_dir, skel_chain,
        frame_to_z=state.frame_to_z,
        frame_conf=frame_conf, pred_iou=pred_iou, cfg=cfg,
        qc_csv_path=chain_dir / "qc.csv",
        crop_window=cw_chain,            # tier-2: score in _pcrop (None -> _sam)
    )
    s = state.qc_summary
    print(f"    {s['n_flagged']}/{s['n_frames']} flagged "
          f"({s['n_queue']} queued)"
          f"({s['flag_rate']:.0%}), {s['n_intervene']} intervene "
          f"-> status '{state.status}'")
    _finish()
    return state