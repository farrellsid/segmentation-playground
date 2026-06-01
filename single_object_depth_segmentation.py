"""
single_object_depth_segmentation.py
===================================

Headless refactor of `single_object_depth_segmentation_.ipynb`.

End-to-end single-object (single-neuron) segmentation across multiple EM frames
using `sam2_utils`:

    1. Set up SAM2 (handled by sam2_utils.setup.build_predictor).
    2. Load CATMAID annotations + chains from disk (CSV / JSON) and apply the
       stack -> tif affine.
    3. For each maximal linear chain (MLC) of the target cell:
         a. IMAGE MODE  -- pick a central frame, build CATMAID-derived prompts,
                           predict an anchor mask, derive an xyxy box.
         b. VIDEO MODE  -- write a downscaled JPEG sequence, anchor the box+point,
                           propagate bidirectionally, collect per-frame masks.
       VRAM is fully released between image and video stages and between chains.
    4. Aggregate every chain's masks per z-layer and write to disk.  <-- STUB

Why this is a script and not a notebook
---------------------------------------
In the notebook, steps 3a/3b were written for ONE hardcoded chain
(`subchain = cell_chain[2]`) and could not be wrapped in a loop over all the
chains that make up a neuron. Here each stage is a function, `run_chain()`
does image+video for a single chain, and `main()` loops over every chain.

Two pieces are intentionally left as STUBS (see the big banners below):
    * refine_prompts()      -- the interactive refinement / manual point-and-click
                               that lived in the notebook (PromptRefiner /
                               PointClicker). Intended to become a PyQt UI.
    * aggregate_and_save()  -- step 7, never implemented in the notebook either.

This script targets a Windows / CUDA setup (paths are Windows raw strings) and
assumes the `sam2_utils` package from the same repo is importable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage.measure import label as cc_label, regionprops

# Project package (same repo as the notebook).
from sam2_utils import setup, diagnostics, alignment, config


# ======================================================================
# CONFIG  (these were the notebook's "top-level knobs")
# ======================================================================

# ---- what to segment -------------------------------------------------
TARGET_CELL_NAME = "AVAL"          # cell to segment
CHAIN_INDICES: list[int] | None = None
#   None  -> run every chain belonging to TARGET_CELL_NAME (the real goal).
#   [2]   -> pilot on a single chain, reproducing the notebook's cell_chain[2].

# ---- SAM2 / resolution ----------------------------------------------
SCALE = 8                          # downsample factor for SAM2 (1=full, 8=~native input)
MODEL_SIZE = "large"               # tiny / small / base_plus / large
SAVE_DOWNSCALE = 4                 # save masks at full-res / N  (used by the save stub)

# ---- negative-prompt filtering --------------------------------------
K_MAX_NEG = 5                      # max negative points per object
NEG_RADIUS = 150                   # exclusion radius (SAM2-res px).
#   NOTE: currently UNUSED -- the notebook filtered negatives by COUNT
#   (K_MAX_NEG) only. Kept as a knob for a future radius-based filter.

# ---- box derivation --------------------------------------------------
MARGIN = 10                        # px of slack around the anchor mask box (SCALE space)

# ---- paths -----------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
DATA_DIR   = SCRIPT_DIR / "data" 
AGG_CSV = DATA_DIR / "aggregate_data_pv.csv"
CHAINS_JSON = DATA_DIR / "chains.json"
ROOTS_JSON = DATA_DIR / "roots.json"

OUT_DIR = Path(r"E:\ZhenLab\Data\output_masks\test2_single")   # final masks (save stub)
# One shared JPEG cache for ALL chains.  Frames are named by file_z so they
# are stable across runs and across chains (e.g. "z1300.jpg").  Per-chain
# symlink views (0-indexed, contiguous) are written under CHAIN_VIEWS_ROOT
# so SAM2's init_state always receives the format it expects.
SHARED_FRAMES_DIR = Path(r"E:\ZhenLab\Data\temp\frames_s{scale}")  # filled in at runtime
CHAIN_VIEWS_ROOT  = Path(r"E:\ZhenLab\Data\temp\chain_views")       # symlink sub-folders

# ---- debug helpers ---------------------------------------------------
SAVE_ANCHOR_OVERLAY = True         # write a PNG of each anchor mask for sanity-checking
                                   # (replaces the notebook's inline matplotlib overlay)

# ---- failure detection -----------------------------------------------
# All thresholds operate in SCALE space on boolean masks.
#
# AREA_RATIO_MAX  -- flag if mask area grows by more than this factor vs prev frame.
#                    3.0 = a sudden 3x explosion is suspicious.
# AREA_RATIO_MIN  -- flag if mask shrinks to below this fraction vs prev frame.
#                    0.25 = mask lost >75% of its area in one step.
# IOU_MIN         -- flag if IoU between consecutive frames drops below this.
#                    0.1 = almost no overlap, mask has drifted or vanished.
# ANNOTATION_CONTAINMENT -- if True, flag any frame where the CATMAID node
#                    for that z is NOT inside the predicted mask.  Requires
#                    a node at every z (guaranteed for this dataset).
# MAX_REANCHOR_DEPTH -- how many times a single propagation pass can be
#                    recursively re-anchored before the remainder is flagged
#                    for manual review and skipped.
AREA_RATIO_MAX       = 3.0
AREA_RATIO_MIN       = 0.25
IOU_MIN              = 0.10
ANNOTATION_CONTAINMENT = True
MAX_REANCHOR_DEPTH   = 3


# ======================================================================
# SCALE / coordinate helpers
# ======================================================================

def parse_file_z(p: Path) -> int:
    """'.../1301____z1300.0.tif' -> 1300 (the file-z token)."""
    token = p.stem.split("z")[-1]
    return int(float(token))


def downscale_image(img: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return img.copy()
    return cv2.resize(img, None, fx=1 / scale, fy=1 / scale,
                      interpolation=cv2.INTER_AREA)


def downscale_points(xy_full: np.ndarray, scale: int) -> np.ndarray:
    return np.asarray(xy_full, dtype=float) / scale


def upscale_mask(mask: np.ndarray, target_hw: tuple[int, int], scale: int) -> np.ndarray:
    """Nearest-neighbour upscale preserves binary edges."""
    if scale == 1:
        return mask.astype(np.uint8)
    h, w = target_hw
    return cv2.resize(mask.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)


# ======================================================================
# DATA LOADING  (replaces the notebook's %store / live CATMAID pull)
# ======================================================================

def load_data() -> tuple[pd.DataFrame, list[dict], object]:
    """Load the annotations DataFrame and chain structures from disk.

    Replaces the notebook cells:
        %store -r aggregate_data_pv
        %store -r chains
        %store -r roots
    with the CSV / JSON files you exported:
        aggregate_data_pv.to_csv(.../aggregate_data_pv.csv, index=False)
        json.dump(chains, .../chains.json)
        json.dump(roots,  .../roots.json)
    """
    df = pd.read_csv(AGG_CSV)

    # node_id is used as an exact-match key all over the pipeline; make sure it
    # didn't come back from CSV as a float (e.g. 25449393.0).
    if "node_id" in df.columns:
        try:
            df["node_id"] = df["node_id"].astype("int64")
        except (ValueError, TypeError):
            pass

    # The notebook computed x_tif / y_tif in a separate cell. If the CSV was
    # exported before that cell ran, compute them now so the rest of the
    # pipeline doesn't care which order you saved in.
    if "x_tif" not in df.columns or "y_tif" not in df.columns:
        xy_tif = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
        df["x_tif"] = xy_tif[:, 0]
        df["y_tif"] = xy_tif[:, 1]

    with open(CHAINS_JSON) as f:
        chains = json.load(f)
    with open(ROOTS_JSON) as f:
        roots = json.load(f)   # not used by the pipeline yet; loaded for parity.

    return df, chains, roots


def get_cell_chains(chains: list[dict], target_cell_name: str) -> list[dict]:
    """All chains belonging to one cell (notebook's cell_chain list)."""
    return [c for c in chains if c["cell_name"] == target_cell_name]


# ======================================================================
# STAGE 3a -- IMAGE MODE: anchor mask + bounding box
# ======================================================================

def pick_anchor_node(subchain: dict, df: pd.DataFrame) -> tuple[int, int]:
    """Mid-section node of the chain and its CATMAID z."""
    mid = len(subchain["nodes"]) // 2
    midnode = subchain["nodes"][mid]
    target_z = int(df.loc[df["node_id"] == str(midnode), "z"].item())
    return midnode, target_z


def locate_anchor_tif(target_z: int) -> tuple[Path, int]:
    """Find the single tif whose file-z matches this CATMAID z.

    file_z = CATMAID_z - FILE_Z_OFFSET  (offset is -7, so file_z = CATMAID_z + 7).
    """
    target_file_z = target_z - config.FILE_Z_OFFSET
    tif_files = sorted(config.WORM_PATH.glob("*.tif"))
    matches = [f for f in tif_files if parse_file_z(f) == target_file_z]
    assert len(matches) == 1, (
        f"Expected 1 tif for file_z={target_file_z}, got {len(matches)}: {matches}"
    )
    return matches[0], target_file_z


def load_anchor_image(tif_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Read the anchor tif, return the SCALE-downscaled image + full-res (H, W).

    We drop the full-res array immediately; only (H_full, W_full) is needed
    later (for upscaling in the save step). Saves ~240 MB peak RAM per the
    notebook's memory notes.
    """
    image_full = cv2.imread(str(tif_path))
    image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
    h_full, w_full = image_full.shape[:2]

    image_sam = downscale_image(image_full, SCALE)
    del image_full
    return image_sam, (h_full, w_full)


def build_prompts(df: pd.DataFrame, midnode: int, target_z: int):
    """CATMAID-derived prompts in SAM2 (SCALE-downscaled) space.

    Returns (list_nodes, list_labels, anchor_xy) where:
        list_nodes  -- [[x, y], ...] with index 0 = the positive midnode
        list_labels -- [1, 0, 0, ...]
        anchor_xy   -- np.float32 [[x, y]] of the positive point (== list_nodes[0])

    Faithful port of the notebook's "Prompt Construction" cell: one positive at
    the midnode, then up to K_MAX_NEG negatives chosen as the nearest other
    nodes on the same z-section.
    """
    list_nodes: list[list[float]] = []
    list_labels: list[int] = []

    cell_node = df.loc[df["node_id"].astype(str) == str(midnode)]
    print(len(df))
    

    # --- positive: the midnode itself ---
    tif_midnode = (cell_node[["x_tif", "y_tif"]] / SCALE).reset_index(drop=True)
    list_nodes.append([tif_midnode["x_tif"].item(), tif_midnode["y_tif"].item()])
    list_labels.append(1)

    # --- negatives: nearest neighbours on the same z-section ---
    cell_x = cell_node["x"].item()
    cell_y = cell_node["y"].item()

    z_points = df[df["z"] == target_z].copy()
    z_points["x"] = pd.to_numeric(z_points["x"], errors="coerce")
    z_points["y"] = pd.to_numeric(z_points["y"], errors="coerce")
    z_points["distance"] = np.sqrt(
        (z_points["x"] - cell_x) ** 2 + (z_points["y"] - cell_y) ** 2
    )
    z_points = z_points.sort_values(by="distance").reset_index(drop=True)
    if len(z_points) and z_points.iloc[0]["distance"] == 0:
        z_points = z_points.drop(0).reset_index(drop=True)  # drop self

    negnodes = (z_points[["x_tif", "y_tif"]] / SCALE).reset_index(drop=True)
    for i in range(min(len(z_points), K_MAX_NEG)):
        list_nodes.append([negnodes.iloc[i]["x_tif"].item(),
                           negnodes.iloc[i]["y_tif"].item()])
        list_labels.append(0)

    anchor_xy = np.array([list_nodes[0]], dtype=np.float32)
    return list_nodes, list_labels, anchor_xy


def predict_image(image_predictor, image_sam, list_nodes, list_labels):
    """Run SAM2 image mode for the anchor frame."""
    with torch.inference_mode():
        image_predictor.set_image(image_sam)
        masks, scores, logits = image_predictor.predict(
            point_coords=np.array(list_nodes),
            point_labels=np.array(list_labels),
            multimask_output=False,
        )
    return masks, scores, logits


def mask_to_box(mask: np.ndarray, hw_sam: tuple[int, int]) -> np.ndarray:
    """Anchor mask -> xyxy box (SCALE space), liver-recipe style.

    Keep the largest connected component (suppresses stray membrane fragments),
    then take its extremal box plus a margin.
    """
    h_sam, w_sam = hw_sam
    m = mask.astype(bool)
    assert m.any(), "empty anchor mask -- re-prompt or flag for human"

    lbl = cc_label(m, connectivity=2)
    m = lbl == (1 + int(np.argmax([r.area for r in regionprops(lbl)])))

    ys, xs = np.where(m)
    x0 = max(int(xs.min()) - MARGIN, 0)
    y0 = max(int(ys.min()) - MARGIN, 0)
    x1 = min(int(xs.max()) + MARGIN, w_sam - 1)
    y1 = min(int(ys.max()) + MARGIN, h_sam - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def save_anchor_overlay(image_sam, mask, nodes, labels, out_path: Path):
    """Lightweight cv2 overlay (no matplotlib) of the anchor mask + prompts.

    Replaces the notebook's inline `viz.show_mask` figure, which doesn't exist
    in a headless run. Purely a debugging aid.
    """
    vis = cv2.cvtColor(image_sam, cv2.COLOR_RGB2BGR).copy()
    overlay = vis.copy()
    overlay[mask.astype(bool)] = (255, 144, 30)          # dodger blue (BGR)
    vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)
    for (x, y), lab in zip(nodes, labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)  # green +, red -
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


# ======================================================================
# ############################  STUB  ##################################
# INTERACTIVE PROMPT REFINEMENT  ->  intended to become a PyQt UI
# ######################################################################
def refine_prompts(image_predictor, image_sam, masks, scores, logits,
                   list_nodes, list_labels, *, target_z, obj_id):
    """STUB. Currently a no-op: returns the automatic CATMAID-derived result
    unchanged so the pipeline runs fully headless.

    ------------------------------------------------------------------
    INTENT (for whoever picks this up next)
    ------------------------------------------------------------------
    In the notebook this is where a human looked at the anchor mask and fixed
    it before the box was derived. Three notebook artifacts did this, all of
    which we are deliberately NOT porting as-is:

        * PromptRefiner (matplotlib, was "currently not working")
        * PointClicker  (matplotlib click-to-read coordinates)
        * a cell of HARD-CODED clicks, e.g. list_nodes.append([820, 428]),
          which only made sense for one image in one session.

    The plan is to replace all of that with a PyQt UI. A working reference
    implementation already exists in the repo:  `Worm_Segmentation_Script.py`
    (the lab's multi-worm tracker). Lift these patterns from it:

        * TrackerWorker(QThread)        -- run SAM2 off the GUI thread; emit
                                           masks back via pyqtSignal so the UI
                                           stays responsive. Here we'd run IMAGE
                                           mode on the anchor frame instead of
                                           video propagation.
        * eventFilter() click mapping   -- left-click = positive point,
                                           right-click = negative, with the
                                           label<->display coordinate transform
                                           (account for QLabel letterboxing).
                                           Clicks live in SAM2 / SCALE space,
                                           same as list_nodes here.
        * per-object sidebar (+/- btns) -- not strictly needed (one object per
                                           chain) but the +/- "prompt mode"
                                           toggle is exactly what we want.
        * re-predict on every edit, feeding prior low-res `logits` back as
          `mask_input` for fast convergence (PromptRefiner tried to do this).

    Desired signature once built (synchronous wrapper around the Qt app, or an
    async callback that blocks the per-chain loop until the user clicks "done"):

        refined_nodes, refined_labels, refined_masks = refine_prompts(...)

    The contract this stub must keep:
        IN : list_nodes (SCALE space, index 0 positive), list_labels, masks
        OUT: same shapes, after human edits. mask_to_box() runs on OUT[0].

    Until then: automatic-only. If the anchor mask is bad, mask_to_box() will
    assert on an empty mask and the chain will be skipped (see run_chain).
    ------------------------------------------------------------------
    """
    print(f"  [STUB] refine_prompts: skipped (automatic prompts only) "
          f"for obj_id={obj_id}, z={target_z}")
    return list_nodes, list_labels, masks


# ======================================================================
# STAGE 3b -- VIDEO MODE: write frames, anchor, propagate
# ======================================================================

def ensure_shared_frames(subset_tifs: list[Path], shared_dir: Path) -> None:
    """Write any not-yet-cached frames into the shared JPEG store.

    Frames are named by file_z (e.g. ``z1300.jpg``) so they are stable
    across runs and shared by every chain that overlaps the same z range.
    Only frames absent from ``shared_dir`` are written; already-cached
    frames are skipped entirely.

    Args:
        subset_tifs: Ordered list of .tif paths for the current chain's z range.
        shared_dir:  Root cache folder (one per SCALE value).
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in subset_tifs
               if not (shared_dir / f"z{parse_file_z(p)}.jpg").exists()]
    if not missing:
        print(f"  all {len(subset_tifs)} frames already cached -- skipping writes")
        return
    print(f"  writing {len(missing)} new frame(s) "
          f"({len(subset_tifs) - len(missing)} already cached)")
    for tif_path in tqdm(missing, desc="  caching frames"):
        img = cv2.imread(str(tif_path))          # BGR is fine for grayscale EM
        img = downscale_image(img, SCALE)        # must match image-mode coord space
        dest = shared_dir / f"z{parse_file_z(tif_path)}.jpg"
        cv2.imwrite(str(dest), img)


def build_chain_view(subset_tifs: list[Path], shared_dir: Path,
                     target_file_z: int, chain_idx: int) -> tuple[str, int]:
    """Create a 0-indexed symlink view of the cached frames for one chain.

    SAM2's ``init_state`` requires a folder of JPEGs named ``00000.jpg``,
    ``00001.jpg``, … in a contiguous sequence.  Rather than re-writing
    (or copying) the shared JPEGs, we create a per-chain subfolder of
    symlinks that map ``{i:05d}.jpg -> ../z{file_z}.jpg``.  On Windows,
    symlinks to files require either Developer Mode or admin rights; if
    symlink creation fails the function falls back to hard-links, then to
    plain copies (last resort).

    Args:
        subset_tifs:    Ordered list of .tif paths defining the chain's z range.
        shared_dir:     Shared frame cache (populated by ensure_shared_frames).
        target_file_z:  file_z of the anchor frame (to find target_frame_idx).
        chain_idx:      Used only to name the view subfolder.

    Returns:
        (view_dir_str, target_frame_idx)
    """
    view_dir = CHAIN_VIEWS_ROOT / f"chain{chain_idx}_s{SCALE}"
    view_dir.mkdir(parents=True, exist_ok=True)

    target_frame_idx = None
    for i, tif_path in enumerate(subset_tifs):
        fz = parse_file_z(tif_path)
        if fz == target_file_z:
            target_frame_idx = i
        link_path = view_dir / f"{i:05d}.jpg"
        if link_path.exists() or link_path.is_symlink():
            continue                                  # already set up
        src = shared_dir / f"z{fz}.jpg"
        try:
            link_path.symlink_to(src)
        except OSError:
            try:
                os.link(src, link_path)               # hard-link fallback
            except OSError:
                import shutil
                shutil.copy2(src, link_path)          # copy as last resort

    assert target_frame_idx is not None, (
        f"anchor file_z={target_file_z} not found in chain {chain_idx}'s z range"
    )
    return str(view_dir), target_frame_idx


def write_video_frames(subchain: dict, df: pd.DataFrame, target_file_z: int,
                       chain_idx: int) -> tuple[str, list[Path], int]:
    """Prepare the per-chain JPEG view that SAM2 video mode consumes.

    Steps:
      1. Determine the chain's z range and collect the matching .tif paths.
      2. Write any not-yet-cached frames to the shared frame store
         (``SHARED_FRAMES_DIR``), keyed by file_z.  Already-cached frames
         are skipped, so a frame written by a previous chain or run is reused
         at zero cost.
      3. Build a 0-indexed symlink view in ``CHAIN_VIEWS_ROOT`` so SAM2
         receives the contiguous ``00000.jpg`` … sequence it expects.

    Returns:
        (view_dir_str, subset_tifs, target_frame_idx)
    """
    # z-extent over ALL chain nodes (a neurite can be non-monotonic in z, so
    # nodes[0] / nodes[-1] are not reliably the z extremes).
    chain_z = [
        int(df.loc[df["node_id"].astype(str) == str(n), "z"].item())
        for n in subchain["nodes"]
    ]
    start_z, end_z = min(chain_z), max(chain_z)

    start_file_z = start_z - config.FILE_Z_OFFSET
    end_file_z   = end_z   - config.FILE_Z_OFFSET

    all_tifs = sorted(config.WORM_PATH.glob("*.tif"), key=parse_file_z)
    subset_tifs = [f for f in all_tifs
                   if start_file_z <= parse_file_z(f) <= end_file_z]

    # Resolve the shared cache dir now that SCALE is known.
    shared_dir = Path(str(SHARED_FRAMES_DIR).format(scale=SCALE))

    ensure_shared_frames(subset_tifs, shared_dir)
    view_dir_str, target_frame_idx = build_chain_view(
        subset_tifs, shared_dir, target_file_z, chain_idx
    )
    return view_dir_str, subset_tifs, target_frame_idx


# ======================================================================
# FAILURE DETECTION
# ======================================================================

@dataclass
class FailureEvent:
    """One detected anomaly in a propagation pass.

    Attributes:
        frame_idx:    0-based index into the chain's subset_tifs / view dir.
        catmaid_z:    CATMAID z of that frame (for re-anchoring via CATMAID node).
        direction:    Which pass produced this failure.
        signals:      Which detection signals fired (subset of
                      {"area_ratio", "iou", "containment"}).
        area_ratio:   Observed area[t] / area[t-1]  (None for frame 0 or empty prev).
        iou:          Observed IoU with previous frame  (None for frame 0 or empty prev).
        node_inside:  Whether the CATMAID node for this z was inside the mask
                      (None if ANNOTATION_CONTAINMENT is False).
        prev_area:    Mask pixel count at t-1  (0 if no previous frame).
        curr_area:    Mask pixel count at t.
    """
    frame_idx:   int
    catmaid_z:   int
    direction:   Literal["forward", "backward"]
    signals:     list[str]
    area_ratio:  float | None
    iou:         float | None
    node_inside: bool | None
    prev_area:   int
    curr_area:   int


def detect_failures(
    segments: dict[int, np.ndarray],
    direction: Literal["forward", "backward"],
    frame_to_catmaid_z: dict[int, int],
    df: pd.DataFrame,
    obj_id: int,
) -> list[FailureEvent]:
    """Scan a single directional propagation result for anomalous frames.

    Args:
        segments:           {frame_idx: bool mask (H, W)} — one direction only.
        direction:          "forward" or "backward", used only for labelling.
        frame_to_catmaid_z: Maps frame_idx -> CATMAID z for this chain.
        df:                 Full annotations DataFrame (for containment check).
        obj_id:             SAM2 object id — used only for log messages.

    Returns:
        List of FailureEvent, one per flagged frame, in frame_idx order.
        Empty list means the pass looks clean.
    """
    failures: list[FailureEvent] = []

    # Sort frames in temporal order for the direction so that "previous frame"
    # is always the frame immediately before in propagation order.
    ordered = sorted(segments.keys(), reverse=(direction == "backward"))

    prev_mask: np.ndarray | None = None
    prev_area: int = 0

    for frame_idx in ordered:
        mask = segments[frame_idx].astype(bool)
        curr_area = int(mask.sum())
        catmaid_z = frame_to_catmaid_z[frame_idx]
        fired: list[str] = []
        area_ratio: float | None = None
        iou_val:    float | None = None
        node_inside: bool | None = None

        # ---- 1. Area ratio (skip if prev was empty to avoid div-by-zero) ----
        if prev_mask is not None and prev_area > 0:
            area_ratio = curr_area / prev_area
            if area_ratio > AREA_RATIO_MAX or area_ratio < AREA_RATIO_MIN:
                fired.append("area_ratio")

        # ---- 2. Temporal IoU ----
        if prev_mask is not None:
            intersection = int((mask & prev_mask).sum())
            union        = int((mask | prev_mask).sum())
            iou_val = intersection / union if union > 0 else 0.0
            if iou_val < IOU_MIN:
                fired.append("iou")

        # ---- 3. CATMAID annotation containment ----
        if ANNOTATION_CONTAINMENT:
            # Look up the node at this CATMAID z for the target cell.
            node_rows = df[df["z"] == catmaid_z]
            if len(node_rows) == 0:
                # Should not happen given dense annotation guarantee, but guard anyway.
                print(f"  [WARN] detect_failures: no node found at catmaid_z={catmaid_z} "
                      f"(frame {frame_idx}, {direction}) -- containment skipped")
                node_inside = None
            else:
                # x_tif / y_tif are in full-res pixel space; divide by SCALE for mask space.
                # Use the first row if somehow multiple nodes share a z (shouldn't happen).
                row = node_rows.iloc[0]
                nx = int(row["x_tif"] / SCALE)
                ny = int(row["y_tif"] / SCALE)
                h, w = mask.shape
                if 0 <= ny < h and 0 <= nx < w:
                    node_inside = bool(mask[ny, nx])
                else:
                    # Node coordinate maps outside the downscaled frame — data issue.
                    print(f"  [WARN] detect_failures: node ({nx},{ny}) out of bounds "
                          f"({w}x{h}) at catmaid_z={catmaid_z} (frame {frame_idx}, "
                          f"{direction}) -- containment skipped")
                    node_inside = None
                if node_inside is False:
                    fired.append("containment")

        # ---- emit failure if any signal fired ----
        if fired:
            ev = FailureEvent(
                frame_idx=frame_idx,
                catmaid_z=catmaid_z,
                direction=direction,
                signals=fired,
                area_ratio=area_ratio,
                iou=iou_val,
                node_inside=node_inside,
                prev_area=prev_area,
                curr_area=curr_area,
            )
            failures.append(ev)
            # Detailed log so failures are easy to read in the console / log file.
            sig_str = ", ".join(fired)
            ar_str  = f"{area_ratio:.3f}" if area_ratio is not None else "n/a"
            iou_str = f"{iou_val:.3f}"   if iou_val    is not None else "n/a"
            con_str = str(node_inside)   if node_inside is not None else "n/a"
            print(
                f"  [FAIL] obj={obj_id} frame={frame_idx} z={catmaid_z} "
                f"dir={direction} | signals=[{sig_str}] | "
                f"area_ratio={ar_str} iou={iou_str} containment={con_str} | "
                f"prev_area={prev_area} curr_area={curr_area}"
            )
            
            nx = int(row["x_tif"] / SCALE)
            ny = int(row["y_tif"] / SCALE)
            print(f"  [DEBUG] containment check: node_id={row['node_id']} "
                f"x_tif={row['x_tif']:.1f} y_tif={row['y_tif']:.1f} "
                f"-> nx={nx} ny={ny} (SCALE={SCALE})")

        prev_mask = mask
        prev_area = curr_area

    if not failures:
        print(f"  [OK] {direction} pass: {len(ordered)} frames, no failures detected "
              f"(obj={obj_id})")
    else:
        print(f"  [SUMMARY] {direction} pass: {len(failures)} failure(s) across "
              f"{len(ordered)} frames (obj={obj_id})")

    return failures


def first_failure_frame(failures: list[FailureEvent]) -> int | None:
    """Return the frame_idx of the first failure in propagation order.

    For a forward pass, 'first' = smallest frame_idx.
    For a backward pass, 'first' = largest frame_idx  (i.e. the frame
    encountered first when propagating backward from the anchor).

    If failures is empty, returns None.
    """
    if not failures:
        return None
    if failures[0].direction == "forward":
        return min(f.frame_idx for f in failures)
    else:
        return max(f.frame_idx for f in failures)


def def_frame_to_catmaid_z(subset_tifs: list[Path]) -> dict[int, int]:
    """Build a frame_idx -> CATMAID_z lookup for a chain's tif list."""
    return {
        i: parse_file_z(tif) + config.FILE_Z_OFFSET
        for i, tif in enumerate(subset_tifs)
    }


def def_catmaid_z_to_node(df: pd.DataFrame, catmaid_z: int) -> int | None:
    """Return the node_id at this CATMAID z, or None if not found."""
    rows = df[df["z"] == catmaid_z]
    if len(rows) == 0:
        return None
    return str(rows.iloc[0]["node_id"])


def def_frame_to_catmaid_z_range(
    subset_tifs: list[Path],
    start_frame: int,
    end_frame: int,
) -> dict[int, int]:
    """Subset of def_frame_to_catmaid_z for a contiguous frame range [start, end]."""
    return {
        i: parse_file_z(subset_tifs[i]) + config.FILE_Z_OFFSET
        for i in range(start_frame, end_frame + 1)
    }


def propagate(video_predictor, frames_dir: str, anchor_box, anchor_xy,
              target_frame_idx: int, obj_id: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Anchor the box+point on the central frame and propagate both directions.

    Returns:
        (fwd_segments, bwd_segments) where each is
        {frame_idx: bool mask (H, W) in SCALE space}.

        The two dicts are kept SEPARATE so that failure detection can treat
        each pass independently — a bad backward frame should not pollute
        the forward result and vice versa.

        The anchor frame itself appears in BOTH dicts (SAM2 emits it in
        each pass).  When merging, forward takes precedence for the anchor.
    """
    # init_state needs a STRING path; offload_video_to_cpu keeps VRAM bounded
    # for long (~340-frame) chains.
    inference_state = video_predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
        # offload_state_to_cpu=True,   # enable if OOM on very long chains
    )

    video_predictor.reset_state(inference_state)     # per-object scoping
    video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=target_frame_idx,
        obj_id=obj_id,
        box=anchor_box,
        points=anchor_xy,            # box + point: more robust on thin neurites
        labels=np.array([1], np.int32),
    )

    fwd_segments: dict[int, np.ndarray] = {}
    bwd_segments: dict[int, np.ndarray] = {}

    def _collect(store: dict, frame_idx, obj_ids, mask_logits):
        # Store only the mask for our object; squeeze the leading dim SAM2 adds.
        for i, oid in enumerate(obj_ids):
            if oid == obj_id:
                mask = (mask_logits[i] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]          # (1, H, W) -> (H, W)
                store[frame_idx] = mask.astype(bool)

    print(f"  [propagate] forward pass  (anchor frame {target_frame_idx})")
    for f, ids, lg in video_predictor.propagate_in_video(inference_state):
        _collect(fwd_segments, f, ids, lg)

    print(f"  [propagate] backward pass (anchor frame {target_frame_idx})")
    for f, ids, lg in video_predictor.propagate_in_video(inference_state, reverse=True):
        _collect(bwd_segments, f, ids, lg)

    print(f"  [propagate] fwd={len(fwd_segments)} frames, "
          f"bwd={len(bwd_segments)} frames")

    # If a pass emitted nothing (anchor is at the boundary of the clip),
    # seed the missing anchor from the other pass so downstream asserts hold.
    if target_frame_idx not in fwd_segments and target_frame_idx in bwd_segments:
        fwd_segments[target_frame_idx] = bwd_segments[target_frame_idx]
    if target_frame_idx not in bwd_segments and target_frame_idx in fwd_segments:
        bwd_segments[target_frame_idx] = fwd_segments[target_frame_idx]

    del inference_state
    return fwd_segments, bwd_segments


def propagate_one_direction(
    video_predictor,
    frames_dir: str,
    anchor_box: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_frame_idx: int,
    obj_id: int,
    direction: Literal["forward", "backward"],
) -> dict[int, np.ndarray]:
    """Like propagate(), but runs only one direction.

    Used by run_segment() when re-anchoring at a failure point: the clean
    portion of the original pass is already kept, so only the affected
    direction needs to be re-run from the new anchor.

    Returns {frame_idx: bool mask (H, W)}.
    """
    inference_state = video_predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
    )
    video_predictor.reset_state(inference_state)
    video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=anchor_frame_idx,
        obj_id=obj_id,
        box=anchor_box,
        points=anchor_xy,
        labels=np.array([1], np.int32),
    )

    segments: dict[int, np.ndarray] = {}

    def _collect(frame_idx, obj_ids, mask_logits):
        for i, oid in enumerate(obj_ids):
            if oid == obj_id:
                mask = (mask_logits[i] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                segments[frame_idx] = mask.astype(bool)

    reverse = (direction == "backward")
    print(f"  [propagate_one_direction] {direction} from frame {anchor_frame_idx}")
    for f, ids, lg in video_predictor.propagate_in_video(inference_state, reverse=reverse):
        _collect(f, ids, lg)

    print(f"  [propagate_one_direction] {direction}: {len(segments)} frames collected")
    assert anchor_frame_idx in segments, (
        f"anchor frame {anchor_frame_idx} missing from {direction} re-anchor segments"
    )
    del inference_state
    return segments


def run_segment(
    *,
    video_predictor,
    frames_dir: str,
    anchor_frame_idx: int,
    direction: Literal["forward", "backward"],
    subset_tifs: list[Path],
    df: pd.DataFrame,
    obj_id: int,
    chain_label: str,
    depth: int,
) -> dict[int, np.ndarray]:
    """Recursively propagate one direction from an anchor, re-anchoring on failure.

    This is the core of the error-recovery loop.  On the first call it is
    invoked with the initial anchor; on recursive calls it is invoked at the
    frame where the previous pass first failed.

    Args:
        video_predictor:  Loaded SAM2 video predictor (caller owns lifetime).
        frames_dir:       Path string for init_state.
        anchor_frame_idx: 0-based frame index to anchor from.
        direction:        Which way to propagate from the anchor.
        subset_tifs:      Ordered tif list for this chain (frame_idx -> file_z).
        df:               Full annotations DataFrame.
        obj_id:           SAM2 object id.
        chain_label:      Human-readable label for log messages (e.g. "chain2[fwd]").
        depth:            Current recursion depth (0 = first call).

    Returns:
        {frame_idx: bool mask} for all frames in the propagated range that
        passed detection, up to (but NOT including) the first failing frame.
        If depth > MAX_REANCHOR_DEPTH the segment is abandoned and {} is
        returned for all frames after the anchor.
    """
    indent = "  " * (depth + 2)   # visual nesting in logs

    if depth > MAX_REANCHOR_DEPTH:
        print(
            f"{indent}[ABORT] {chain_label} depth={depth} exceeds MAX_REANCHOR_DEPTH="
            f"{MAX_REANCHOR_DEPTH} at frame {anchor_frame_idx} ({direction}) -- "
            "remaining frames flagged for manual review"
        )
        return {}

    print(f"{indent}[run_segment] {chain_label} depth={depth} "
          f"anchor_frame={anchor_frame_idx} direction={direction}")

    # ---- image mode: get anchor mask + box at this frame ----
    anchor_tif  = subset_tifs[anchor_frame_idx]
    anchor_catmaid_z = parse_file_z(anchor_tif) + config.FILE_Z_OFFSET
    anchor_node = def_catmaid_z_to_node(df, anchor_catmaid_z)
    assert anchor_node is not None, (
        f"{chain_label}: no CATMAID node at z={anchor_catmaid_z} "
        f"(frame {anchor_frame_idx}) -- annotation gap violates dense-annotation invariant"
    )

    image_predictor, _ = setup.build_predictor(
        size=MODEL_SIZE, kind="image", checkpoint_dir=CHECKPOINT_DIR
    )
    diagnostics.snapshot(f"{chain_label} depth={depth} image model load")

    image_sam, _ = load_anchor_image(anchor_tif)
    hw_sam = image_sam.shape[:2]
    list_nodes, list_labels, anchor_xy = build_prompts(df, anchor_node, anchor_catmaid_z)
    masks, scores, logits = predict_image(
        image_predictor, image_sam, list_nodes, list_labels
    )

    # refine_prompts is a no-op stub until the PyQt UI is built.
    list_nodes, list_labels, masks = refine_prompts(
        image_predictor, image_sam, masks, scores, logits,
        list_nodes, list_labels,
        target_z=anchor_catmaid_z, obj_id=obj_id,
    )

    try:
        anchor_box = mask_to_box(masks[0], hw_sam)
    except AssertionError as e:
        print(f"{indent}[SKIP] empty anchor mask at frame {anchor_frame_idx} "
              f"z={anchor_catmaid_z}: {e}")
        image_predictor.reset_predictor()
        del image_predictor
        diagnostics.cleanup_vram()
        return {}

    print(f"{indent}anchor_box={anchor_box}  z={anchor_catmaid_z}")

    if SAVE_ANCHOR_OVERLAY:
        save_anchor_overlay(
            image_sam, masks[0], list_nodes, list_labels,
            OUT_DIR / "anchor_overlays" /
            f"{TARGET_CELL_NAME}_{chain_label}_depth{depth}_z{anchor_catmaid_z}.png",
        )

    image_predictor.reset_predictor()
    del image_predictor
    diagnostics.cleanup_vram()

    # ---- propagate one direction ----
    raw_segments = propagate_one_direction(
        video_predictor, frames_dir,
        anchor_box, anchor_xy, anchor_frame_idx,
        obj_id, direction,
    )

    # ---- detection ----
    frame_to_z = def_frame_to_catmaid_z(subset_tifs)
    failures = detect_failures(
        raw_segments, direction, frame_to_z, df, obj_id
    )

    fail_frame = first_failure_frame(failures)

    if fail_frame is None:
        # Clean pass — return everything.
        print(f"{indent}[CLEAN] {chain_label} depth={depth} {direction}: "
              f"{len(raw_segments)} frames accepted")
        return raw_segments

    # ---- split: keep clean portion, recurse on tail ----
    if direction == "forward":
        # Keep [anchor_frame_idx, fail_frame - 1], re-anchor at fail_frame.
        clean = {k: v for k, v in raw_segments.items() if k < fail_frame}
        print(f"{indent}[SPLIT fwd] keeping frames "
              f"[{anchor_frame_idx}..{fail_frame-1}] ({len(clean)} frames), "
              f"re-anchoring at {fail_frame}")
        tail = run_segment(
            video_predictor=video_predictor,
            frames_dir=frames_dir,
            anchor_frame_idx=fail_frame,
            direction="forward",
            subset_tifs=subset_tifs,
            df=df,
            obj_id=obj_id,
            chain_label=chain_label,
            depth=depth + 1,
        )
    else:
        # Backward: keep [fail_frame + 1, anchor_frame_idx], re-anchor at fail_frame.
        clean = {k: v for k, v in raw_segments.items() if k > fail_frame}
        print(f"{indent}[SPLIT bwd] keeping frames "
              f"[{fail_frame+1}..{anchor_frame_idx}] ({len(clean)} frames), "
              f"re-anchoring at {fail_frame}")
        tail = run_segment(
            video_predictor=video_predictor,
            frames_dir=frames_dir,
            anchor_frame_idx=fail_frame,
            direction="backward",
            subset_tifs=subset_tifs,
            df=df,
            obj_id=obj_id,
            chain_label=chain_label,
            depth=depth + 1,
        )

    return {**clean, **tail}


# ======================================================================
# PER-CHAIN DRIVER  (image + video, VRAM reset each chain)
# ======================================================================

def run_chain(subchain: dict, df: pd.DataFrame, chain_idx: int) -> dict | None:
    """Run the full image->video pipeline for ONE chain, with failure recovery.

    Pipeline:
      1. Image mode  -- anchor mask + box at the mid-chain frame.
      2. Video mode  -- bidirectional propagation (fwd + bwd as separate dicts).
      3. Detection   -- scan each pass independently for anomalous frames.
      4. Recovery    -- for each failing pass, split at the first bad frame and
                        call run_segment() recursively (up to MAX_REANCHOR_DEPTH).
      5. Merge       -- combine clean fwd and bwd segments into one dict keyed
                        by frame_idx; forward takes precedence at the anchor.

    Models are built and torn down so VRAM is fully released between stages.
    """
    obj_id = chain_idx + 1   # one object per MLC; 1-based for SAM2
    chain_label = f"chain{chain_idx}"
    print(f"\n=== chain {chain_idx} (obj_id={obj_id}) ===")

    # ---- anchor selection + prompts (no model needed yet) ----
    midnode, target_z = pick_anchor_node(subchain, df)
    tif_path, target_file_z = locate_anchor_tif(target_z)
    print(f"  anchor: node={midnode}, CATMAID_z={target_z}, tif={tif_path.name}")
    image_sam, full_hw = load_anchor_image(tif_path)
    hw_sam = image_sam.shape[:2]
    list_nodes, list_labels, anchor_xy = build_prompts(df, midnode, target_z)

    # ---- IMAGE MODE ----
    image_predictor, _ = setup.build_predictor(
        size=MODEL_SIZE, kind="image", checkpoint_dir=CHECKPOINT_DIR
    )
    diagnostics.snapshot("after image model load")
    masks, scores, logits = predict_image(
        image_predictor, image_sam, list_nodes, list_labels
    )

    # interactive refinement (STUB -- no-op for now)
    list_nodes, list_labels, masks = refine_prompts(
        image_predictor, image_sam, masks, scores, logits,
        list_nodes, list_labels, target_z=target_z, obj_id=obj_id,
    )

    try:
        anchor_box = mask_to_box(masks[0], hw_sam)
    except AssertionError as e:
        print(f"  !! {e}  -- skipping chain {chain_idx}")
        image_predictor.reset_predictor()
        del image_predictor
        diagnostics.cleanup_vram()
        return None
    print(f"  anchor_box: {anchor_box}")

    if SAVE_ANCHOR_OVERLAY:
        save_anchor_overlay(
            image_sam, masks[0], list_nodes, list_labels,
            OUT_DIR / "anchor_overlays" /
            f"{TARGET_CELL_NAME}_{chain_label}_z{target_z}.png",
        )

    image_predictor.reset_predictor()
    del image_predictor
    diagnostics.cleanup_vram()

    # ---- VIDEO MODE: write/link frames ----
    video_predictor, _ = setup.build_predictor(size=MODEL_SIZE, kind="video")
    diagnostics.snapshot("after video model load")
    frames_dir, subset_tifs, target_frame_idx = write_video_frames(
        subchain, df, target_file_z, chain_idx
    )

    # Build the frame -> CATMAID_z map once; shared by detection and recovery.
    frame_to_z = def_frame_to_catmaid_z(subset_tifs)

    # ---- initial bidirectional propagation ----
    fwd_raw, bwd_raw = propagate(
        video_predictor, frames_dir, anchor_box, anchor_xy,
        target_frame_idx, obj_id,
    )

    # ---- detection: each pass independently ----
    print(f"\n  --- detection: forward ---")
    fwd_failures = detect_failures(fwd_raw, "forward",  frame_to_z, df, obj_id)
    print(f"\n  --- detection: backward ---")
    bwd_failures = detect_failures(bwd_raw, "backward", frame_to_z, df, obj_id)

    # ---- recovery: split + re-anchor if needed ----
    fwd_fail_frame = first_failure_frame(fwd_failures)
    bwd_fail_frame = first_failure_frame(bwd_failures)

    if fwd_fail_frame is None:
        fwd_segments = fwd_raw
    else:
        # Keep the clean prefix; re-anchor at the failure frame forward.
        fwd_clean = {k: v for k, v in fwd_raw.items() if k < fwd_fail_frame}
        print(f"\n  --- recovery: forward re-anchor at frame {fwd_fail_frame} ---")
        fwd_tail = run_segment(
            video_predictor=video_predictor,
            frames_dir=frames_dir,
            anchor_frame_idx=fwd_fail_frame,
            direction="forward",
            subset_tifs=subset_tifs,
            df=df,
            obj_id=obj_id,
            chain_label=f"{chain_label}[fwd]",
            depth=1,
        )
        fwd_segments = {**fwd_clean, **fwd_tail}

    if bwd_fail_frame is None:
        bwd_segments = bwd_raw
    else:
        # Keep the clean suffix; re-anchor at the failure frame backward.
        bwd_clean = {k: v for k, v in bwd_raw.items() if k > bwd_fail_frame}
        print(f"\n  --- recovery: backward re-anchor at frame {bwd_fail_frame} ---")
        bwd_tail = run_segment(
            video_predictor=video_predictor,
            frames_dir=frames_dir,
            anchor_frame_idx=bwd_fail_frame,
            direction="backward",
            subset_tifs=subset_tifs,
            df=df,
            obj_id=obj_id,
            chain_label=f"{chain_label}[bwd]",
            depth=1,
        )
        bwd_segments = {**bwd_clean, **bwd_tail}

    # ---- merge: fwd wins at anchor frame ----
    # bwd_segments covers [0, anchor]; fwd_segments covers [anchor, end].
    # Unioning them gives full coverage; fwd takes precedence at the anchor.
    merged: dict[int, np.ndarray] = {**bwd_segments, **fwd_segments}
    print(f"\n  merged segments: {len(merged)} frames total "
          f"(fwd={len(fwd_segments)}, bwd={len(bwd_segments)})")

    # Sanity: every frame in subset_tifs should have a mask.
    n_tifs = len(subset_tifs)
    missing = [i for i in range(n_tifs) if i not in merged]
    if missing:
        print(f"  [WARN] {len(missing)} frame(s) have no mask after recovery "
              f"(flagged for manual review): frames {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))
    else:
        print(f"  [OK] full coverage: all {n_tifs} frames have a mask")

    del video_predictor
    diagnostics.cleanup_vram()

    return {
        "chain_idx":        chain_idx,
        "obj_id":           obj_id,
        "subchain":         subchain,
        "target_z":         target_z,
        "target_frame_idx": target_frame_idx,
        "subset_tifs":      subset_tifs,      # frame_idx -> tif Path, for z mapping
        "frame_to_z":       frame_to_z,       # frame_idx -> CATMAID_z
        "fwd_segments":     fwd_segments,     # clean forward masks
        "bwd_segments":     bwd_segments,     # clean backward masks
        "video_segments":   merged,           # unified {frame_idx: bool mask}
        "missing_frames":   missing,          # frames that exceeded retry limit
        "full_hw":          full_hw,          # (H_full, W_full) for upscaling
    }


# ======================================================================
# ############################  STUB  ##################################
# STEP 7 -- AGGREGATE MASKS PER LAYER + SAVE TO DISK
# ######################################################################
def aggregate_and_save(chain_results: list[dict]):
    """STUB. Not implemented in the notebook either (step 7 was prose only).

    ------------------------------------------------------------------
    INTENT (for whoever picks this up next)
    ------------------------------------------------------------------
    Each chain in `chain_results` produced `video_segments`:
        {video_frame_idx: {obj_id: bool mask in SCALE space}}
    and a `subset_tifs` list so video_frame_idx -> file_z -> CATMAID_z:
        frame_to_z = {i: parse_file_z(tif) + config.FILE_Z_OFFSET
                      for i, tif in enumerate(subset_tifs)}

    Different chains cover different (possibly overlapping) z ranges, so the
    aggregation key must be CATMAID_z, NOT the per-chain video frame index.

    Target behaviour (from the notebook's cell-0 memory notes):
      * For each CATMAID z that ANY chain touches, OR the union of all chains'
        masks on that layer  -> one combined mask per z (the full neuron on
        every frame).
      * Upscale each combined mask from SCALE space to full-res / SAVE_DOWNSCALE
        using upscale_mask(mask, (H_full // SAVE_DOWNSCALE, W_full // SAVE_DOWNSCALE), ...).
      * Write masks ONE AT A TIME inside the loop (do NOT stack them all into one
        array first -- that was the v2 memory fix to avoid VRAM/RAM blowup).
      * Output filenames keyed by CATMAID z under OUT_DIR.

    Open design question carried over from the notebook (cell 6): how to combine
    overlapping masks from different chains on the same layer -- logical OR is
    the obvious default; revisit if chains for different cells ever share OUT_DIR.
    ------------------------------------------------------------------
    """
    n = len(chain_results)
    total_frames = sum(len(r["video_segments"]) for r in chain_results)
    total_missing = sum(len(r["missing_frames"]) for r in chain_results)
    print(f"\n[STUB] aggregate_and_save: {n} chain(s), "
          f"{total_frames} chain-frames total, {total_missing} frame(s) without masks "
          f"-- NOT written to {OUT_DIR}")
    print("[STUB] implement per-CATMAID-z union + full-res upscale + one-at-a-time save.")


# ======================================================================
# MAIN
# ======================================================================

def main():
    diagnostics.snapshot("startup")
    df, chains, roots = load_data()
    print(f"loaded {len(df)} nodes, {len(chains)} chains")

    cell_chain = get_cell_chains(chains, TARGET_CELL_NAME)
    if not cell_chain:
        raise SystemExit(f"No chains found for cell '{TARGET_CELL_NAME}'")

    indices = list(range(len(cell_chain))) if CHAIN_INDICES is None else CHAIN_INDICES
    print(f"running {len(indices)} of {len(cell_chain)} chains for "
          f"{TARGET_CELL_NAME}")

    chain_results: list[dict] = []
    for chain_idx in indices:
        subchain = cell_chain[chain_idx]
        result = run_chain(subchain, df, chain_idx)
        if result is not None:
            chain_results.append(result)

    aggregate_and_save(chain_results)
    print("\ndone.")


if __name__ == "__main__":
    main()