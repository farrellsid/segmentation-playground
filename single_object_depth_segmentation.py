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
from pathlib import Path

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


def propagate(video_predictor, frames_dir: str, anchor_box, anchor_xy,
              target_frame_idx: int, obj_id: int) -> dict:
    """Anchor the box+point on the central frame and propagate both directions.

    Returns video_segments: {frame_idx: {obj_id: bool mask (SCALE space)}}.
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

    video_segments: dict = {}

    def _collect(frame_idx, obj_ids, mask_logits):
        video_segments[frame_idx] = {
            oid: (mask_logits[i] > 0.0).cpu().numpy()
            for i, oid in enumerate(obj_ids)
        }

    for f, ids, lg in video_predictor.propagate_in_video(inference_state):
        _collect(f, ids, lg)
    for f, ids, lg in video_predictor.propagate_in_video(inference_state, reverse=True):
        _collect(f, ids, lg)

    print(f"  propagated {len(video_segments)} frames")
    # TODO: per-frame degradation detection + auto re-prompt
    #   (pred_iou / area-ratio / skeleton-node containment / temporal IoU).
    del inference_state
    return video_segments


# ======================================================================
# PER-CHAIN DRIVER  (image + video, VRAM reset each chain)
# ======================================================================

def run_chain(subchain: dict, df: pd.DataFrame, chain_idx: int) -> dict | None:
    """Run the full image->video pipeline for ONE chain.

    Models are built and torn down inside this function so VRAM is fully
    released between the image and video stages and again before the next
    chain (per the requested image+video-per-chain structure).
    """
    obj_id = chain_idx + 1   # one object per MLC; 1-based for SAM2
    print(f"\n=== chain {chain_idx} (obj_id={obj_id}) ===")

    # ---- anchor selection + prompts (no model needed yet) ----
    midnode, target_z = pick_anchor_node(subchain, df)
    tif_path, target_file_z = locate_anchor_tif(target_z)
    print(f"  anchor: node={midnode}, CATMAID_z={target_z}, tif={tif_path.name}")
    image_sam, full_hw = load_anchor_image(tif_path)
    hw_sam = image_sam.shape[:2]
    list_nodes, list_labels, anchor_xy = build_prompts(df, midnode, target_z)

    # ---- IMAGE MODE ----
    image_predictor, _ = setup.build_predictor(size=MODEL_SIZE, kind="image", checkpoint_dir=CHECKPOINT_DIR)
    diagnostics.snapshot("after image model load")
    masks, scores, logits = predict_image(image_predictor, image_sam,
                                           list_nodes, list_labels)

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
            OUT_DIR / "anchor_overlays" / f"{TARGET_CELL_NAME}_chain{chain_idx}_z{target_z}.png",
        )

    # free the image model before loading the video model
    image_predictor.reset_predictor()
    del image_predictor
    diagnostics.cleanup_vram()

    # ---- VIDEO MODE ----
    video_predictor, _ = setup.build_predictor(size=MODEL_SIZE, kind="video")
    diagnostics.snapshot("after video model load")
    frames_dir, subset_tifs, target_frame_idx = write_video_frames(
        subchain, df, target_file_z, chain_idx
    )
    video_segments = propagate(
        video_predictor, frames_dir, anchor_box, anchor_xy,
        target_frame_idx, obj_id,
    )

    del video_predictor
    diagnostics.cleanup_vram()

    return {
        "chain_idx": chain_idx,
        "obj_id": obj_id,
        "subchain": subchain,
        "target_z": target_z,
        "target_frame_idx": target_frame_idx,
        "subset_tifs": subset_tifs,     # index -> tif Path, for z mapping in save
        "video_segments": video_segments,
        "full_hw": full_hw,             # (H_full, W_full) of the anchor frame
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
    print(f"\n[STUB] aggregate_and_save: {n} chain(s), "
          f"{total_frames} chain-frames total -- NOT written to {OUT_DIR}")
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