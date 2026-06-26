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

This module is now a package: the body has been split into concern-focused
submodules (config, state, frames, masks, predict, crop, propagate, qc,
orchestrator). This file re-exports the full public surface so every caller and
test keeps importing `pipeline` unchanged.
"""

from __future__ import annotations

from .config import PipelineConfig
from .crop import (
    chain_crop_window,
    chain_masks_in_sam,
    grow_crop_window,
    mask_union_box_px,
    neuron_crop_window,
    node_crop_window,
    remap_mask_to_window,
    window_from_sam_box,
    _neuron_skeleton_box_tif,
    prepare_chain_crop_frames,
    prepare_video_frames,
    _chain_skeleton_box_tif,
    _prior_queued_z,
)
from .frames import (
    FrameStore,
    TifFrameStore,
    load_frame_sam,
    _downscale_image,
    _ensure_cached_frames,
    _link_frame,
    _parse_file_z,
    _read_tif_window,
)
from .masks import (
    fill_small_holes,
    postprocess_mask,
    remove_small_islands,
    save_masks,
    smooth_edges,
)
from .orchestrator import run_chain
from .predict import (
    anchor_crop_predict,
    box_from_mask,
    build_prompts,
    image_predict,
    neighbor_chains,
    score_anchor,
    select_anchor,
    _largest_cc_frac,
    _negative_points,
    _point_in_mask,
    _positive_point,
    _select_anchor_mask,
)
from .propagate import FrameResult, MultiObjectPropagationSession, PropagationSession, propagate, _attach_iou_hook
from .qc import run_qc
from .state import (
    AnchorScore,
    ChainState,
    Prompts,
    load_state,
    save_state,
    state_from_dict,
    state_to_dict,
    _anchor_score_to_dict,
    _config_from_dict,
    _config_to_dict,
    _prompts_from_dict,
    _prompts_to_dict,
)

__all__ = [
    # classes
    "FrameStore",
    "TifFrameStore",
    "PipelineConfig",
    "Prompts",
    "ChainState",
    "AnchorScore",
    "FrameResult",
    "PropagationSession",
    "MultiObjectPropagationSession",
    # functions
    "run_chain",
    "select_anchor",
    "load_frame_sam",
    "build_prompts",
    "neighbor_chains",
    "image_predict",
    "box_from_mask",
    "score_anchor",
    "anchor_crop_predict",
    "prepare_video_frames",
    "mask_union_box_px",
    "chain_masks_in_sam",
    "chain_crop_window",
    "node_crop_window",
    "grow_crop_window",
    "window_from_sam_box",
    "neuron_crop_window",
    "remap_mask_to_window",
    "prepare_chain_crop_frames",
    "propagate",
    "save_masks",
    "postprocess_mask",
    "remove_small_islands",
    "fill_small_holes",
    "smooth_edges",
    "run_qc",
    "save_state",
    "load_state",
    "state_to_dict",
    "state_from_dict",
]
