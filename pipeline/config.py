"""Run settings: the notebook's top-level knobs, in one dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    image_size: Optional[int] = None   # override SAM2's internal input resolution (px);
                                       # None keeps the checkpoint default (1024). SAM2
                                       # resizes every frame/crop to this size, so raising
                                       # it is the only in-model way to feed more pixels
                                       # (memory ~quadratic; off-distribution for the
                                       # pretrained encoder). Read at predictor-build; the
                                       # actual value is asserted post-build (setup.py).

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
    chain_crop_pad_tif: int = 512      # padding around the skeleton/mask xy-extent, _tif px.
                                       # Generous on purpose: windows sized from the first-pass
                                       # mask (or skeleton) that looked fine often clip the cell
                                       # in practice, so every tier-2 window gets 512/side of
                                       # slack. crop_scale bumps coarser if the wider extent
                                       # would exceed chain_crop_max_px (coverage over resolution).
    chain_crop_scale: int = 2          # target read downscale (1 = full-res)
    chain_crop_max_px: int = 1536      # cap on the crop's longest input edge (bounds VRAM)
    # FLOOR on the crop's _tif extent. A low-motion chain (neurite barely moves in xy)
    # otherwise gets a tiny over-zoomed window where SAM2 loses inter-frame context and
    # the mask collapses to empty (the AIYL chain_02 over-zoom failure). This pads
    # the window out (centred) so the crop always carries enough surrounding context to
    # track. 1024 _tif px -> ~512 px input at crop_scale 2, still ~4x the neurite
    # resolution of the scale-8 full frame.
    chain_crop_min_tif: int = 1024

    # Collapse fallback for chain_crop_from_mask. When the first pass produced masks but
    # they COLLAPSED (no usable foreground to size a window from), size from the skeleton
    # bbox is a guess that can land small and off-centre. Instead drop a fixed
    # chain_crop_collapse_size_tif-square window centred on the anchor NODE, a predictable
    # window regardless of how the (untrustworthy) skeleton wanders. 1024 _tif px -> ~512
    # input px at crop_scale 2, matching chain_crop_min_tif. 0 disables (keep skeleton
    # sizing). Only consulted when masks exist but are empty; a chain with no prior masks
    # at all still sizes from the skeleton.
    chain_crop_collapse_size_tif: int = 1024

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
    # Among the multimask candidates that contain the positive node, prefer the one
    # that contains NO negative node (the nearest same-z neighbours already seeded as
    # negatives in build_prompts). Adapts the 2025 lightweight-SAM2 paper's
    # anchor-containment selection (their Hoechst nucleus centre, our skeleton node) to
    # our bleed problem: a bleeding mask swallows a neighbour's node, so excluding
    # negatives is the anti-bleed pick. Only consulted when multimask_anchor is on;
    # default OFF so it can be measured against plain multimask selection.
    multimask_exclude_neg: bool = False

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
    postprocess_masks: bool = False    # MASTER toggle for the whole post-process stage.
                                       # Off = baseline (raw masks). Flip this to A/B the
                                       # outputs with vs without cleanup.
    postproc_open_px: int = 1
    postproc_close_px: int = 1
    postproc_keep_largest_cc: bool = True
    postproc_fill_holes: bool = True
    # Size-aware cleanup (gated by the master toggle above; each runs only when > 0, so the
    # defaults of 0 leave the baseline postproc unchanged). These are the granular ops:
    #   remove_islands_min_size: drop detached components smaller than N px, KEEPING all
    #     larger ones (unlike keep_largest_cc, so a real second cross-section survives).
    #   fill_small_holes_area: fill interior holes smaller than N px, leaving large cavities.
    #   smooth_radius: morphological close-then-open with a disk of N px to smooth a
    #     frayed/netty boundary. Keep small or thin neurites erode. All in scale-8 _sam px.
    postproc_remove_islands_min_size: int = 0
    postproc_fill_small_holes_area: int = 0
    postproc_smooth_radius: int = 0

    # which per-frame severity enters the human triage queue. A frame is
    # queued when flag_count >= this. 2 = intervene-level (>=2 corroborating signals),
    # the default: single-signal flags are dominated by
    # dilation-sensitive `noskel` noise (flag_rate moved 0.33->0.19 over a 0..10px
    # dilation sweep) while the intervene set is dilation-robust (rate moved <0.005).
    # Set to 1 to restore the legacy "queue every flag" behaviour.
    qc_triage_min_signals: int = 2
