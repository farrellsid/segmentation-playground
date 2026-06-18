"""Thin per-chain driver: run_chain composes the phase functions end-to-end."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .crop import (
    chain_crop_window,
    mask_union_box_px,
    prepare_chain_crop_frames,
    prepare_video_frames,
    _prior_queued_z,
)
from .frames import FrameStore, TifFrameStore, load_frame_sam
from .masks import postprocess_mask, save_masks
from .predict import (
    anchor_crop_predict,
    box_from_mask,
    build_prompts,
    image_predict,
    score_anchor,
    select_anchor,
)
from .propagate import propagate
from .qc import run_qc
from .state import ChainState, Prompts, _anchor_score_to_dict, load_state


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
            # when the prior masks are _sam, a prior tier-2 (state.json has a
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
                select_area_bounds=area_bounds, select_exclude_neg=cfg.multimask_exclude_neg)
        else:
            mask_anchor, state.image_score, _logits = image_predict(
                image_predictor, image_sam, state.prompts,
                multimask=cfg.multimask_anchor, select_contain_radius_px=contain_r,
                select_area_bounds=area_bounds, select_exclude_neg=cfg.multimask_exclude_neg)
            cw, prompts_anchor = None, state.prompts
        image_hw_anchor = mask_anchor.shape[:2]
        if cfg.multimask_anchor:
            print("    multimask auto-select on (3 candidates -> 1)"
                  + (", excluding negatives" if cfg.multimask_exclude_neg else ""))
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
            # tier-2: the WHOLE chain propagates in _pcrop, so the seed stays there, the
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
        # state.image_score / state.anchor_score, else the failing reason is lost (only the
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
