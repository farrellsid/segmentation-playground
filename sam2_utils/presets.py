"""presets.py: named run configurations for `batch.py` and `eval.score_batch`.

A preset bundles everything that distinguishes one kind of run (which worm/dataset,
the output + frames roots, the `PipelineConfig` knobs, the tier-2 / gif settings, and a
default neuron set) so a run is just `--preset <name> [--neurons ...]` instead of a long
command. Edit a preset here (or add a new one) rather than spelling out flags each time.

  eval     : SEM-Dauer 1 cross-worm GT: large model, tier-2, the GT paths;
             scored by `eval.score_batch` against the VAST GT (`score_out`).
  original : the target/production worm (CATMAID 336, the "sensory ablated dauer"): the
             params the batch driver used to hardcode (module-level knobs, moved here).

Any CLI flag (`--neurons`, `--model-size`, `--output-root`, `--clean`, `--no-tier2`, ...)
OVERRIDES the preset. `pipeline` is a dict of `PipelineConfig` kwargs (kept as a plain dict
so this module stays import-light: no `pipeline` import; the driver constructs the config).
"""
from __future__ import annotations


from . import config

# Neuron sets for the ORIGINAL/target worm (moved out of batch.py so the run config
# lives in one editable place).
ALL_NEURONS = [
    'AIAR', 'RIS', 'GLRVR', 'PLNR', 'SAADL', 'AVBR', 'PVQL', 'URADR',
    'AVBL', 'RIBL', 'SAAVR', 'RMED', 'PLNL', 'AVHL', 'AVM',
    'SDQL', 'PVWL_or_R_3', 'RMGL', 'RMHL', 'SMDDL', 'AINL', 'PVPL', 'RMDL',
    'RMFL', 'AVDR', 'URYDR', 'SMBVL', 'ALA', 'RICL', 'SMDVL', 'RIGL', 'SABD',
    'ADAL', 'AIAL', 'AVAR', 'FLPR', 'URAVL', 'RMEL', 'URYVL', 'URBL', 'AIZL',
    'AVJR', 'URBR', 'RIML', 'AIMR', 'ALNR', 'PVDL', 'SMBDL', 'SAAVL', 'ALMR',
    'RIAL', 'VB1', 'SDQR', 'PVPR', 'SIADR', 'AIZR', 'AIYR', 'RIR', 'PVCL',
    'PVR', 'SIBDL', 'RMDVR', 'RIAR', 'RID', 'SMDVR', 'AUAL', 'PVWL_or_R_1',
    'RICR', 'AVFL', 'AIBR', 'BDUL', 'SIADL', 'AVFR', 'SMDDR', 'PVT',
    'ALML', 'RMER', 'PVQR', 'RIPL', 'RMGR', 'AVHR', 'RIPR', 'RMHR', 'RMFR',
    'PVWL_or_R_2', 'AIYL', 'BDUR', 'RIVR', 'AVKR', 'RMEV', 'RMDR', 'AIML',
    'AVER', 'RIFR', 'SIBDR', 'RIMR', 'RMDDR', 'AVKL', 'RIBR', 'CANR',
    'DVA', 'SIAVR', 'AVJL', 'RIFL', 'SAADR', 'AIBL', 'URAVR',
    'AVEL', 'ADAR', 'AINR', 'SIBVL', 'RMDVL', 'SIAVL', 'AVL', 'AUAR',
    'SMBVR', 'DVC', 'URADL', 'PVCR', 'URYVR', 'AVAL', 'RMDDL', 'SIBVR',
    'PVDR', 'URYDL', 'ALNL', 'FLPL', 'AVDL', 'SABVL', 'RIH', 'RIGR', 'RIVL', 'SMBDR']

KEY_NEURONS = ['AIYR', 'AIYL', 'AIAR', 'AIAL', 'AIZL', 'AIZR', 'AIBL', 'AIBR',
               'URAVR', 'URAVL', 'URADL', 'URADR', 'RIH', 'RIPL', 'RIPR']

# The shared accuracy-first PipelineConfig knobs (same for both worms today).
_PIPELINE = dict(model_size="large", scale=8, save_downscale=8,
                 k_max_neg=3, neg_radius=150, box_margin=10,
                 chain_crop_from_mask=True)

# Fixed neuron subset for the resolution-experiment presets (original_fullres /
# _tier2forced / _bigimg / _stitch). Baked into the presets, not passed at submit, so
# every variant runs the IDENTICAL subset and the cross-variant comparison is fair.
EXP_NEURONS = KEY_NEURONS + ["AVAL"]

PRESETS = {
    "eval": {
        "dataset": "sem-dauer-1",
        # multimask anchor selection ON (node-anchored pick of SAM2's 3 candidates) to fight
        # the cross-worm bleed; off in the earlier baseline runs (out_gt_multichain). The
        # anti-bleed refinement is a separate flag, "multimask_exclude_neg": True, flip it on
        # once this (A) is measured against the baseline; see the ADR + _select_anchor_mask.
        "pipeline": {**_PIPELINE, "multimask_anchor": True},
        "output_root": config.GT_PRED_DIR / "batch_masks",
        "frames_root": config.GT_PRED_DIR / "frames",
        "tier2_on_flagged": True, "tier2_all": False, "gif_mode": "off",
        "clean": False, "neurons": None,          # GT run requires an explicit --neurons/--neuron-limit/--all
        "score_out": "eval/out_gt",                # default out dir for eval.score_batch
    },
    "original": {
        "dataset": "target",
        "pipeline": dict(_PIPELINE),
        "output_root": config.OUTPUT_ROOT,
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": True, "tier2_all": True, "gif_mode": "all",
        "clean": False, "neurons": KEY_NEURONS[0:3],
        "score_out": None,                         # no cross-worm GT scoring for the target worm
    },

    # --- resolution experiments (target worm, EXP_NEURONS subset, distinct outputs) ---
    # A head-to-head of ways to spend the cluster's compute on resolution. All share the
    # subset, model, and seed knobs so only the resolution strategy differs. See the spec
    # docs/superpowers/specs/2026-07-06-fullres-resolution-experiments-design.md. NOTE
    # (verified): SAM2 resizes every frame/crop to its internal image_size (1024 by
    # default), so "whole image at full res" does NOT raise effective resolution vs
    # scale-8; only cropping (tier-2/stitch) or raising image_size (bigimg) does.
    "original_fullres": {
        # Variant 1: whole frame at full res, no tier-2. Expected equal-or-worse than
        # scale-8 (SAM2 downsamples the ~9.7k-px frame ~9.5x to 1024); a measured baseline.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "scale": 1, "save_downscale": 1},
        "output_root": config.OUTPUT_ROOT.parent / "exp_fullres",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": False, "tier2_all": False, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
    "original_tier2forced": {
        # Variant 2: tier-2 crop on EVERY chain with the fallback floor dropped to 0, so
        # no chain reverts to the scale-8 full frame. The high-res mechanism that actually
        # works, applied universally.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "chain_crop_min_image_score": 0.0},
        "output_root": config.OUTPUT_ROOT.parent / "exp_tier2forced",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": True, "tier2_all": True, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
    "original_bigimg": {
        # Variant 3: raise SAM2's internal input resolution to 2048 and feed scale-4 frames
        # (~2432 px >= 2048) so the extra pixels are real, no tier-2. RISKY: off-distribution
        # for the pretrained encoder and may OOM on a 40GB A100 (memory ~quadratic in
        # image_size x ~300 stored frames). The build self-verifies image_size took effect.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "scale": 4, "save_downscale": 4, "image_size": 2048},
        "output_root": config.OUTPUT_ROOT.parent / "exp_bigimg",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": False, "tier2_all": False, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
    "original_wholeimg_s4": {
        # Control: whole frame, no tier-2, IDENTICAL to original_fullres but scale=4
        # (~2432 px) instead of scale=1 (~9728 px), with image_size at its default 1024.
        # Tests the "SAM2 downsamples every whole-image input to 1024" claim: both feed SAM2
        # a 1024 view, so the masks should be SIMILAR (not identical: different pre-downsample
        # filter + JPEG per size). A small delta vs original_fullres confirms the 1024
        # bottleneck; a large delta would refute it. Pairs with bigimg, which instead CHANGES
        # image_size, so the two isolate the scale knob from the input-size knob.
        "dataset": "target",
        "pipeline": {**_PIPELINE, "scale": 4, "save_downscale": 4},
        "output_root": config.OUTPUT_ROOT.parent / "exp_wholeimg_s4",
        "frames_root": config.FRAMES_ROOT,
        "tier2_on_flagged": False, "tier2_all": False, "gif_mode": "all",
        "clean": False, "neurons": EXP_NEURONS,
        "score_out": None,
    },
}


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]
