"""CPU tests for the resolution-experiment presets and the image_size override plumbing.

Torch-free: exercises the pure hydra-override helper, the fail-loud post-build check, and
that every experiment preset builds a valid PipelineConfig with the intended knobs. The
actual SAM2 build (and whether image_size=2048 fits a given GPU) is only observable on the
cluster; here we lock in the config-layer contract those runs depend on.
"""
from __future__ import annotations

import types

import pytest

from pipeline import PipelineConfig
from sam2_utils import presets, setup


# --- image_size hydra override + fail-loud check ---------------------------------------

def test_image_size_overrides_none_is_empty():
    assert setup._image_size_overrides(None) == []
    assert setup._image_size_overrides(0) == []


def test_image_size_overrides_sets_model_key():
    assert setup._image_size_overrides(2048) == ["++model.image_size=2048"]


def test_assert_image_size_noop_when_not_requested():
    # None/0 requested -> never raises, whatever the model reports.
    setup._assert_image_size(types.SimpleNamespace(image_size=1024), None, "t")
    setup._assert_image_size(types.SimpleNamespace(image_size=1024), 0, "t")


def test_assert_image_size_passes_on_match():
    setup._assert_image_size(types.SimpleNamespace(image_size=2048), 2048, "t")


def test_assert_image_size_raises_on_silent_noop():
    # The exact failure the check exists to catch: override did not take, model still 1024.
    with pytest.raises(RuntimeError, match="did not take"):
        setup._assert_image_size(types.SimpleNamespace(image_size=1024), 2048, "t")


# --- experiment presets ----------------------------------------------------------------

EXP_PRESETS = ["original_fullres", "original_tier2forced", "original_bigimg",
               "original_wholeimg_s4"]


@pytest.mark.parametrize("name", EXP_PRESETS)
def test_exp_preset_builds_valid_config(name):
    p = presets.get_preset(name)
    cfg = PipelineConfig(**p["pipeline"])          # would raise on an unknown/typo'd knob
    assert cfg.model_size == "large"               # experiments hold the model fixed
    assert p["neurons"] == presets.EXP_NEURONS     # identical subset across variants
    assert p["score_out"] is None                  # target worm: not GT-scored


def test_exp_neurons_is_key_plus_aval():
    assert presets.EXP_NEURONS == presets.KEY_NEURONS + ["AVAL"]
    assert len(presets.EXP_NEURONS) == 16
    assert "AVAL" in presets.EXP_NEURONS


def test_fullres_is_whole_image_no_downscale():
    cfg = PipelineConfig(**presets.get_preset("original_fullres")["pipeline"])
    assert cfg.scale == 1 and cfg.save_downscale == 1   # scale == save_downscale (qc guard)
    assert cfg.image_size is None                       # SAM2 default 1024 internally


def test_tier2forced_drops_the_fallback_floor():
    p = presets.get_preset("original_tier2forced")
    cfg = PipelineConfig(**p["pipeline"])
    assert cfg.chain_crop_min_image_score == 0.0        # nothing falls back to full frame
    assert p["tier2_all"] is True


def test_bigimg_raises_image_size_and_feeds_finer_frames():
    cfg = PipelineConfig(**presets.get_preset("original_bigimg")["pipeline"])
    assert cfg.image_size == 2048
    # on-disk frame must be finer than image_size or the extra pixels are empty upscale
    assert cfg.scale == 4 and cfg.save_downscale == 4


def test_wholeimg_s4_is_the_scale_control():
    # Same whole-image, no-tier-2, default-image_size path as fullres, only scale differs,
    # so it isolates the scale knob. image_size None (unlike bigimg) is what makes it a
    # control for "does whole-image scale down to the same 1024 view".
    ctrl = PipelineConfig(**presets.get_preset("original_wholeimg_s4")["pipeline"])
    fullres = PipelineConfig(**presets.get_preset("original_fullres")["pipeline"])
    assert ctrl.image_size is None and fullres.image_size is None
    assert ctrl.scale == 4 and fullres.scale == 1
    assert ctrl.save_downscale == ctrl.scale        # qc guard
