import numpy as np

import pipeline
from pipeline import PipelineConfig, Prompts


def test_multimask_generous_defaults_false():
    assert PipelineConfig().multimask_generous is False


def _mask(hw, box):
    m = np.zeros(hw, dtype=bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def _prompts_at(x, y):
    return Prompts(points_sam=np.array([[x, y]], dtype=float), labels=np.array([1]))


def test_generous_prefers_larger_but_capped():
    hw = (100, 100)
    # positive node at (50, 50), sam-space.
    small = _mask(hw, (45, 55, 45, 55))    # nucleus-only, contains node, 1% area
    soma = _mask(hw, (30, 70, 30, 70))     # whole soma, contains node, 16% area
    whole = np.ones(hw, dtype=bool)        # whole-frame blob, 100% area, over the cap
    masks = np.stack([small, soma, whole])
    scores = np.array([0.95, 0.80, 0.99])  # SAM score alone would pick 'small' or 'whole'

    # strict (default): highest score among gate-passers -> 'small' (nucleus).
    idx_strict, _, _ = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(0.001, 0.5))
    assert idx_strict == 0

    # generous: prefer larger area among gate-passers, but 'whole' fails the area
    # cap (area_ok is unchanged), so it never wins -> 'soma'.
    idx_gen, mask_gen, _ = pipeline._select_anchor_mask(
        masks, scores, _prompts_at(50, 50), hw,
        contain_radius_px=3, area_bounds=(0.001, 0.5), generous=True)
    assert idx_gen == 1                     # soma, not the nucleus, not the whole-frame blob
    assert idx_gen != idx_strict
    assert mask_gen.sum() == soma.sum()
