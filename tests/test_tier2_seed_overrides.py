from dataclasses import replace
from pipeline.config import PipelineConfig
from batch import _tier2_overrides


def test_overrides_empty_when_all_none():
    assert _tier2_overrides(PipelineConfig()) == {}


def test_overrides_map_set_fields_to_base_names():
    cfg = PipelineConfig(tier2_k_max_neg=3, tier2_seed_negatives=True,
                         tier2_multimask_generous=False)
    assert _tier2_overrides(cfg) == {
        "k_max_neg": 3, "seed_negatives": True, "multimask_generous": False}


def test_all_none_rerun_equals_plain_chain_crop():
    cfg = PipelineConfig()
    assert replace(cfg, chain_crop=True, **_tier2_overrides(cfg)) == replace(cfg, chain_crop=True)
