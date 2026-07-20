from sam2_utils.presets import PRESETS


def test_perslice_guard_presets_set_blowup_guard():
    for name in ("original_perslice_only_guard", "original_perslice_guard"):
        p = PRESETS[name]["pipeline"]
        assert p["per_slice_reseed"] is True
        assert p["blowup_guard"] is True


def test_genfirst_negcrop_seeds_split_across_passes():
    p = PRESETS["original_genfirst_negcrop"]["pipeline"]
    # first _sam pass: generous, no negatives
    assert p["multimask_generous"] is True
    assert p["k_max_neg"] == 0 and p["seed_negatives"] is False
    assert p["chain_crop_from_mask"] is True
    # tier-2 crop pass overrides: negatives on, not generous
    assert p["tier2_k_max_neg"] == 3 and p["tier2_seed_negatives"] is True
    assert p["tier2_multimask_generous"] is False
    assert PRESETS["original_genfirst_negcrop"]["tier2_all"] is True
