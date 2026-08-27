"""The launcher's config layer, tested without constructing a window."""
import json
import os

import pytest

import launcher


def test_default_profile_is_review_mode():
    assert launcher.DEFAULT_PROFILE["ui_mode"] == "review"


def test_load_profile_returns_defaults_when_absent(tmp_path):
    prof = launcher.load_profile(tmp_path / "nope.json")
    assert prof == launcher.DEFAULT_PROFILE


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "profile.json"
    prof = dict(launcher.DEFAULT_PROFILE, output_root="D:/x", neurons=["AIAL"])
    launcher.save_profile(prof, path)
    assert launcher.load_profile(path) == prof


def test_load_merges_unknown_keys_over_defaults(tmp_path):
    path = tmp_path / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reviewer": "lucinda"}), encoding="utf-8")
    prof = launcher.load_profile(path)
    assert prof["reviewer"] == "lucinda"
    assert prof["ui_mode"] == launcher.DEFAULT_PROFILE["ui_mode"]


def test_build_launch_kwargs_maps_the_profile(tmp_path):
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path),
                neurons=["AIAL", "AIYL"], reviewer="lucinda", ui_mode="review")
    kw = launcher.build_launch_kwargs(prof)
    assert kw["output_root"] == tmp_path
    assert kw["neurons"] == ["AIAL", "AIYL"]
    assert kw["reviewer"] == "lucinda"
    assert kw["ui_mode"] == "review"


def test_empty_neuron_list_becomes_none(tmp_path):
    """An empty tick list means every neuron, not zero neurons."""
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path), neurons=[])
    assert launcher.build_launch_kwargs(prof)["neurons"] is None


def test_missing_output_root_is_rejected():
    prof = dict(launcher.DEFAULT_PROFILE, output_root="")
    with pytest.raises(ValueError):
        launcher.build_launch_kwargs(prof)


def test_full_mode_requires_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "torch_available", lambda: False)
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path), ui_mode="full")
    with pytest.raises(ValueError):
        launcher.build_launch_kwargs(prof)


def test_review_mode_does_not_require_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "torch_available", lambda: False)
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path), ui_mode="review")
    assert launcher.build_launch_kwargs(prof)["ui_mode"] == "review"


def test_machine_paths_default_to_empty():
    """Empty means 'not set', which the checks report as such. A wrong default
    would be worse than none: it reads as configured and fails later."""
    assert launcher.DEFAULT_PROFILE["worm_path"] == ""
    assert launcher.DEFAULT_PROFILE["checkpoint_dir"] == ""


def test_apply_profile_env_exports_every_machine_path(monkeypatch):
    for var in ("SAM2_OUTPUT_ROOT", "SAM2_FRAMES_ROOT",
                "SAM2_WORM_PATH", "SAM2_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    launcher.apply_profile_env({"output_root": "/o", "frames_root": "/f",
                                "worm_path": "/w", "checkpoint_dir": "/c"})
    assert os.environ["SAM2_OUTPUT_ROOT"] == "/o"
    assert os.environ["SAM2_FRAMES_ROOT"] == "/f"
    assert os.environ["SAM2_WORM_PATH"] == "/w"
    assert os.environ["SAM2_CHECKPOINT_DIR"] == "/c"


def test_apply_profile_env_leaves_unset_paths_alone(monkeypatch):
    """An empty field must not export an empty env var: config would then read ''
    as the path instead of falling back to its default."""
    monkeypatch.delenv("SAM2_WORM_PATH", raising=False)
    launcher.apply_profile_env({"output_root": "/o", "worm_path": ""})
    assert "SAM2_WORM_PATH" not in os.environ


def test_a_profile_written_before_these_keys_existed_still_loads(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"output_root": "/o", "reviewer": "lucinda"}),
                    encoding="utf-8")
    prof = launcher.load_profile(path)
    assert prof["worm_path"] == "" and prof["checkpoint_dir"] == ""
    assert prof["reviewer"] == "lucinda"
