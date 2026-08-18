"""The launcher's config layer, tested without constructing a window."""
import json

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
