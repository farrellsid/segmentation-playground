"""The launcher's config layer, tested without constructing a window."""
import importlib
import json
import os

import pytest

import launcher


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Reload sam2_utils.config back to real defaults after every test in this file.

    apply_profile_env now reloads sam2_utils.config in place (that reload IS the
    fix for C1: config.WORM_PATH etc. were frozen at their defaults because the
    module was already imported by the time the old apply_profile_env ran). A test
    below that calls apply_profile_env with a tmp_path leaves that reload in
    sys.modules for the rest of the process, exactly the leak
    tests/test_config_env.py's own fixture guards against, so this file needs the
    same guard for the same reason.
    """
    yield
    for var in ("SAM2_OUTPUT_ROOT", "SAM2_FRAMES_ROOT",
                "SAM2_WORM_PATH", "SAM2_CHECKPOINT_DIR"):
        os.environ.pop(var, None)
    from sam2_utils import config
    importlib.reload(config)


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


def test_apply_profile_env_makes_config_actually_see_the_new_paths(monkeypatch, tmp_path):
    """C1 regression test: setting the env vars is not the same as config observing
    them, because sam2_utils.config reads them at MODULE IMPORT time and the module
    is already imported by the time apply_profile_env runs (machine_checks imports
    it during the launcher's own startup). Asserting only os.environ, which the
    other apply_profile_env tests above do, is exactly what missed this: the env
    var can be set correctly while config.WORM_PATH still reports the old value.

    Reproduces the reviewer's actual failure: fill in the profile, apply it, then
    ask config for the path pipeline.raw_em_problem() would have read.
    """
    for var in ("SAM2_OUTPUT_ROOT", "SAM2_FRAMES_ROOT",
                "SAM2_WORM_PATH", "SAM2_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)

    # Force sam2_utils.config to already be imported with its defaults, the same
    # state it is in by the time a real launcher session reaches apply_profile_env
    # (machine_checks imports it first, well before do_launch).
    import importlib
    from sam2_utils import config
    importlib.reload(config)
    assert config.WORM_PATH != tmp_path / "worm"

    launcher.apply_profile_env({
        "output_root": str(tmp_path / "out"),
        "frames_root": str(tmp_path / "frames"),
        "worm_path": str(tmp_path / "worm"),
        "checkpoint_dir": str(tmp_path / "ckpts"),
    })

    from sam2_utils import config as cfg_after
    assert cfg_after.WORM_PATH == tmp_path / "worm"
    assert cfg_after.OUTPUT_ROOT == tmp_path / "out"
    assert cfg_after.FRAMES_ROOT == tmp_path / "frames"
    assert cfg_after.CHECKPOINT_DIR == tmp_path / "ckpts"


def test_full_mode_refuses_at_the_decision_point_not_just_the_combo(tmp_path, monkeypatch):
    """I5's second half: a stale widget state (a saved profile from a machine that
    could run full mode, reopened on one that cannot) must not be enough to start a
    session that fails at predictor build. build_launch_kwargs is the decision
    point do_launch actually calls, so the refusal belongs here, not only in the
    combo's enabled state.
    """
    monkeypatch.setattr(launcher, "torch_available", lambda: True)
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path), ui_mode="full",
               checkpoint_dir="", worm_path="", frames_root="")
    with pytest.raises(ValueError, match="checkpoint"):
        launcher.build_launch_kwargs(prof)


def test_full_mode_is_accepted_when_the_machine_can_actually_run_it(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "torch_available", lambda: True)
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    prof = dict(launcher.DEFAULT_PROFILE, output_root=str(tmp_path), ui_mode="full",
               checkpoint_dir=str(ckpt), worm_path="", frames_root="")
    kw = launcher.build_launch_kwargs(prof)
    assert kw["ui_mode"] == "full"
