"""launcher.machine_checks: what this machine can actually do, before a session starts.

Full mode used to unlock on a bare `import torch`, so a machine with torch and no
checkpoint offered the reprop controls and failed minutes later, and one with no raw EM
offered recrop and failed deep inside a tif read. These are the checks that make the
window tell the truth first.

No Qt and no GPU: machine_checks is pure, the same split that lets
tests/test_launcher_config.py exercise build_launch_kwargs without a display.

    py -3 -m pytest tests/test_launcher_machine_checks.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import launcher


def _named(checks, name):
    for c in checks:
        if c.name == name:
            return c
    raise AssertionError(f"no check named {name!r} in {[c.name for c in checks]}")


def _stack(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "1301____z1300.0.tif").write_bytes(b"")
    return d


def _good_profile(tmp_path):
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    (ckpt / "sam2.1_hiera_large.pt").write_bytes(b"")
    frames = tmp_path / "frames"
    frames.mkdir()
    return {"worm_path": str(_stack(tmp_path)), "checkpoint_dir": str(ckpt),
            "frames_root": str(frames)}


@pytest.fixture(autouse=True)
def _torch_present(monkeypatch):
    """Default every test to a machine that HAS torch, so each test states only the
    one thing it is about."""
    monkeypatch.setattr(launcher, "torch_available", lambda: True)
    monkeypatch.setattr(launcher, "_device_name", lambda: "cuda")


def test_every_check_has_a_name_and_a_verdict(tmp_path):
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert {c.name for c in checks} == {"torch", "device", "checkpoint",
                                        "raw EM", "frames cache"}
    assert all(isinstance(c.ok, bool) for c in checks)


def test_a_fully_configured_machine_passes_everything(tmp_path):
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert all(c.ok for c in checks), [c.name for c in checks if not c.ok]
    assert launcher.can_run_model(checks) is True


def test_missing_torch_fails_and_says_how_to_fix_it(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "torch_available", lambda: False)
    checks = launcher.machine_checks(_good_profile(tmp_path))
    torch_check = _named(checks, "torch")
    assert torch_check.ok is False and torch_check.fix
    assert launcher.can_run_model(checks) is False


def test_missing_torch_reports_device_as_unknown_not_a_bug(monkeypatch, tmp_path):
    """M7: with torch absent, the real _device_name() imports torch inside
    setup_device and raises. The generic _check() wrapper then catches that and
    reports device as ok=False with "this is a bug; report it", which is wrong
    twice over: a missing torch is the torch check's job to report, and the spec
    promises device is never ok=False.

    The file's autouse fixture always stubs _device_name to a harmless "cuda",
    including in the missing-torch case, which is exactly why this went uncaught:
    no test here ever let a torch-absent machine reach the real function. This one
    makes _device_name blow up if called at all, so it fails against the old code
    (which called it unconditionally) and passes only once the device check learns
    to ask torch_available() first and skip the call entirely.
    """
    monkeypatch.setattr(launcher, "torch_available", lambda: False)

    def _must_not_be_called():
        raise ModuleNotFoundError("No module named 'torch' (simulating the real "
                                  "setup_device failure a torch-absent machine hits)")
    monkeypatch.setattr(launcher, "_device_name", _must_not_be_called)

    checks = launcher.machine_checks(_good_profile(tmp_path))
    device_check = _named(checks, "device")
    assert device_check.ok is True, device_check
    assert "torch" in device_check.detail.lower()


def test_a_missing_checkpoint_is_a_pass_with_a_download_warning(tmp_path):
    """setup.ensure_checkpoint downloads rather than failing, so this is not a
    blocker. It IS a surprise on a laptop, so the size is stated."""
    prof = _good_profile(tmp_path)
    (pathlib.Path(prof["checkpoint_dir"]) / "sam2.1_hiera_large.pt").unlink()
    check = _named(launcher.machine_checks(prof), "checkpoint")
    assert check.ok is True
    assert "download" in check.detail.lower()


def test_an_uncreatable_checkpoint_dir_fails(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory", encoding="utf-8")
    prof = dict(_good_profile(tmp_path), checkpoint_dir=str(blocker))
    check = _named(launcher.machine_checks(prof), "checkpoint")
    assert check.ok is False and launcher.can_run_model(
        launcher.machine_checks(prof)) is False


def test_a_missing_raw_em_fails_with_the_reason_from_pipeline(tmp_path):
    prof = dict(_good_profile(tmp_path), worm_path=str(tmp_path / "gone"))
    check = _named(launcher.machine_checks(prof), "raw EM")
    assert check.ok is False and "gone" in check.detail


def test_raw_em_does_not_block_review_only_work(tmp_path):
    """Recrop needs the raw EM; re-predict and resume propagation do not. A missing
    stack must not disable the model outright."""
    prof = dict(_good_profile(tmp_path), worm_path="")
    assert launcher.can_run_model(launcher.machine_checks(prof)) is True


def test_an_unwritable_frames_cache_fails(tmp_path):
    blocker = tmp_path / "cache_is_a_file"
    blocker.write_text("x", encoding="utf-8")
    prof = dict(_good_profile(tmp_path), frames_root=str(blocker))
    assert _named(launcher.machine_checks(prof), "frames cache").ok is False


def test_the_writability_probe_leaves_nothing_behind(tmp_path):
    prof = _good_profile(tmp_path)
    launcher.machine_checks(prof)
    assert list(pathlib.Path(prof["frames_root"]).iterdir()) == []


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_the_device_is_reported_and_never_blocks(monkeypatch, tmp_path, device):
    monkeypatch.setattr(launcher, "_device_name", lambda: device)
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert check.ok is True and device in check.detail


def test_mps_carries_the_upstream_caveat(monkeypatch, tmp_path):
    """SAM2 on MPS is preliminary upstream and may differ from CUDA. A reviewer
    comparing her masks with the batch's needs to know that."""
    monkeypatch.setattr(launcher, "_device_name", lambda: "mps")
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert "preliminary" in check.detail.lower()


def test_slow_devices_say_so(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_device_name", lambda: "cpu")
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert "slow" in check.detail.lower()


def test_one_broken_check_does_not_hide_the_others(monkeypatch, tmp_path):
    """machine_checks catches per check, the same reason validate_bundle returns a
    list: a reviewer should see everything wrong at once."""
    def _boom():
        raise RuntimeError("torch import exploded")
    monkeypatch.setattr(launcher, "torch_available", _boom)
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert len(checks) == 5
    assert _named(checks, "torch").ok is False
    assert _named(checks, "raw EM").ok is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
