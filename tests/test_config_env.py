"""Paths must be settable without editing tracked source."""
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Reload sam2_utils.config back to real defaults after every test in this file.

    monkeypatch.setenv/delenv only restores the environment variable at
    teardown, it does not undo an importlib.reload() done during the test.
    Once a test here reloads sam2_utils.config with SAM2_OUTPUT_ROOT or
    SAM2_FRAMES_ROOT set, the module keeps those values (a tmp_path under
    pytest's temp dir) for the rest of the process, no matter what
    monkeypatch does to the env var afterward. Any later test or module that
    imports sam2_utils.config would silently see the polluted paths unless we
    force one more reload here, with the env vars actually removed, after
    each test.
    """
    yield
    os.environ.pop("SAM2_OUTPUT_ROOT", None)
    os.environ.pop("SAM2_FRAMES_ROOT", None)
    from sam2_utils import config
    importlib.reload(config)


def _reload(monkeypatch, **env):
    from sam2_utils import config
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(config)


def test_output_root_honours_env(monkeypatch, tmp_path):
    cfg = _reload(monkeypatch, SAM2_OUTPUT_ROOT=str(tmp_path / "out"))
    assert cfg.OUTPUT_ROOT == Path(tmp_path / "out")


def test_frames_root_honours_env(monkeypatch, tmp_path):
    cfg = _reload(monkeypatch, SAM2_FRAMES_ROOT=str(tmp_path / "frames"))
    assert cfg.FRAMES_ROOT == Path(tmp_path / "frames")


def test_defaults_survive_without_env(monkeypatch):
    from sam2_utils import config
    monkeypatch.delenv("SAM2_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("SAM2_FRAMES_ROOT", raising=False)
    cfg = importlib.reload(config)
    assert isinstance(cfg.OUTPUT_ROOT, Path)
    assert isinstance(cfg.FRAMES_ROOT, Path)
