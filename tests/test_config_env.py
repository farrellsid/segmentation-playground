"""Paths must be settable without editing tracked source."""
import importlib
from pathlib import Path


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
