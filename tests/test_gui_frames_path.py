"""A bundle's state.json points at a relative frames dir; it must resolve locally."""
from pathlib import Path

import gui


def test_relative_path_resolves_against_the_chain_dir(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_00"
    chain_dir.mkdir(parents=True)
    assert gui.resolve_frames_dir("frames", chain_dir) == chain_dir / "frames"


def test_nested_relative_path_resolves(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_00"
    chain_dir.mkdir(parents=True)
    assert gui.resolve_frames_dir("frames/s2", chain_dir) == chain_dir / "frames" / "s2"


def test_absolute_path_is_returned_unchanged(tmp_path):
    chain_dir = tmp_path / "AIAL" / "chain_00"
    chain_dir.mkdir(parents=True)
    recorded = str(tmp_path / "elsewhere" / "frames")
    assert gui.resolve_frames_dir(recorded, chain_dir) == Path(recorded)


def test_none_returns_none(tmp_path):
    assert gui.resolve_frames_dir(None, tmp_path) is None


def test_posix_absolute_is_not_treated_as_relative(tmp_path):
    """A Narval scratch path must stay absolute, not get joined onto the chain dir.

    We assert equality (not inequality) to catch pathlib substitutions on Windows
    where a naive join would produce an invented drive path instead of the original.
    """
    out = gui.resolve_frames_dir("/localscratch/12345/frames/x", tmp_path)
    assert out == Path("/localscratch/12345/frames/x")
