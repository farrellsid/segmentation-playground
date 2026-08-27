"""pipeline.raw_em_problem: the single rule for whether a path is the raw EM stack.

Recrop re-reads full-resolution tifs, so a wrong or unset path is the difference
between a working recrop and a failure deep inside a tif read. Both the launcher's
preflight and the GUI's recrop guard ask this one function, so the rule cannot drift
between the place that reports it and the place that enforces it.

    py -3 -m pytest tests/test_raw_em_problem.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline


def _stack(tmp_path):
    """A directory shaped like the real stack: '1301____z1300.0.tif'."""
    d = tmp_path / "raw"
    d.mkdir()
    for file_z in (1300, 1301, 1302):
        (d / f"{file_z + 1}____z{file_z}.0.tif").write_bytes(b"")
    return d


def test_a_real_looking_stack_has_no_problem(tmp_path):
    assert pipeline.raw_em_problem(_stack(tmp_path)) is None


def test_an_unset_path_says_so(tmp_path):
    assert "not set" in pipeline.raw_em_problem("")


def test_a_missing_directory_is_named(tmp_path):
    problem = pipeline.raw_em_problem(tmp_path / "nope")
    assert "nope" in problem


def test_a_file_is_not_a_stack(tmp_path):
    f = tmp_path / "stack.tif"
    f.write_bytes(b"")
    assert pipeline.raw_em_problem(f) is not None


def test_an_empty_directory_is_a_problem(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert pipeline.raw_em_problem(d) is not None


def test_tifs_that_are_not_the_stack_do_not_pass(tmp_path):
    """A folder of unrelated tifs is the dangerous case: it looks configured, and
    every frame lookup then fails one at a time instead of once, up front."""
    d = tmp_path / "other"
    d.mkdir()
    (d / "screenshot.tif").write_bytes(b"")
    (d / "figure_panel.tif").write_bytes(b"")
    problem = pipeline.raw_em_problem(d)
    assert problem is not None and "z" in problem


def test_one_good_frame_among_others_is_enough(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "notes.tif").write_bytes(b"")
    (d / "1301____z1300.0.tif").write_bytes(b"")
    assert pipeline.raw_em_problem(d) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
