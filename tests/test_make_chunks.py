"""Pure-logic tests for cluster/make_chunks.py chunking (torch-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cluster"))

from make_chunks import chunk_neurons, write_chunks


def test_chunks_are_disjoint_and_cover_everything():
    neurons = [f"N{i}" for i in range(20)]
    chunks = chunk_neurons(neurons, 7)
    # 20 / 7 -> 3 chunks of sizes 7, 7, 6
    assert [len(c) for c in chunks] == [7, 7, 6]
    flat = [n for c in chunks for n in c]
    assert flat == neurons                 # order preserved, nothing dropped/duplicated
    assert len(set(flat)) == len(neurons)  # disjoint


def test_exact_multiple_has_no_short_tail():
    chunks = chunk_neurons([f"N{i}" for i in range(14)], 7)
    assert [len(c) for c in chunks] == [7, 7]


def test_chunk_larger_than_list_gives_one_chunk():
    chunks = chunk_neurons(["A", "B"], 7)
    assert chunks == [["A", "B"]]


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_neurons(["A"], 0)


def test_write_chunks_round_trips(tmp_path):
    chunks = [["A", "B"], ["C"]]
    out = tmp_path / "neuron_chunks.txt"
    write_chunks(chunks, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["A B", "C"]
