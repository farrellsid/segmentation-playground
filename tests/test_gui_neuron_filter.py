"""ReviewContext.neurons scopes a session's chain list to a launcher-chosen subset.

find_chain(neuron, chain_idx) indexes POSITIONALLY within a neuron's own chains,
not the global list, so filtering out OTHER neurons must never change what a
kept neuron's chains resolve to. This proves that equivalence directly rather
than assuming it from the filter's implementation.
"""
from pathlib import Path

import gui


def _make_chains():
    return [
        {"cell_name": "AIAL", "chain_idx": 0},
        {"cell_name": "AIAL", "chain_idx": 1},
        {"cell_name": "AIAR", "chain_idx": 0},
        {"cell_name": "AUAL", "chain_idx": 0},
        {"cell_name": "AIAL", "chain_idx": 2},
    ]


def test_find_chain_same_result_with_and_without_a_filter(tmp_path):
    chains = _make_chains()
    unfiltered = gui.ReviewContext(tmp_path, chains=chains)
    filtered = gui.ReviewContext(tmp_path, chains=chains, neurons=["AIAL", "AUAL"])

    for idx in range(3):
        assert filtered.find_chain("AIAL", idx) == unfiltered.find_chain("AIAL", idx)
    assert filtered.find_chain("AUAL", 0) == unfiltered.find_chain("AUAL", 0)


def test_filtered_chains_excludes_neurons_outside_the_subset():
    chains = _make_chains()
    ctx = gui.ReviewContext(Path("."), chains=chains, neurons=["AIAL"])
    assert all(c["cell_name"] == "AIAL" for c in ctx.chains)
    assert ctx.find_chain("AIAR", 0) is None


def test_no_filter_returns_every_chain():
    chains = _make_chains()
    ctx = gui.ReviewContext(Path("."), chains=chains)
    assert ctx.chains == chains
