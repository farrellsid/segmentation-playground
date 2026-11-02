"""_peek_chain: the pure "what would next/prev CHAIN open" lookup, factored
out of _step_chain so the background prefetch (Task 3) can ask the same
question without opening anything or touching GUI state.

    py -3 -m pytest tests/test_gui_peek_chain.py
"""
from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


def _fake_gui(chains, *, neuron, chain_idx):
    return types.SimpleNamespace(
        queue=types.SimpleNamespace(refresh=lambda: None),
        _mode_chains=lambda: list(chains),
        neuron=neuron, chain_idx=chain_idx)


def test_peek_forward_returns_the_next_chain_in_the_list():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 1)


def test_peek_forward_wraps_around_at_the_end():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAR", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 0)


def test_peek_backward_wraps_around_at_the_start():
    g = _fake_gui([("AIAL", 0), ("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, -1) == ("AIAR", 0)


def test_peek_returns_none_for_an_empty_queue():
    g = _fake_gui([], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) is None


def test_peek_returns_none_when_the_open_chain_is_the_only_one():
    g = _fake_gui([("AIAL", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) is None


def test_peek_when_the_open_chain_left_the_list_steps_from_the_edge():
    # the current chain was disposed mid-review and no longer appears; forward
    # lands on the first chain of whatever is left, backward on the last
    g = _fake_gui([("AIAL", 1), ("AIAR", 0)], neuron="AIAL", chain_idx=0)
    assert gui.ReviewGUI._peek_chain(g, +1) == ("AIAL", 1)
    assert gui.ReviewGUI._peek_chain(g, -1) == ("AIAR", 0)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
