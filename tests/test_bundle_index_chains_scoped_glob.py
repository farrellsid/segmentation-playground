"""index_chains(neurons=...) must not read chains outside the wanted neurons.

Real failure this guards against: a bundle export for one or two neurons under a
huge merged tree (129 neurons, a symlink forest) was reading every state.json in
the ENTIRE tree before filtering to the wanted names, so one bad/unreadable file
under an unrelated neuron killed exports that had nothing to do with it - and
scanned thousands of unwanted files every time regardless.
"""
from __future__ import annotations

import json

from sam2_utils import bundle


def _write_state(path, neuron, chain_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"neuron": neuron, "chain_idx": chain_idx}),
                    encoding="utf-8")


def test_scoped_neurons_never_touches_an_unwanted_neurons_bad_file(tmp_path):
    root = tmp_path / "tree"
    _write_state(root / "AIAL" / "chain_00" / "state.json", "AIAL", 0)
    _write_state(root / "AIAL" / "chain_01" / "state.json", "AIAL", 1)
    # ZZBAD sorts after AIAL, so an unscoped scan reaches it too; it is a
    # directory where a file is expected, guaranteed to raise on read.
    (root / "ZZBAD" / "chain_00" / "state.json").mkdir(parents=True)

    chains = bundle.index_chains(root, neurons=["AIAL"])

    assert len(chains) == 2
    assert all(c["cell_name"] == "AIAL" for c in chains)


def test_unscoped_call_still_covers_every_neuron(tmp_path):
    root = tmp_path / "tree"
    _write_state(root / "AIAL" / "chain_00" / "state.json", "AIAL", 0)
    _write_state(root / "AUAL" / "chain_00" / "state.json", "AUAL", 0)

    chains = bundle.index_chains(root, neurons=None)

    assert {c["cell_name"] for c in chains} == {"AIAL", "AUAL"}
