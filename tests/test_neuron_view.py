"""Unit tests for the torch/napari-free helpers in gui_neuron: neuron enumeration from
on-disk chains and the multi-label neuron volume builder.

gui_neuron imports torch and napari only lazily (inside methods), so importing the
module and exercising these module-level helpers needs neither.

    py -3 -m pytest tests/test_neuron_view.py
    py -3 tests/test_neuron_view.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import gui_neuron


def test_neurons_on_disk_groups_chains():
    d = pathlib.Path(tempfile.mkdtemp())
    for neuron, idx in [("AVAL", 0), ("AVAL", 2), ("AVAR", 1)]:
        (d / neuron / f"chain_{idx:02d}").mkdir(parents=True)
    assert gui_neuron.neurons_on_disk(d) == [("AVAL", [0, 2]), ("AVAR", [1])]


def test_neurons_on_disk_empty_when_missing():
    d = pathlib.Path(tempfile.mkdtemp()) / "nope"
    assert gui_neuron.neurons_on_disk(d) == []


def test_build_neuron_label_volume_writes_each_branch():
    hw = (4, 4)
    b1 = np.zeros(hw, bool); b1[0:2, 0:2] = True
    b2 = np.zeros(hw, bool); b2[2:4, 2:4] = True
    vol = gui_neuron.build_neuron_label_volume({1: {0: b1}, 2: {0: b2}}, t=1, hw=hw)
    assert vol.shape == (1, 4, 4)
    assert vol[0, 0, 0] == 1 and vol[0, 3, 3] == 2 and vol[0, 0, 3] == 0


def test_build_neuron_label_volume_higher_label_wins_overlap():
    hw = (2, 2)
    a = np.ones(hw, bool)
    vol = gui_neuron.build_neuron_label_volume({1: {0: a}, 2: {0: a}}, t=1, hw=hw)
    assert (vol[0] == 2).all()


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
