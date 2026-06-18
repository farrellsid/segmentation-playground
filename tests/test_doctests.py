"""Run the doctests embedded in the pure-utility modules.

These modules have no GPU or torch dependency, so their docstring examples run
as part of the CPU-only suite and double as always-current usage reference.
"""
import doctest
import pathlib
import sys

# Make `from sam2_utils import ...` work no matter how this file is invoked
# (pytest from the repo root, or `py -3 tests/test_doctests.py` from anywhere).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils import alignment


def test_alignment_doctests():
    results = doctest.testmod(alignment, verbose=False)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in alignment"


if __name__ == "__main__":
    test_alignment_doctests()
    print("doctests OK")
