"""A review tree must never advertise chains it does not have.

`_manifest.csv` and `_triage.csv` sit at the tree root and ARE the GUI's chain queue.
The copy loop used to skip a neuron the source lacked with a warning, while the CSV
filter still keyed on the REQUESTED neurons, so the queue listed chains with no
directory behind them. Real instance: `manual_verify_RMDD` came out claiming 115 chains
over 57 real ones, because the source tree's own manifest still lists RMDDL while its
RMDDL directory is absent. A warning printed dozens of lines earlier is not a guard.

Torch-free (pandas only):
    py -3 -m pytest tests/test_make_review_tree_missing_neuron.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.make_review_tree import make_review_tree


def _source(tmp_path, present=("RMDDR",), listed=("RMDDL", "RMDDR")):
    """A source tree whose manifest lists more neurons than it has directories for,
    which is the real inconsistency observed on disk."""
    src = tmp_path / "src"
    for n in present:
        for i in range(2):
            (src / n / f"chain_{i:02d}" / "masks").mkdir(parents=True)
            (src / n / f"chain_{i:02d}" / "state.json").write_text("{}")
    rows = [{"neuron": n, "chain_idx": i, "status": "done"} for n in listed for i in range(2)]
    for name in ("_manifest.csv", "_triage.csv"):
        pd.DataFrame(rows).to_csv(src / name, index=False)
    (src / "_run_meta.json").write_text("{}")
    return src


class TestMissingNeuron:
    def test_refuses_by_default_and_writes_nothing(self, tmp_path):
        src = _source(tmp_path)
        out = tmp_path / "out"
        with pytest.raises(SystemExit) as e:
            make_review_tree(src, ["RMDDL", "RMDDR"], out)
        assert "RMDDL" in str(e.value)
        assert not out.exists(), "a refused run must not leave a half-built tree"

    def test_allow_missing_filters_the_queue_to_what_was_copied(self, tmp_path):
        src = _source(tmp_path)
        out = tmp_path / "out"
        make_review_tree(src, ["RMDDL", "RMDDR"], out, allow_missing=True)
        for name in ("_manifest.csv", "_triage.csv"):
            df = pd.read_csv(out / name)
            assert set(df["neuron"]) == {"RMDDR"}, f"{name} advertises a neuron not on disk"
            assert len(df) == 2
        assert (out / "RMDDR").is_dir()
        assert not (out / "RMDDL").exists()

    def test_queue_matches_disk_for_every_neuron_listed(self, tmp_path):
        """The invariant worth guarding directly: every row in the queue has a chain
        directory behind it."""
        src = _source(tmp_path)
        out = tmp_path / "out"
        make_review_tree(src, ["RMDDL", "RMDDR"], out, allow_missing=True)
        df = pd.read_csv(out / "_manifest.csv")
        for _, row in df.iterrows():
            d = out / row["neuron"] / f"chain_{int(row['chain_idx']):02d}"
            assert d.is_dir(), f"queue lists {d} which does not exist"


class TestNormalPath:
    def test_all_present_still_works(self, tmp_path):
        src = _source(tmp_path, present=("RMDDL", "RMDDR"))
        out = tmp_path / "out"
        make_review_tree(src, ["RMDDL", "RMDDR"], out)
        df = pd.read_csv(out / "_manifest.csv")
        assert set(df["neuron"]) == {"RMDDL", "RMDDR"}
        assert len(df) == 4

    def test_rerun_that_now_skips_a_neuron_clears_its_stale_rows(self, tmp_path):
        """Re-running for the same tree must not leave rows from a previous run whose
        neuron has since gone missing, or the queue drifts back out of sync."""
        full = _source(tmp_path / "a", present=("RMDDL", "RMDDR"))
        out = tmp_path / "out"
        make_review_tree(full, ["RMDDL", "RMDDR"], out)
        assert len(pd.read_csv(out / "_manifest.csv")) == 4

        partial = _source(tmp_path / "b", present=("RMDDR",))
        make_review_tree(partial, ["RMDDL", "RMDDR"], out, allow_missing=True)
        df = pd.read_csv(out / "_manifest.csv")
        assert set(df["neuron"]) == {"RMDDR"}, "stale RMDDL rows survived the re-run"

    def test_other_neurons_are_still_preserved_across_runs(self, tmp_path):
        """The merge behaviour the script exists for: adding a neuron must not erase
        one added by an earlier call."""
        a = _source(tmp_path / "a", present=("RMDDR",), listed=("RMDDR",))
        b = _source(tmp_path / "b", present=("RMEV",), listed=("RMEV",))
        out = tmp_path / "out"
        make_review_tree(a, ["RMDDR"], out)
        make_review_tree(b, ["RMEV"], out)
        df = pd.read_csv(out / "_manifest.csv")
        assert set(df["neuron"]) == {"RMDDR", "RMEV"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
