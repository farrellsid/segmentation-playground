"""A recropped chain's geometry must come home with its masks.

import_bundle merges masks and qc.csv and deliberately never copies state.json, which
carries a bundle-relative frames_dir. A recrop, though, rewrites the chain's GEOMETRY,
so masks-only means the master tree keeps describing the old window and every consumer
places the new pixels at the old offsets, silently.

    py -3 -m pytest tests/test_import_bundle_geometry.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils import bundle

OLD_WINDOW = {"origin_tif": [0, 0], "size_tif": [1024, 1024],
              "crop_scale": 2, "sam_scale": 8}
NEW_WINDOW = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
              "crop_scale": 4, "sam_scale": 8}


def _state(window, *, frames_dir, frame_to_z=None, n_frames=3, anchor=1, extra=None):
    state = {"neuron": "AIAL", "chain_idx": 0, "crop_window": window,
             "frames_dir": frames_dir, "n_frames": n_frames,
             "anchor_frame_idx": anchor,
             "frame_to_z": frame_to_z or {"0": 1500, "1": 1501, "2": 1502},
             "config": {"output_root": "/master", "frames_root": "/master/frames"},
             "qc_summary": {"flagged": 2}}
    if extra:
        state.update(extra)
    return state


class TestGeometryOf:
    def test_it_reads_exactly_the_recrop_sensitive_fields(self):
        assert set(bundle.GEOMETRY_FIELDS) == {
            "crop_window", "frame_to_z", "n_frames", "anchor_frame_idx"}

    def test_two_states_with_the_same_window_compare_equal(self):
        a = _state(OLD_WINDOW, frames_dir="/a")
        b = _state(OLD_WINDOW, frames_dir="/b")
        assert bundle.geometry_of(a) == bundle.geometry_of(b), \
            "frames_dir differs on every machine and must not read as a recrop"

    def test_a_new_window_compares_different(self):
        assert bundle.geometry_of(_state(OLD_WINDOW, frames_dir="/a")) != \
            bundle.geometry_of(_state(NEW_WINDOW, frames_dir="/a"))


class TestMergeGeometry:
    def test_the_new_window_replaces_the_old(self):
        merged = bundle.merge_geometry(_state(OLD_WINDOW, frames_dir="/master/f"),
                                       _state(NEW_WINDOW, frames_dir="frames"))
        assert merged["crop_window"] == NEW_WINDOW

    def test_the_config_block_never_crosses_machines(self):
        """state.json embeds the whole PipelineConfig, including output_root and
        frames_root. Copying it back writes the reviewer's Mac paths into the master
        tree."""
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        incoming = _state(NEW_WINDOW, frames_dir="frames")
        incoming["config"] = {"output_root": "/Users/lucinda/bundle",
                              "frames_root": "/Users/lucinda/frames"}
        merged = bundle.merge_geometry(master, incoming)
        assert merged["config"] == master["config"]

    def test_the_stale_frames_dir_is_cleared(self):
        """The _pcrop view dir is namespaced by neuron, chain and crop scale, so a new
        window at the same scale REUSES the directory name. The master's recorded dir
        still exists and still holds the old window's frames."""
        merged = bundle.merge_geometry(_state(OLD_WINDOW, frames_dir="/master/f"),
                                       _state(NEW_WINDOW, frames_dir="frames"))
        assert merged["frames_dir"] is None

    def test_reviewer_owned_summaries_are_not_touched_here(self):
        """qc.csv is copied as a file by the existing REVIEWER_OWNED path; the state's
        qc_summary belongs to the master's own run and is not part of this merge."""
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        incoming = _state(NEW_WINDOW, frames_dir="frames",
                          extra={"qc_summary": {"flagged": 99}})
        assert bundle.merge_geometry(master, incoming)["qc_summary"] == {"flagged": 2}

    def test_the_master_state_is_not_mutated_in_place(self):
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        bundle.merge_geometry(master, _state(NEW_WINDOW, frames_dir="frames"))
        assert master["crop_window"] == OLD_WINDOW

    def test_a_sam_chain_with_no_window_merges_without_error(self):
        merged = bundle.merge_geometry(_state(None, frames_dir="/master/f"),
                                       _state(None, frames_dir="frames"))
        assert merged["crop_window"] is None


class TestImportCarriesItHome:
    def _tree(self, tmp_path, name, window, *, frames_dir, source_tree="t"):
        from sam2_utils import chain_meta
        root = tmp_path / name
        chain_dir = root / "AIAL" / "chain_00"
        (chain_dir / "masks").mkdir(parents=True)
        (chain_dir / "masks" / "mask_1500.png").write_bytes(b"png" + name.encode())
        (chain_dir / "qc.csv").write_text("frame,flag\n0,0\n", encoding="utf-8")
        (chain_dir / "state.json").write_text(
            json.dumps(_state(window, frames_dir=frames_dir)), encoding="utf-8")
        chain_meta.write_meta(chain_dir, chain_meta.build_meta(
            {"neuron": "AIAL", "chain_idx": 0, "crop_window": window,
             "config": {"save_downscale": 8}},
            neuron_id=7, source_tree=source_tree, chain_dir=chain_dir))
        return root, chain_dir

    def _bundle(self, tmp_path, window):
        root, chain_dir = self._tree(tmp_path, "bundle", window, frames_dir="frames")
        (chain_dir / "frames").mkdir()
        (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
            {"schema_version": 1, "source_tree": "master",
             "chains": [{"cell_name": "AIAL", "chain_idx": 0,
                         "chain_dir": "AIAL/chain_00"}]}), encoding="utf-8")
        for rel in bundle.BUNDLE_DATA_FILES:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("[]" if rel.endswith(".json") else "node_id\n", encoding="utf-8")
        # No root ledgers here on purpose: _read_ledger returns an empty frame for a
        # MISSING ledger, which is the case a bundle with no verdicts recorded actually
        # produces. (A zero-byte ledger is a different thing and does raise, but that is
        # not what this test is about.)
        return root

    def test_a_recropped_chain_updates_the_masters_window(self, tmp_path):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["crop_window"] == NEW_WINDOW
        assert state["frames_dir"] is None
        assert state["config"]["output_root"] == "/master", \
            "the reviewer's config must never land in the master tree"

    def test_the_masters_meta_json_is_replaced(self, tmp_path):
        import import_bundle as ib
        from sam2_utils import chain_meta
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        assert chain_meta.read_meta(master_chain)["crop_window"] == NEW_WINDOW

    def test_an_unrecropped_chain_leaves_the_state_alone(self, tmp_path):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, OLD_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["frames_dir"] == "/master/f", \
            "a plain mask edit must not invalidate the master's frames"

    def test_a_dry_run_reports_the_recrop_and_writes_nothing(self, tmp_path, capsys):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, dry_run=True, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["crop_window"] == OLD_WINDOW
        out = capsys.readouterr().out
        assert "recrop" in out.lower() and "2048" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
