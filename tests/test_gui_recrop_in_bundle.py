"""A recrop inside a bundle must leave a bundle, not a broken one.

run_chain prepares frames in the machine's cache and save_state records an absolute
path. In a bundle that fails validate_bundle and stops the bundle opening anywhere
else, so the GUI adopts the frames and rewrites the record. In a plain output tree
none of that applies and behaviour is unchanged.

run_chain is stubbed: this is about the bookkeeping around it, not about SAM2.

    py -3 -m pytest tests/test_gui_recrop_in_bundle.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui
from sam2_utils import bundle, chain_meta


def _bundle_chain(tmp_path, *, manifest=True):
    root = tmp_path / "tree"
    chain_dir = root / "AIAL" / "chain_00"
    (chain_dir / "frames").mkdir(parents=True)
    (chain_dir / "frames" / "00000.jpg").write_bytes(b"old")
    if manifest:
        (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
            {"schema_version": 1, "source_tree": "t", "chains": []}), encoding="utf-8")
    meta = chain_meta.build_meta(
        {"neuron": "AIAL", "chain_idx": 0,
         "crop_window": {"origin_tif": [0, 0], "size_tif": [1024, 1024],
                         "crop_scale": 2, "sam_scale": 8},
         "config": {"save_downscale": 8}},
        neuron_id=7, source_tree="t", chain_dir=chain_dir)
    chain_meta.write_meta(chain_dir, meta)
    return root, chain_dir


def _new_view(tmp_path):
    d = tmp_path / "cache" / "AIAL_chain00_s4"
    d.mkdir(parents=True)
    (d / "00000.jpg").write_bytes(b"new")
    return d


def _harness(monkeypatch, tmp_path, root, chain_dir):
    """A ReviewGUI stand-in carrying only what _recrop_to_window touches."""
    view = _new_view(tmp_path)
    new_window = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                  "crop_scale": 4, "sam_scale": 8}
    saved = {}

    class _State:
        neuron, chain_idx = "AIAL", 0
        frames_dir = str(view)
        crop_window = new_window

    def _run_chain(state, **kwargs):
        state.frames_dir = str(view)
        state.crop_window = new_window

    monkeypatch.setattr(gui.pipeline, "run_chain", _run_chain)
    monkeypatch.setattr(gui.pipeline, "ChainState", lambda **kw: _State())
    monkeypatch.setattr(gui.pipeline, "save_state",
                        lambda state, path: saved.update(
                            path=path, frames_dir=state.frames_dir))
    monkeypatch.setattr(gui.pipeline, "state_to_dict",
                        lambda state: {"neuron": "AIAL", "chain_idx": 0,
                                       "crop_window": state.crop_window,
                                       "config": {"save_downscale": 8}})
    monkeypatch.setattr(gui.pipeline, "raw_em_problem", lambda *a, **k: None)

    # cfg must be a real PipelineConfig: _recrop_to_window calls dataclasses.replace on
    # it, which raises on anything that is not a dataclass.
    self = types.SimpleNamespace(
        ctx=types.SimpleNamespace(output_root=root, image_predictor=None,
                                  video_predictor=None, annotate_df=None,
                                  cfg=gui.pipeline.PipelineConfig(),
                                  ensure_predictors=lambda **kw: None),
        neuron="AIAL", chain_idx=0, chain={}, _cw=None,
        queue=types.SimpleNamespace(set_status=lambda *a, **k: None),
        reviewer="lucinda",
        _close_session=lambda: None,
        open_chain=lambda *a, **k: None,
    )
    # cw_new is printed via cw_new.size_tif, so a bare object() raises before the code
    # under test is reached.
    cw_new = types.SimpleNamespace(size_tif=[2048, 2048])
    return self, saved, view, cw_new


def test_a_recrop_in_a_bundle_records_a_relative_frames_dir(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved["frames_dir"] == "frames"


def test_the_new_frames_are_inside_the_bundle(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert (chain_dir / "frames" / "00000.jpg").read_bytes() == b"new"
    assert not view.exists()


def test_meta_json_follows_the_new_window(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    meta = chain_meta.read_meta(chain_dir)
    assert meta["crop_window"]["size_tif"] == [2048, 2048]
    assert meta["mask_scale"] == 4


def test_an_output_tree_keeps_the_absolute_path(monkeypatch, tmp_path):
    """Outside a bundle the frames cache is exactly where frames belong, and moving
    them into the tree would break every other chain that shares the cache."""
    root, chain_dir = _bundle_chain(tmp_path, manifest=False)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved["frames_dir"] == str(view)
    assert view.exists()


def test_recrop_refuses_without_the_raw_em(monkeypatch, tmp_path, capsys):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    monkeypatch.setattr(gui.pipeline, "raw_em_problem",
                        lambda *a, **k: "raw EM path is not set")
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved == {}, "nothing may be written when the recrop cannot run"
    assert "raw EM" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
