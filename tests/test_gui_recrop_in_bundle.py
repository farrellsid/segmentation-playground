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


class _RecropState:
    """What run_chain hands back after a successful recrop, for the config test above."""
    neuron, chain_idx = "AIAL", 0
    frames_dir = None
    crop_window = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                   "crop_scale": 4, "sam_scale": 8}
    frame_to_z = {0: 1500}


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
        # A successful recrop's run_chain fills this during frame prep, after the
        # anchor phase succeeds. The empty-anchor guard (I6) in _recrop_to_window
        # checks it before persisting anything, so every "recrop went fine" harness
        # in this file needs it set; test_an_empty_anchor_recrop_refuses_to_persist
        # below is the one that leaves it None on purpose.
        frame_to_z = {0: 1500}

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
                                  # ReviewContext sets these on its own cfg, so a
                                  # stand-in that leaves them None is not faithful.
                                  cfg=gui.pipeline.PipelineConfig(
                                      output_root=root,
                                      frames_root=tmp_path / "frames_cache"),
                                  ensure_predictors=lambda **kw: None),
        neuron="AIAL", chain_idx=0, chain={}, _cw=None,
        queue=types.SimpleNamespace(set_status=lambda *a, **k: None),
        reviewer="lucinda",
        _state=None,      # a real ReviewGUI always has this; None means
                          # no recorded config, so the base config is used
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


def test_the_recrop_reruns_with_the_chains_own_model_settings(monkeypatch, tmp_path):
    """The chain, not the GUI, decides how it is segmented.

    _recrop_to_window used to build a fresh PipelineConfig from ReviewContext's defaults,
    so a chain produced with backend "sam3" and negative prompts came back as "sam2"
    without them, silently, every time it was recropped. Measured on a real chain in
    AIA_for_lucinda before this was fixed.
    """
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    seen = {}
    monkeypatch.setattr(gui.pipeline, "ChainState",
                        lambda **kw: seen.update(cfg=kw.get("config")) or _RecropState())
    self._state = types.SimpleNamespace(config=gui.pipeline.PipelineConfig(
        backend="sam3", seed_negatives=True, seed_mask=True,
        output_root=pathlib.Path("/scratch/somewhere/else")))
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    cfg = seen["cfg"]
    assert cfg.backend == "sam3", "the recrop silently downgraded the backend"
    assert cfg.seed_negatives is True and cfg.seed_mask is True
    assert cfg.output_root == root, "paths must come from this machine, not the state.json"
    assert cfg.chain_crop is True and cfg.chain_crop_fallback is False,         "the recrop's own overrides must still win"


def _harness_empty_anchor(monkeypatch, tmp_path, root, chain_dir):
    """A ReviewGUI stand-in whose run_chain simulates I6: an empty anchor mask in the
    new window. orchestrator.run_chain sets crop_window during the anchor phase, then
    returns early, BEFORE frame prep, when the anchor mask is empty: frame_to_z stays
    None while crop_window already describes the new (untested) window. masks/ on
    disk is untouched, still the OLD window's masks."""
    new_window = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                  "crop_scale": 4, "sam_scale": 8}
    saved = {}

    class _EmptyAnchorState:
        neuron, chain_idx = "AIAL", 0
        frames_dir = None
        frame_to_z = None
        crop_window = new_window

    def _run_chain(state, **kwargs):
        pass  # nothing to do: the stand-in state already models run_chain's early return

    monkeypatch.setattr(gui.pipeline, "run_chain", _run_chain)
    monkeypatch.setattr(gui.pipeline, "ChainState", lambda **kw: _EmptyAnchorState())
    monkeypatch.setattr(gui.pipeline, "save_state",
                        lambda state, path: saved.update(called=True))
    monkeypatch.setattr(gui.pipeline, "state_to_dict",
                        lambda state: (_ for _ in ()).throw(
                            AssertionError("state_to_dict must not run when nothing is saved")))
    monkeypatch.setattr(gui.pipeline, "raw_em_problem", lambda *a, **k: None)

    def _must_not_reopen(*a, **k):
        raise AssertionError("open_chain must not run when the recrop wrote nothing")

    self = types.SimpleNamespace(
        ctx=types.SimpleNamespace(output_root=root, image_predictor=None,
                                  video_predictor=None, annotate_df=None,
                                  cfg=gui.pipeline.PipelineConfig(),
                                  ensure_predictors=lambda **kw: None),
        neuron="AIAL", chain_idx=0, chain={}, _cw=None,
        queue=types.SimpleNamespace(
            set_status=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("queue status must not change when nothing was written"))),
        reviewer="lucinda",
        _state=None,      # a real ReviewGUI always has this; None means
                          # no recorded config, so the base config is used
        _close_session=lambda: None,
        open_chain=_must_not_reopen,
    )
    cw_new = types.SimpleNamespace(size_tif=[2048, 2048])
    return self, saved, cw_new


def test_an_empty_anchor_recrop_refuses_to_persist(monkeypatch, tmp_path, capsys):
    """I6: an empty anchor mask in the recropped window must not overwrite the
    chain's only recorded frame_to_z with None, and must not touch meta.json,
    disk masks, the review queue, or reopen the chain. The chain stays exactly as
    it was, with a message naming what happened.
    """
    root, chain_dir = _bundle_chain(tmp_path)
    meta_before = (chain_dir / "meta.json").read_bytes()
    self, saved, cw_new = _harness_empty_anchor(monkeypatch, tmp_path, root, chain_dir)

    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")

    assert saved == {}, "an empty-anchor recrop must not write state.json"
    assert (chain_dir / "meta.json").read_bytes() == meta_before, \
        "meta.json must not be touched when nothing was written"
    assert not (chain_dir / "state.json").exists(), \
        "no state.json existed before the recrop; the refusal must not create one"
    out = capsys.readouterr().out
    assert "no anchor mask" in out.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
