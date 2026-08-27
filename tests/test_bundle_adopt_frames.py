"""bundle.adopt_chain_frames + chain_meta.refresh_meta: what a recrop must fix up.

A recrop prepares frames in the machine's frames cache and records an absolute path.
Inside a bundle that breaks two things at once: validate_bundle rejects an absolute
frames_dir, and the bundle stops opening anywhere else. These are the two repairs.

    py -3 -m pytest tests/test_bundle_adopt_frames.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils import bundle, chain_meta


def _view(tmp_path, name="view", n=3):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"{i:05d}.jpg").write_bytes(b"jpg")
    return d


def _chain_dir(tmp_path):
    d = tmp_path / "AIAL" / "chain_00"
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "00000.jpg").write_bytes(b"old")
    return d


class TestIsBundle:
    def test_a_directory_with_a_manifest_is_a_bundle(self, tmp_path):
        (tmp_path / bundle.BUNDLE_MANIFEST).write_text("{}", encoding="utf-8")
        assert bundle.is_bundle(tmp_path) is True

    def test_a_plain_output_tree_is_not(self, tmp_path):
        assert bundle.is_bundle(tmp_path) is False


class TestAdoptChainFrames:
    def test_the_new_frames_land_inside_the_chain_dir(self, tmp_path):
        chain_dir, view = _chain_dir(tmp_path), _view(tmp_path)
        rel = bundle.adopt_chain_frames(chain_dir, view)
        assert rel == "frames"
        assert sorted(p.name for p in (chain_dir / "frames").glob("*.jpg")) == [
            "00000.jpg", "00001.jpg", "00002.jpg"]

    def test_the_old_frames_are_gone_not_mixed_in(self, tmp_path):
        """A leftover frame from the old window would be drawn under a mask from the
        new one, and nothing about the directory would look wrong."""
        chain_dir = _chain_dir(tmp_path)
        for i in range(9):
            (chain_dir / "frames" / f"{i:05d}.jpg").write_bytes(b"old")
        bundle.adopt_chain_frames(chain_dir, _view(tmp_path, n=3))
        assert len(list((chain_dir / "frames").glob("*.jpg"))) == 3

    def test_the_source_is_moved_not_copied(self, tmp_path):
        chain_dir, view = _chain_dir(tmp_path), _view(tmp_path)
        bundle.adopt_chain_frames(chain_dir, view)
        assert not view.exists()

    def test_an_empty_source_refuses(self, tmp_path):
        """A failed prep must not empty the chain's frames and leave the bundle
        unopenable."""
        chain_dir = _chain_dir(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            bundle.adopt_chain_frames(chain_dir, empty)
        assert (chain_dir / "frames" / "00000.jpg").exists()

    def test_adopting_the_frames_already_in_place_is_a_no_op(self, tmp_path):
        chain_dir = _chain_dir(tmp_path)
        rel = bundle.adopt_chain_frames(chain_dir, chain_dir / "frames")
        assert rel == "frames" and (chain_dir / "frames" / "00000.jpg").exists()

    def test_a_delete_that_silently_fails_refuses_instead_of_nesting(self, tmp_path, monkeypatch):
        """I4: shutil.rmtree(dest, ignore_errors=True) swallows a failed delete (e.g.
        napari still holding a handle on the old frames/), and shutil.move onto an
        EXISTING directory moves the source INSIDE it instead of replacing it. That
        leaves chain_dir/frames/<view name>/ on disk: state.frames_dir is recorded as
        "frames", whose 00000.jpg is still the OLD window's frame, and
        validate_bundle passes because jpgs are present.

        Simulates the swallowed failure by making rmtree a no-op, the same effect a
        held file handle has, and checks the old frames are still there UNMOVED
        rather than nested one level down.
        """
        chain_dir = _chain_dir(tmp_path)
        view = _view(tmp_path)
        monkeypatch.setattr(bundle.shutil, "rmtree", lambda *a, **k: None)
        with pytest.raises(RuntimeError):
            bundle.adopt_chain_frames(chain_dir, view)
        assert (chain_dir / "frames" / "00000.jpg").read_bytes() == b"old", (
            "the old frames must be exactly where they were, not nested under a "
            "half-deleted directory")
        assert not (chain_dir / "frames" / view.name).exists(), \
            "the new view must not have been moved inside the old one"
        assert view.exists(), "the new frames must stay put when nothing was moved"


class TestRefreshMeta:
    def _write_meta(self, chain_dir, crop_window):
        meta = chain_meta.build_meta(
            {"neuron": "AIAL", "chain_idx": 0, "crop_window": crop_window,
             "config": {"save_downscale": 8}},
            neuron_id=7, source_tree="reprop_maskseed", chain_dir=chain_dir)
        chain_meta.write_meta(chain_dir, meta)

    def test_a_new_window_reaches_meta_json(self, tmp_path):
        chain_dir = _chain_dir(tmp_path)
        self._write_meta(chain_dir, {"origin_tif": [0, 0], "size_tif": [1024, 1024],
                                     "crop_scale": 2, "sam_scale": 8})
        new_state = {"neuron": "AIAL", "chain_idx": 0,
                     "crop_window": {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                                     "crop_scale": 4, "sam_scale": 8},
                     "config": {"save_downscale": 8}}
        chain_meta.refresh_meta(chain_dir, new_state)
        meta = chain_meta.read_meta(chain_dir)
        assert meta["crop_window"]["size_tif"] == [2048, 2048]
        assert meta["mask_scale"] == 4, "a recrop can change crop_scale, and an exporter "\
                                        "that kept the old one misplaces every pixel"

    def test_identity_and_provenance_survive(self, tmp_path):
        """refresh_meta must not need the registry: the chain's identity is already
        recorded, and inventing a new neuron_id would be worse than failing."""
        chain_dir = _chain_dir(tmp_path)
        self._write_meta(chain_dir, None)
        chain_meta.refresh_meta(chain_dir, {"neuron": "AIAL", "chain_idx": 0,
                                            "crop_window": None,
                                            "config": {"save_downscale": 8}})
        meta = chain_meta.read_meta(chain_dir)
        assert meta["neuron_id"] == 7
        assert meta["provenance"]["source_tree"] == "reprop_maskseed"

    def test_a_chain_with_no_meta_is_left_alone(self, tmp_path):
        """A plain output tree does not always carry meta.json, and there is nothing
        to keep in sync there."""
        chain_dir = _chain_dir(tmp_path)
        assert chain_meta.refresh_meta(chain_dir, {"neuron": "AIAL", "chain_idx": 0,
                                                   "crop_window": None,
                                                   "config": {}}) is None
        assert not (chain_dir / chain_meta.META_FILENAME).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
