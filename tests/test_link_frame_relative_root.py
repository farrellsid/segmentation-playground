r"""_link_frame must produce a READABLE frame even when frames_root is relative.

The failure this guards: `dst.symlink_to(src)` with a relative `src` creates a link
whose target resolves relative to the LINK'S OWN directory, not the cwd. On Narval,
`SAM2_FRAMES_ROOT` was left at its Windows default, so on Linux `F:\ZhenLab\Data`
is a relative directory name, `src` was relative, and every per-chain view frame came
out a dangling symlink. Opening one raises FileNotFoundError, so SAM2's init_state
died on `00000.jpg` and all 15 legacy `_sam` chains of the 2026-08-21 reprop arrays
failed while all 209 tier-2 chains passed (tier-2 writes real JPEGs, it never links).

Torch-free:
    py -3 -m pytest tests/test_link_frame_relative_root.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pipeline.frames import _link_frame


def _make_cache(root: pathlib.Path) -> pathlib.Path:
    cache = root / "frames_cache_s8"
    cache.mkdir(parents=True)
    src = cache / "z1468.jpg"
    src.write_bytes(b"not-really-a-jpeg-but-readable")
    return src


class TestLinkFrame:
    def test_absolute_src_is_readable(self, tmp_path):
        src = _make_cache(tmp_path)
        dst = tmp_path / "chain_views" / "AIBL_chain12_s8" / "00000.jpg"
        dst.parent.mkdir(parents=True)
        _link_frame(src, dst)
        assert dst.read_bytes() == b"not-really-a-jpeg-but-readable"

    def test_relative_src_is_still_readable(self, tmp_path, monkeypatch):
        """The Narval case: frames_root relative, so src is relative too."""
        _make_cache(tmp_path)
        monkeypatch.chdir(tmp_path)
        src = pathlib.Path("frames_cache_s8") / "z1468.jpg"     # relative, as on Narval
        dst = pathlib.Path("chain_views") / "AIBL_chain12_s8" / "00000.jpg"
        dst.parent.mkdir(parents=True)
        _link_frame(src, dst)
        # the whole point: readable, not a link into nowhere
        assert dst.read_bytes() == b"not-really-a-jpeg-but-readable"

    def test_relative_src_does_not_leave_a_dangling_link(self, tmp_path, monkeypatch):
        _make_cache(tmp_path)
        monkeypatch.chdir(tmp_path)
        src = pathlib.Path("frames_cache_s8") / "z1468.jpg"
        dst = pathlib.Path("chain_views") / "deep" / "nested" / "00000.jpg"
        dst.parent.mkdir(parents=True)
        _link_frame(src, dst)
        assert os.path.exists(dst), "dangling symlink: os.path.exists follows the link"

    def test_missing_source_still_raises(self, tmp_path):
        dst = tmp_path / "00000.jpg"
        with pytest.raises(OSError):
            _link_frame(tmp_path / "nope.jpg", dst)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
