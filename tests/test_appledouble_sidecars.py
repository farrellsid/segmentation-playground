"""macOS sidecar files must not break a frames directory.

Reported from a real review session on a Mac with the bundle on a LaCie drive. macOS
writes an AppleDouble sidecar ("._00036.jpg") beside each real file whenever it touches a
filesystem with no resource forks, which is every external drive formatted exFAT or NTFS.

SAM2's own loader globs every *.jpg in the directory and parses the stem as an int
(sam2/utils/misc.py), so the sidecar aborted init_state with

    ValueError: invalid literal for int() with base 10: '._00036'

taking the whole GUI down with it. Our own frame listing in video_viz had the identical
int(stem) call and would have failed on the same file, one step later.

    py -3 -m pytest tests/test_appledouble_sidecars.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline
from sam2_utils import video_viz


def _frames(tmp_path, n=3, sidecars=True):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(n):
        (d / f"{i:05d}.jpg").write_bytes(b"jpg")
        if sidecars:
            # What macOS actually writes: same name, "._" prefix, binary metadata.
            (d / f"._{i:05d}.jpg").write_bytes(b"\x00\x05\x16\x07")
    return d


class TestCleaning:
    def test_the_sidecars_go_and_the_frames_stay(self, tmp_path):
        d = _frames(tmp_path)
        removed = pipeline.clean_frame_sidecars(d)
        assert removed == 3
        assert sorted(p.name for p in d.glob("*.jpg")) == [
            "00000.jpg", "00001.jpg", "00002.jpg"]

    def test_a_clean_directory_is_left_alone(self, tmp_path):
        d = _frames(tmp_path, sidecars=False)
        assert pipeline.clean_frame_sidecars(d) == 0
        assert len(list(d.glob("*.jpg"))) == 3

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        """Called on the way into a propagation, so it must not invent a new failure
        mode ahead of the one the caller is about to report properly."""
        assert pipeline.clean_frame_sidecars(tmp_path / "nope") == 0

    def test_sidecars_of_other_types_go_too(self, tmp_path):
        """macOS writes one per file, not just per jpg. A stray ._state.json in a frames
        directory is the same junk."""
        d = _frames(tmp_path, sidecars=False)
        (d / "._notes.txt").write_bytes(b"\x00")
        (d / ".DS_Store").write_bytes(b"\x00")
        assert pipeline.clean_frame_sidecars(d) == 1, "._ files only, .DS_Store is not ours"
        assert (d / ".DS_Store").exists()

    def test_a_real_file_starting_with_a_dot_is_not_touched(self, tmp_path):
        d = _frames(tmp_path, sidecars=False)
        (d / ".keep").write_bytes(b"")
        pipeline.clean_frame_sidecars(d)
        assert (d / ".keep").exists()


class TestOurOwnListingSurvivesThem:
    def test_frame_indices_ignores_a_sidecar_instead_of_raising(self, tmp_path):
        """video_viz did int(p.stem) over every *.jpg, so it hit the same ValueError one
        step after the propagation would have."""
        d = _frames(tmp_path)
        segments = {0: {1: None}, 1: {1: None}, 2: {1: None}}
        assert video_viz._frame_indices(segments, d) == [0, 1, 2]

    def test_frame_indices_still_drops_frames_with_no_jpeg(self, tmp_path):
        d = _frames(tmp_path, n=2, sidecars=False)
        segments = {0: {1: None}, 1: {1: None}, 5: {1: None}}
        assert video_viz._frame_indices(segments, d) == [0, 1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
