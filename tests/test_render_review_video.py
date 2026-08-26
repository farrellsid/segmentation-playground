"""Padding a neuron's chains onto one canvas.

The reprop report's tour re-crops every frame out of the full _sam frame. A bundle
cannot do that: it only ships each chain's own crop. So frames are padded onto a common
canvas instead, which introduces two failure modes the report version never had.

Chains can sit at different crop_scale, meaning different nanometres per pixel. Padding
those together without normalising puts chains at different magnifications inside one
video with nothing on screen to say so.

And one outlier chain must not size the whole video. That is exactly what went wrong in
the merged reprop render, where a single legacy chain's window dragged every frame to
full-frame size and the masks became invisible.

Torch-free, data-free.
    py -3 -m pytest tests/test_render_review_video.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


class TestCommonCanvas:
    def test_takes_the_max_of_each_axis(self):
        assert render_review.common_canvas([(100, 50), (40, 300)]) == (100, 300)

    def test_caps_so_one_outlier_cannot_size_the_video(self):
        """The merged reprop render's exact failure, in miniature."""
        sizes = [(120, 130), (118, 126), (1152, 1154)]
        h, w = render_review.common_canvas(sizes, cap=400)
        assert h <= 400 and w <= 400

    def test_empty_is_refused(self):
        with pytest.raises(ValueError):
            render_review.common_canvas([])


class TestFitToCanvas:
    def test_pads_a_small_frame_without_stretching_it(self):
        img = np.full((10, 12, 3), 200, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 50))
        assert out.shape == (40, 50, 3)
        assert (out == 200).sum() == 10 * 12 * 3, "content should be padded, not scaled"

    def test_scales_before_padding_so_nm_per_px_matches(self):
        """A chain at half the nm/px of its neighbours must be halved before padding,
        or the video shows two magnifications with nothing to signal it."""
        img = np.full((20, 20, 3), 150, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 40), scale=0.5)
        assert out.shape == (40, 40, 3)
        assert (out == 150).sum() < 20 * 20 * 3, "scale=0.5 should shrink the content"

    def test_a_frame_larger_than_the_canvas_is_scaled_down_to_fit(self):
        img = np.full((80, 90, 3), 100, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 40))
        assert out.shape == (40, 40, 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
