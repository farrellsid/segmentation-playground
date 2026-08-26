"""Padding a neuron's chains onto one canvas, and neuron_video itself.

The reprop report's tour re-crops every frame out of the full _sam frame. A bundle
cannot do that: it only ships each chain's own crop. So frames are padded onto a common
canvas instead, which introduces two failure modes the report version never had.

Chains can sit at different crop_scale, meaning different nanometres per pixel. Padding
those together without normalising puts chains at different magnifications inside one
video with nothing on screen to say so.

And one outlier chain must not size the whole video. That is exactly what went wrong in
the merged reprop render, where a single legacy chain's window dragged every frame to
full-frame size and the masks became invisible.

TestNeuronVideoScaleDirection and TestNeuronVideoMaskSpace cover neuron_video itself,
the two places `common_canvas`/`fit_to_canvas` unit tests cannot reach: crop_scale
carries a physical meaning (nm per pixel), and a tree's mask lives in a different space
from its frame.

Torch-free, data-free.
    py -3 -m pytest tests/test_render_review_video.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
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


# --------------------------------------------------------------------------- #
# neuron_video fixtures and helpers
# --------------------------------------------------------------------------- #

def _write_bundle_chain(root, neuron, chain_idx, *, crop_scale, canvas_hw,
                        square_hw=None, square_origin=(0, 0), z=1500,
                        frame_val=80, with_mask=True):
    """One bundle chain: one frame plus (optionally) one mask, both `canvas_hw`,
    the mask carrying a filled square of `square_hw` at `square_origin` so a test
    can measure how big that square comes out after neuron_video scales it."""
    d = root / neuron / f"chain_{chain_idx:02d}"
    (d / "frames").mkdir(parents=True)
    (d / "masks").mkdir(parents=True)
    h, w = canvas_hw
    frame = np.full((h, w, 3), frame_val, dtype=np.uint8)
    cv2.imwrite(str(d / "frames" / "00000.jpg"), frame)
    if with_mask:
        mask = np.zeros((h, w), dtype=np.uint8)
        if square_hw is not None:
            oy, ox = square_origin
            sh, sw = square_hw
            mask[oy:oy + sh, ox:ox + sw] = 255
        cv2.imwrite(str(d / "masks" / f"mask_{z:04d}.png"), mask)
    state = {"neuron": neuron, "chain_idx": chain_idx,
             "frame_to_z": {"0": z},
             "crop_window": {"crop_scale": crop_scale}}
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def _capture_writer(monkeypatch):
    """Stand in for the video writer so a test can inspect the exact `segments` dict
    neuron_video built, instead of decoding a video back into pixels.

    BOTH writers are patched on purpose. These tests are about scale, mask placement
    and frame coverage, none of which depends on the container, so patching only the
    one that happens to be the current default silently un-tests all of them the day
    that default changes. It did: they broke the moment mp4 became the default.
    """
    captured = {}

    def fake(segments, tmp, out_path, obj_id=None, preview_scale=1, color=None):
        captured["segments"] = {k: dict(v) for k, v in segments.items()}
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"stub")
        return str(out_path)

    monkeypatch.setattr(render_review.video_viz, "to_gif", fake)
    monkeypatch.setattr(render_review.video_viz, "to_mp4", fake)
    return captured


def _mask_width(mask: np.ndarray) -> int:
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return 0
    return int(cols.max() - cols.min() + 1)


class TestNeuronVideoScaleDirection:
    """Finding 1: crop_scale is the downscale applied to the tif crop, so a LARGER
    crop_scale means each stored pixel covers MORE physical area, a coarser chain.
    Two chains holding the same physical structure must show it at the same on-canvas
    size once neuron_video normalises them, regardless of which one is finer."""

    def test_coarser_crop_scale_chain_is_shrunk_to_match(self, tmp_path, monkeypatch):
        root = tmp_path / "bundle"
        neuron = "AIBL"
        # chain_00: crop_scale 2 (finer), the same physical structure spans 16px.
        # chain_01: crop_scale 8 (4x coarser), so the same structure spans 4px.
        # origins/sizes are multiples of 4 so a 0.25 downscale lands on exact pixel
        # boundaries whichever direction the scale bug pushes it, no interpolation
        # ambiguity either way.
        _write_bundle_chain(root, neuron, 0, crop_scale=2, canvas_hw=(40, 40),
                            square_hw=(16, 16), square_origin=(12, 12), z=1500)
        _write_bundle_chain(root, neuron, 1, crop_scale=8, canvas_hw=(40, 40),
                            square_hw=(4, 4), square_origin=(16, 16), z=1600)
        (root / "bundle.json").write_text('{"schema_version": 1, "chains": []}',
                                          encoding="utf-8")
        captured = _capture_writer(monkeypatch)

        out = render_review.neuron_video(root, neuron, tmp_path / "out.gif")
        assert out is not None

        width_by_obj = {}
        for seg in captured["segments"].values():
            for oid, m in seg.items():
                w = _mask_width(m)
                if w:
                    width_by_obj[oid] = w

        assert 1 in width_by_obj and 2 in width_by_obj, (
            "both chains' masks should have survived scaling with nonzero width, "
            f"got {width_by_obj}")
        w0, w1 = width_by_obj[1], width_by_obj[2]
        assert abs(w0 - w1) <= 1, (
            f"chain_00 (crop_scale=2) square is {w0}px wide on canvas, chain_01 "
            f"(crop_scale=8) is {w1}px wide: the same physical structure should "
            f"occupy the same number of canvas pixels in every chain")


class TestNeuronVideoMaskSpace:
    """Finding 2: chain_frames returns a tree chain's frame as the FULL scale-8 _sam
    frame, but a tier-2 chain's mask is saved in its own smaller _pcrop crop space.
    The mask must be placed at its (x0, y0) offset in the full frame, not pasted at
    the origin."""

    def test_tree_tier2_mask_lands_at_its_crop_window_offset(self, tmp_path, monkeypatch):
        root = tmp_path / "tree"
        neuron = "AIBL"
        d = root / neuron / "chain_00"
        (d / "masks").mkdir(parents=True)
        (root / "_manifest.csv").write_text("neuron,chain_idx\nAIBL,0\n", encoding="utf-8")

        # A _pcrop mask, crop_scale 2, filled solid so its whole remapped footprint
        # in _sam space should read True.
        pcrop_mask = np.full((120, 200), 255, dtype=np.uint8)
        cv2.imwrite(str(d / "masks" / "mask_1500.png"), pcrop_mask)

        # size_tif/8 = (50, 30) w,h in _sam px; origin_tif/8 = (10, 10) x,y.
        state = {
            "neuron": neuron, "chain_idx": 0,
            "frame_to_z": {"0": 1500},
            "crop_window": {
                "origin_tif": [80.0, 80.0],
                "size_tif": [400, 240],
                "crop_scale": 2,
                "sam_scale": 8,
            },
        }
        (d / "state.json").write_text(json.dumps(state), encoding="utf-8")

        full_frame = np.full((300, 400, 3), 90, dtype=np.uint8)

        def fake_load(z, *, scale):
            return full_frame.copy(), (2400, 3200)

        monkeypatch.setattr(render_review.pipeline, "load_frame_sam", fake_load)
        captured = _capture_writer(monkeypatch)

        out = render_review.neuron_video(root, neuron, tmp_path / "out.gif")
        assert out is not None

        segs = list(captured["segments"].values())
        assert len(segs) == 1
        mask = segs[0][1]
        assert mask.shape == (300, 400)
        # Inside the crop_window's remapped footprint: rows 10:40, cols 10:60.
        assert mask[15, 15], (
            "the mask should be placed at its crop_window offset (rows 10:40, "
            "cols 10:60 in the full _sam frame), not at the origin")
        # Well outside that footprint, and specifically the origin a paste-at-(0,0)
        # bug would light up.
        assert not mask[0, 0]
        assert not mask[200, 200]


class TestNeuronVideoFrameCoverage:
    """Finding 3: every frame of every chain must reach the output, including a
    frame that has no mask at that z, so a missing mask reads as an empty chain
    rather than making the frame vanish from the video."""

    def test_every_frame_reaches_output_even_with_no_mask(self, tmp_path, monkeypatch):
        root = tmp_path / "bundle"
        neuron = "AIBL"
        d = root / neuron / "chain_00"
        (d / "frames").mkdir(parents=True)
        (d / "masks").mkdir(parents=True)
        for i in range(2):
            img = np.full((20, 24, 3), 10 * (i + 1), dtype=np.uint8)
            cv2.imwrite(str(d / "frames" / f"{i:05d}.jpg"), img)
        # Only frame 0 (z=1500) gets a mask; frame 1 (z=1501) has none.
        cv2.imwrite(str(d / "masks" / "mask_1500.png"),
                   np.full((20, 24), 255, dtype=np.uint8))
        state = {"neuron": neuron, "chain_idx": 0,
                 "frame_to_z": {"0": 1500, "1": 1501},
                 "crop_window": {"crop_scale": 1}}
        (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (root / "bundle.json").write_text('{"schema_version": 1, "chains": []}',
                                          encoding="utf-8")
        captured = _capture_writer(monkeypatch)

        out = render_review.neuron_video(root, neuron, tmp_path / "out.gif")
        assert out is not None

        segments = captured["segments"]
        assert sorted(segments) == [0, 1], "both frames should reach the output"
        has_mask = [bool(seg) for seg in segments.values()]
        assert sum(has_mask) == 1, "exactly one frame has a mask"
        assert not all(has_mask), "the frame with no mask must map to an empty entry"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
