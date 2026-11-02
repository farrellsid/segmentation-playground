"""Unit tests for the merged-render tour geometry in experiments/report_assets.py.

The bug being guarded: the whole-neuron "merged" render used ONE window covering
every reprop'd chain across the arbor. Measured on real data, only ~1 chain is
present at any given z while the masks travel 500-700px down the stack, so that
window was nearly the whole worm cross-section and the mask covered a median of
0.11-0.27% of the frame (AIAR: 84 of 176 frames under 0.1%). The tour renders each
chain in its own window instead, padded to one common size so the frames still form
a single animation.

Also guards the `full_hw` misuse this pass fixed: pipeline.load_frame_sam returns the
PRE-downscale shape alongside a downscaled image, so passing it as the _sam canvas
size was 64x too big.

Torch-free (numpy only):
    py -3 -m pytest tests/test_report_tour_windows.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.report_assets import (_compute_window, center_window,
                                       common_window_size)


FRAME = (1152, 1154)  # the real _sam frame, (H, W)


class TestCommonWindowSize:
    def test_takes_the_max_of_each_axis_independently(self):
        # widest and tallest come from different chains, both must survive
        windows = {0: (0, 0, 100, 50), 1: (0, 0, 40, 300)}
        assert common_window_size(windows, FRAME) == (100, 300)

    def test_single_window_is_its_own_size(self):
        assert common_window_size({7: (10, 20, 110, 220)}, FRAME) == (100, 200)

    def test_capped_at_the_frame_so_a_window_can_always_be_placed(self):
        windows = {0: (0, 0, 5000, 9000)}
        assert common_window_size(windows, FRAME) == (1154, 1152)

    def test_empty_is_refused_rather_than_returning_a_degenerate_size(self):
        with pytest.raises(ValueError):
            common_window_size({}, FRAME)


class TestCenterWindow:
    def test_expands_about_its_own_centre(self):
        # 20x20 box centred at (110, 110) grown to 100x100 keeps that centre
        out = center_window((100, 100, 120, 120), (100, 100), FRAME)
        assert out == (60, 60, 160, 160)

    def test_output_is_exactly_the_requested_size(self):
        for win in [(0, 0, 10, 10), (500, 400, 530, 460), (1100, 1100, 1150, 1148)]:
            x0, y0, x1, y1 = center_window(win, (300, 260), FRAME)
            assert (x1 - x0, y1 - y0) == (300, 260)

    def test_shifts_instead_of_clipping_at_the_top_left_edge(self):
        # a mask hard against the origin must still get a full-size window,
        # slid inward, NOT a truncated one (a short crop is what made _overlay
        # resize the mask onto a smaller EM and misregister it)
        assert center_window((0, 0, 20, 20), (300, 260), FRAME) == (0, 0, 300, 260)

    def test_shifts_instead_of_clipping_at_the_bottom_right_edge(self):
        H, W = FRAME
        x0, y0, x1, y1 = center_window((W - 10, H - 10, W, H), (300, 260), FRAME)
        assert (x1, y1) == (W, H)
        assert (x1 - x0, y1 - y0) == (300, 260)

    def test_a_window_larger_than_the_frame_is_refused(self):
        with pytest.raises(ValueError):
            center_window((0, 0, 10, 10), (2000, 260), FRAME)

    def test_every_real_chain_window_fits_the_common_size(self):
        # the invariant the tour depends on: fit each chain's own window into the
        # common size and every result is in-frame, exactly-sized, and still
        # contains the window it came from
        windows = {
            0: (100, 200, 362, 473), 1: (600, 50, 986, 375),
            2: (0, 900, 262, 1152), 3: (900, 0, 1154, 273),
        }
        size = common_window_size(windows, FRAME)
        for ci, win in windows.items():
            x0, y0, x1, y1 = center_window(win, size, FRAME)
            assert (x1 - x0, y1 - y0) == size, ci
            assert 0 <= x0 and x1 <= FRAME[1], ci
            assert 0 <= y0 and y1 <= FRAME[0], ci
            assert x0 <= win[0] and win[2] <= x1, ci
            assert y0 <= win[1] and win[3] <= y1, ci




class TestOverlayPalette:
    """Guards the palette these renders draw with. Every consumer alpha-blends onto
    greyscale EM, so an achromatic entry is invisible by construction: AIAL chain_16
    (obj_id 17, 17 % 10 == 7 -> tab10's grey) vanished for all 22 of its frames."""

    def test_no_achromatic_entry(self):
        from sam2_utils import video_viz
        for c in video_viz._PALETTE:
            assert c.max() - c.min() > 0.15, f"{c} is too close to grey for EM overlay"

    def test_every_object_id_gets_a_chromatic_colour(self):
        from sam2_utils import video_viz
        for oid in range(1, 60):
            c = video_viz._color_for(oid)
            assert c.max() - c.min() > 0.15, f"obj {oid} -> {c}"

    def test_palette_is_not_empty_after_filtering(self):
        from sam2_utils import video_viz
        assert len(video_viz._PALETTE) >= 8


class TestNeuronGifTripleCli:
    """A VARIANT=mask cluster run writes no box-seed tree at all (mask-seed won on
    overfill on all four AIA/AIY sides, so running box too spends half the GPU time
    producing the loser). The renderer has to cope with a missing box tree instead of
    aborting every output for the chain, which is what the old unconditional
    render(box_tree, ...) did by way of build_view's SystemExit."""

    def _manifest(self, tmp_path):
        m = tmp_path / "chains.csv"
        m.write_text("neuron,chain_idx,anchor_z\nAIZL,5,1508\nAIZL,6,1515\n")
        return m

    def _argv(self, tmp_path, with_box):
        argv = ["neuron-gif-triple", "--before", "B", "--mask-tree", "M",
                "--manifest", str(self._manifest(tmp_path)), "--neuron", "AIZL",
                "--out-before", "b.gif", "--out-mask", "m.gif"]
        if with_box:
            argv += ["--box-tree", "X", "--out-box", "x.gif"]
        return argv

    def test_runs_without_a_box_tree(self, tmp_path, monkeypatch):
        from experiments import report_assets
        seen = {}
        monkeypatch.setattr(report_assets, "render_reprop_tour",
                            lambda trees, neuron, cis, outs, **kw: seen.update(
                                trees=trees, neuron=neuron, cis=cis, outs=outs))
        report_assets.main(self._argv(tmp_path, with_box=False))
        assert set(seen["trees"]) == {"before", "mask"}
        assert set(seen["outs"]) == {"before", "mask"}
        assert seen["cis"] == [5, 6]

    def test_box_tree_is_still_passed_through_when_given(self, tmp_path, monkeypatch):
        from experiments import report_assets
        seen = {}
        monkeypatch.setattr(report_assets, "render_reprop_tour",
                            lambda trees, neuron, cis, outs, **kw: seen.update(trees=trees))
        report_assets.main(self._argv(tmp_path, with_box=True))
        assert set(seen["trees"]) == {"before", "mask", "box"}

    def test_box_tree_without_an_output_path_is_refused(self, tmp_path):
        from experiments import report_assets
        argv = self._argv(tmp_path, with_box=False) + ["--box-tree", "X"]
        with pytest.raises(SystemExit):
            report_assets.main(argv)


class TestComputeWindowUsesContent:
    """_compute_window must size from where the mask's True pixels ARE, not from the
    array holding them. chain_masks_in_sam hands back a crop-sized array for a tier-2
    `_pcrop` chain and a whole-frame array at x0=y0=0 for a legacy `_sam` one, so
    sizing by array bounds makes the window depend on which space a chain happens to
    live in. Real case: AIZL chain_26 is a 126x95 blob stored on a 1154x1152 array,
    and it dragged an entire 44-chain tour to full-frame."""

    def _win(self, mask, x0, y0):
        return _compute_window({0: {1000: (mask, x0, y0)}}, FRAME)

    def test_legacy_full_frame_array_sizes_from_the_blob(self):
        mask = np.zeros(FRAME, dtype=bool)
        mask[794:889, 564:690] = True          # AIZL chain_26's real content bbox
        x0, y0, x1, y1 = self._win(mask, 0, 0)
        assert (x1 - x0) < 300 and (y1 - y0) < 300, (x1 - x0, y1 - y0)
        assert x0 <= 564 and x1 >= 690 and y0 <= 794 and y1 >= 889

    def test_same_blob_gives_the_same_window_in_either_space(self):
        # a crop-space chain and a legacy chain holding the identical blob must be
        # framed identically; before the fix these differed by the whole frame
        full = np.zeros(FRAME, dtype=bool)
        full[300:340, 200:260] = True
        crop = np.zeros((120, 140), dtype=bool)
        crop[20:60, 30:90] = True              # same absolute pixels, placed at 170,280
        assert self._win(full, 0, 0) == self._win(crop, 170, 280)

    def test_padding_scales_with_the_blob_not_the_array(self):
        small = np.zeros(FRAME, dtype=bool)
        small[500:510, 500:510] = True
        x0, y0, x1, y1 = self._win(small, 0, 0)
        assert (x1 - x0) <= 10 + 2 * 15 + 5    # 10px blob + min_pad either side

    def test_all_empty_masks_still_refused(self):
        with pytest.raises(SystemExit):
            self._win(np.zeros(FRAME, dtype=bool), 0, 0)


class TestReproWindowUnionsVariants:
    """reprop_window must union the variants as BOXES. The old code merged their mask
    DICTS (`{**mask_masks, **box_masks}`), which is keyed by z, so box-seed's mask
    replaced mask-seed's at every z they both cover, i.e. all of them. The window then
    came from box-seed alone while the comment claimed a union, and a chain where the
    two variants diverge spatially got cropped to one of them.

    Data-free: sam_hw is stubbed so nothing here needs the EM store on F:."""

    @pytest.fixture(autouse=True)
    def _no_em_store(self, monkeypatch):
        from experiments import report_assets
        monkeypatch.setattr(report_assets, "sam_hw", lambda z: FRAME)

    def _tree(self, tmp_path, name, box):
        """A minimal on-disk chain: chain_masks_in_sam reads masks/ plus state.json."""
        import json

        import cv2
        d = tmp_path / name / "N" / "chain_00"
        (d / "masks").mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"neuron": "N", "chain_idx": 0}))
        x0, y0, x1, y1 = box
        for z in (1000, 1001):
            m = np.zeros(FRAME, dtype=np.uint8)
            m[y0:y1, x0:x1] = 255
            cv2.imwrite(str(d / "masks" / f"mask_{z:04d}.png"), m)
        return tmp_path / name

    def test_window_covers_both_variants(self, tmp_path):
        from experiments.report_assets import reprop_window
        # the two variants sit in disjoint places; the window must contain both
        mask_tree = self._tree(tmp_path, "mask", (100, 100, 160, 160))
        box_tree = self._tree(tmp_path, "box", (700, 700, 760, 760))
        w = reprop_window({"before": tmp_path / "nope", "mask": mask_tree,
                           "box": box_tree}, "N", [0])
        assert w[0] <= 100 and w[1] <= 100, w
        assert w[2] >= 760 and w[3] >= 760, w

    def test_works_with_only_the_mask_variant(self, tmp_path):
        from experiments.report_assets import reprop_window
        mask_tree = self._tree(tmp_path, "mask", (100, 100, 160, 160))
        w = reprop_window({"before": tmp_path / "nope", "mask": mask_tree}, "N", [0])
        assert w[0] <= 100 and w[2] >= 160

    def test_before_tree_never_sets_the_window(self, tmp_path):
        # the before tail is the thing under test; letting it size the frame would
        # hide the overfill the comparison exists to show
        from experiments.report_assets import reprop_window
        mask_tree = self._tree(tmp_path, "mask", (100, 100, 160, 160))
        before = self._tree(tmp_path, "before", (0, 0, 1100, 1100))
        w = reprop_window({"before": before, "mask": mask_tree}, "N", [0])
        assert (w[2] - w[0]) < 200 and (w[3] - w[1]) < 200, w

    def test_no_reprop_masks_anywhere_is_refused(self, tmp_path):
        from experiments.report_assets import reprop_window
        with pytest.raises(SystemExit):
            reprop_window({"before": tmp_path, "mask": tmp_path / "nope"}, "N", [0])

    def test_render_sides_refuses_a_tree_with_no_output_path(self, tmp_path):
        from experiments.report_assets import render_reprop_sides
        with pytest.raises(ValueError):
            render_reprop_sides({"before": tmp_path, "mask": tmp_path}, "N", [0],
                                {"before": tmp_path / "b.gif"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
