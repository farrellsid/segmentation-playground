"""Unit tests for pipeline.propagate.propagate_from_verified_masks: seeding MULTIPLE
frames with human-verified masks (not just a single anchor), then propagating.

The fake predictor below mirrors the real SAM2VideoPredictor's documented behaviour
(verified against the installed sam2_video_predictor.py source, see the function's own
docstring): propagate_in_video's default start_frame_idx is the earliest conditioning
frame, and a frame already conditioned is returned verbatim rather than re-predicted."""
import importlib

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

prop = importlib.import_module("pipeline.propagate")


class _FakeVideoPredictor:
    """Mirrors SAM2VideoPredictor's public surface enough to exercise multi-frame
    seeding + bidirectional propagation. Tracks which frames were seeded (as
    conditioning) and, during a sweep, marks every frame it visits so the test can
    assert every frame in the video got covered exactly once per direction."""

    def __init__(self, num_frames: int, hw=(4, 4)):
        self.num_frames = num_frames
        self.hw = hw
        self.added_masks: list[tuple[int, np.ndarray]] = []
        self.visits: list[int] = []

    def init_state(self, video_path, offload_video_to_cpu=True):
        return {"cond_frames": set()}

    def reset_state(self, inference_state):
        inference_state["cond_frames"] = set()

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        inference_state["cond_frames"].add(int(frame_idx))
        self.added_masks.append((int(frame_idx), np.asarray(mask, dtype=bool)))

    def propagate_in_video(self, inference_state, start_frame_idx=None,
                           max_frame_num_to_track=None, reverse=False):
        cond = inference_state["cond_frames"]
        if start_frame_idx is None:
            start_frame_idx = min(cond)
        n = self.num_frames
        if max_frame_num_to_track is None:
            max_frame_num_to_track = n
        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            order = range(start_frame_idx, end - 1, -1) if start_frame_idx > 0 else []
        else:
            end = min(start_frame_idx + max_frame_num_to_track, n - 1)
            order = range(start_frame_idx, end + 1)
        h, w = self.hw
        for f in order:
            self.visits.append(f)
            # a conditioned frame reads back as a filled mask (the "verified" content);
            # a tracked-only frame reads back as an empty mask, so the test can tell
            # which frames the fake actually treated as pre-seeded.
            val = 10.0 if f in cond else -10.0
            mask_logits = torch.full((1, h, w), val, dtype=torch.float32)
            yield f, [1], mask_logits


def test_seeds_every_given_frame_as_conditioning():
    vp = _FakeVideoPredictor(num_frames=10)
    mask = np.ones((4, 4), dtype=bool)
    masks_by_frame = {2: mask, 5: mask, 8: mask}
    prop.propagate_from_verified_masks(vp, "unused", masks_by_frame, obj_id=1)
    seeded = {f for f, _m in vp.added_masks}
    assert seeded == {2, 5, 8}


def test_bidirectional_sweep_covers_the_whole_video_once_per_direction():
    vp = _FakeVideoPredictor(num_frames=10)
    mask = np.ones((4, 4), dtype=bool)
    # earliest verified frame is 2: forward sweep should run 2..9, reverse 2..0
    video_segments, _conf, _iou = prop.propagate_from_verified_masks(
        vp, "unused", {2: mask, 5: mask, 8: mask}, obj_id=1)
    assert sorted(video_segments.keys()) == list(range(10))
    forward_visits = vp.visits[:8]     # 2..9
    reverse_visits = vp.visits[8:]     # 2..0
    assert forward_visits == list(range(2, 10))
    assert reverse_visits == list(range(2, -1, -1))


def test_conditioned_frames_read_back_as_the_seeded_mask():
    vp = _FakeVideoPredictor(num_frames=10)
    mask = np.ones((4, 4), dtype=bool)
    video_segments, _conf, _iou = prop.propagate_from_verified_masks(
        vp, "unused", {2: mask, 5: mask, 8: mask}, obj_id=1)
    for f in (2, 5, 8):
        assert video_segments[f][1].all(), f"frame {f} should read back fully True"
    # a frame between two conditioning frames was only ever tracked, not seeded
    assert not video_segments[3][1].any()


def test_requires_at_least_one_mask():
    vp = _FakeVideoPredictor(num_frames=5)
    with pytest.raises(ValueError):
        prop.propagate_from_verified_masks(vp, "unused", {}, obj_id=1)
