import importlib

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

prop = importlib.import_module("pipeline.propagate")
from pipeline.state import Prompts


class _FakeVideoPredictor:
    """Minimal fake covering both seeding paths propagate_directional needs:
    add_new_points_or_box (point/box seed) and propagate_in_video (single-direction
    sweep). Mirrors the real predictor's documented range behaviour: reverse=False
    covers [start, num_frames-1], reverse=True covers [0, start]."""

    def __init__(self, num_frames: int, hw=(4, 4)):
        self.num_frames = num_frames
        self.hw = hw
        self.seeded_at: list[int] = []
        self.visits: list[int] = []

    def init_state(self, video_path, offload_video_to_cpu=True):
        return {"cond_frame": None}

    def reset_state(self, inference_state):
        inference_state["cond_frame"] = None

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None,
                              points=None, labels=None, clear_old_points=False):
        inference_state["cond_frame"] = int(frame_idx)
        self.seeded_at.append(int(frame_idx))

    def propagate_in_video(self, inference_state, start_frame_idx=None,
                           max_frame_num_to_track=None, reverse=False):
        start = inference_state["cond_frame"] if start_frame_idx is None else start_frame_idx
        n = self.num_frames
        if max_frame_num_to_track is None:
            max_frame_num_to_track = n
        if reverse:
            end = max(start - max_frame_num_to_track, 0)
            order = range(start, end - 1, -1) if start > 0 else []
        else:
            end = min(start + max_frame_num_to_track, n - 1)
            order = range(start, end + 1)
        h, w = self.hw
        for f in order:
            self.visits.append(f)
            mask_logits = torch.full((1, h, w), 10.0, dtype=torch.float32)
            yield f, [1], mask_logits


def _prompts():
    return Prompts(points_sam=np.asarray([[2.0, 2.0]]), labels=np.asarray([1]))


def test_forward_from_start_covers_start_to_end():
    vp = _FakeVideoPredictor(num_frames=10)
    video_segments, _conf, _iou = prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=0, obj_id=1, reverse=False)
    assert vp.seeded_at == [0]
    assert sorted(video_segments.keys()) == list(range(10))
    assert vp.visits == list(range(0, 10))


def test_backward_from_end_covers_end_to_start():
    vp = _FakeVideoPredictor(num_frames=10)
    video_segments, _conf, _iou = prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=9, obj_id=1, reverse=True)
    assert vp.seeded_at == [9]
    assert sorted(video_segments.keys()) == list(range(10))
    assert vp.visits == list(range(9, -1, -1))


def test_only_seeds_once_not_bidirectional():
    vp = _FakeVideoPredictor(num_frames=10)
    prop.propagate_directional(
        vp, "unused", _prompts(), seed_frame_idx=0, obj_id=1, reverse=False)
    assert len(vp.seeded_at) == 1
