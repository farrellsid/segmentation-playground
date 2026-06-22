"""Unit tests for pipeline.image_predict forwarding prompts.box_sam to SAM2's
predictor, the data path that lets a GUI-drawn box shape the re-predicted mask.

A fake predictor records the keyword arguments image_predict passes to .predict(),
so we assert: box+points forwards both, box-only forwards point_coords=None, and no
box forwards box=None (the batch path, unchanged). image_predict enters a
torch.inference_mode() block, so the test is skipped when torch is absent (the
CPU-only test policy keeps torch optional).

Run either way:
    py -3 -m pytest tests/test_image_predict_box.py
    py -3 tests/test_image_predict_box.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

pytest.importorskip("torch")     # image_predict uses torch.inference_mode()

import pipeline
from pipeline import Prompts


class _FakePredictor:
    """Records the kwargs of the last .predict() call; returns a fixed 1-mask result."""

    def __init__(self):
        self.last = None

    def set_image(self, image):
        self.image = image

    def predict(self, *, point_coords, point_labels, box, multimask_output):
        self.last = dict(point_coords=point_coords, point_labels=point_labels,
                         box=box, multimask_output=multimask_output)
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 1:3, 1:3] = True
        return masks, np.array([0.9], dtype=float), np.zeros((1, 4, 4), dtype=float)


_IMG = np.zeros((4, 4, 3), dtype=np.uint8)


def test_box_plus_points_forwards_both():
    fp = _FakePredictor()
    pr = Prompts(points_sam=np.array([[2.0, 2.0]]), labels=np.array([1]),
                 box_sam=np.array([0.0, 0.0, 3.0, 3.0]))
    pipeline.image_predict(fp, _IMG, pr)
    assert fp.last["point_coords"] is not None
    assert fp.last["point_labels"] is not None
    assert np.allclose(fp.last["box"], [0, 0, 3, 3])


def test_box_only_forwards_none_points():
    fp = _FakePredictor()
    pr = Prompts(points_sam=np.empty((0, 2)), labels=np.empty((0,), int),
                 box_sam=np.array([0.0, 0.0, 3.0, 3.0]))
    pipeline.image_predict(fp, _IMG, pr)
    assert fp.last["point_coords"] is None
    assert fp.last["point_labels"] is None
    assert np.allclose(fp.last["box"], [0, 0, 3, 3])


def test_no_box_forwards_box_none():
    fp = _FakePredictor()
    pr = Prompts(points_sam=np.array([[2.0, 2.0]]), labels=np.array([1]))   # box_sam defaults None
    pipeline.image_predict(fp, _IMG, pr)
    assert fp.last["box"] is None
    assert fp.last["point_coords"] is not None


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
