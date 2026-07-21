import numpy as np
import pytest
torch = pytest.importorskip("torch")   # smoke only runs where torch is present
import run_perframe


def test_segment_frame_prompt_shapes(monkeypatch):
    # a fake image predictor: returns one candidate mask per set_image/predict, torch-free
    class FakePred:
        def set_image(self, img): self._hw = img.shape[:2]
        def predict(self, **kw):
            h, w = self._hw
            m = np.zeros((1, h, w), bool); m[0, :h // 2, :w // 2] = True
            return m, np.array([0.9]), np.zeros((1, 256, 256), np.float32)
        def reset_predictor(self): pass
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    cell_masks, lab, score = run_perframe.segment_frame_prompt(
        FakePred(), frame, node_index, mem, negatives=True, selection="pred_iou",
        resolver="argmax", cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score
