import numpy as np
import run_perframe


class FakeAMG:
    """Stand-in for SAM2AutomaticMaskGenerator.generate: returns AMG-style dicts."""
    def __init__(self, masks): self._masks = masks
    def generate(self, image):
        return [{"segmentation": m, "area": int(m.sum()), "predicted_iou": 0.9,
                 "stability_score": 0.95} for m in self._masks]


def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_segment_frame_amg_labels_and_keeps_competitors():
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_index = [(10, 10, "AVAL", "a"), (30, 30, "AVAR", "b")]
    mem = np.zeros((40, 40), np.float32)
    junk = _disk(20, 4, 3)
    amg = FakeAMG([_disk(10, 10, 5), _disk(30, 30, 5), junk])  # 2 cells + 1 junk
    cell_masks, lab, score = run_perframe.segment_frame_amg(
        amg, frame, node_index, mem, match="metric", resolver="argmax",
        cfg=run_perframe.PerframeCfg(scale=8))
    assert set(cell_masks) == {"AVAL", "AVAR"}
    assert lab.shape == (40, 40)
    assert "own_coverage" in score
    # both cells actually carry the pixels of their matched AMG disk
    assert cell_masks["AVAL"][10, 10]
    assert cell_masks["AVAR"][30, 30]
    # the junk competitor is resolved into the label map (it can push back on the
    # cell masks) but never survives as a labelled cell: its footprint is background
    # in the cell-only label map once competitor labels are dropped.
    assert not lab[junk].any()
