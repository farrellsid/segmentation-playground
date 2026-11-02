"""Regression test for the triage `reasons` column.

`batch._reasons_for_row` returned the bound `str.join` method instead of the
joined string, so every `_triage.csv` `reasons` cell serialized as
`<built-in method join of str object at 0x...>` instead of the tag list.
"""
import pandas as pd

import batch


def test_reasons_for_row_returns_joined_string():
    row = pd.Series({
        "skeleton_contained": False, "area_ratio": 3.0,
        "temporal_iou": 0.1, "pred_iou": 0.9,
    })
    r = batch._reasons_for_row(row)
    assert isinstance(r, str)
    assert r == "noskel area x3.0 tIoU 0.10"


def test_reasons_for_row_empty_when_clean():
    row = pd.Series({
        "skeleton_contained": True, "area_ratio": 1.0,
        "temporal_iou": 0.9, "pred_iou": 0.9,
    })
    assert batch._reasons_for_row(row) == ""
