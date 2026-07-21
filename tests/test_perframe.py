import numpy as np
import pandas as pd
from sam2_utils import perframe as pf


def test_nodes_in_frame_filters_z_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"], "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401], "x_tif": [800.0, 1600.0, 240.0], "y_tif": [80.0, 160.0, 800.0],
    })
    got = pf.nodes_in_frame(df, 1400, scale=8)
    assert sorted(got) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert pf.nodes_in_frame(df, 1401, scale=8) == [(30.0, 100.0, "AVAL", "c")]
