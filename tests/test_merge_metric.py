import pandas as pd
from eval import merge_metric as mm

def test_nodes_by_z_groups_and_scales():
    df = pd.DataFrame({
        "node_id": ["a", "b", "c"],
        "cell_name": ["AVAL", "AVAR", "AVAL"],
        "z": [1400, 1400, 1401],
        "x_tif": [800.0, 1600.0, 240.0],
        "y_tif": [80.0, 160.0, 800.0],
    })
    got = mm.nodes_by_z(df, scale=8)
    assert set(got) == {1400, 1401}
    assert sorted(got[1400]) == [(100.0, 10.0, "AVAL", "a"), (200.0, 20.0, "AVAR", "b")]
    assert got[1401] == [(30.0, 100.0, "AVAL", "c")]
