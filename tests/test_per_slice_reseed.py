import pandas as pd
from pipeline import predict, PipelineConfig


def test_centreline_by_z_uses_nodes_and_interpolates_gaps():
    # chain nodes at z=1400 (x_tif=80,y_tif=800) and z=1402 (x_tif=120,y_tif=840); z=1401 is a gap
    chain = {"cell_name": "AVAL", "nodes": ["n0", "n2"]}
    df = pd.DataFrame({
        "node_id": ["n0", "n2", "other"],
        "cell_name": ["AVAL", "AVAL", "AVBR"],
        "z": [1400, 1402, 1401],
        "x_tif": [80.0, 120.0, 9999.0],   # 'other' is a different neuron, must be ignored
        "y_tif": [800.0, 840.0, 9999.0],
    })
    got = predict.centreline_by_z(chain, df)
    assert set(got) == {1400, 1401, 1402}
    assert got[1400] == (80.0, 800.0)
    assert got[1402] == (120.0, 840.0)
    # z=1401 interpolated halfway between the two chain nodes, NOT the foreign node
    assert got[1401] == (100.0, 820.0)


def test_per_slice_reseed_flag_defaults_false():
    assert PipelineConfig().per_slice_reseed is False
