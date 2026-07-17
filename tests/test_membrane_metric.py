import numpy as np
from sam2_utils import membrane as mb


def test_membrane_map_responds_on_dark_ridge():
    patch = np.full((24, 24), 200, dtype=np.uint8)
    patch[:, 11:13] = 20  # a dark vertical ridge (membranes are dark)
    m = mb.membrane_map(patch)
    assert m.shape == (24, 24)
    assert m.dtype == np.float32
    assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0
    assert m[:, 10:14].mean() > m[:, 0:3].mean()  # ridge lights up vs flat area
