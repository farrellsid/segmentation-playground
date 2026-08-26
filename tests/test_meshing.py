"""Mesh helpers: Taubin smoothing and PLY output.

Taubin rather than Laplacian is the load-bearing choice. Laplacian smoothing shrinks
a surface toward its centroid, and a neurite is mostly thin tube, so repeated
Laplacian passes thin exactly the structure the mesh exists to show. Taubin
alternates a positive and a slightly larger negative step, which cancels most of that
shrinkage.

Torch-free, data-free:
    py -3 -m pytest tests/test_meshing.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils.meshing import taubin_smooth, write_ply


def _tube(n_rings=24, per_ring=16, radius=3.0):
    """A closed-ish triangulated cylinder along z, the shape neurites actually are."""
    ang = np.linspace(0, 2 * np.pi, per_ring, endpoint=False)
    verts, faces = [], []
    for i in range(n_rings):
        for a in ang:
            verts.append([float(i), radius * np.cos(a), radius * np.sin(a)])
    for i in range(n_rings - 1):
        for j in range(per_ring):
            a = i * per_ring + j
            b = i * per_ring + (j + 1) % per_ring
            c = a + per_ring
            d = b + per_ring
            faces.append([a, b, c])
            faces.append([b, d, c])
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


def _mean_radius(verts):
    return float(np.hypot(verts[:, 1], verts[:, 2]).mean())


class TestTaubinSmooth:
    def test_returns_same_shape_and_leaves_faces_alone(self):
        v, f = _tube()
        out = taubin_smooth(v, f, iterations=3)
        assert out.shape == v.shape
        assert out.dtype == np.float64

    def test_preserves_thin_tube_volume_better_than_laplacian(self):
        """The property Taubin was chosen for. lamb only, with mu=0, IS Laplacian."""
        v, f = _tube()
        before = _mean_radius(v)
        lap = taubin_smooth(v, f, iterations=12, lamb=0.5, mu=0.0)
        tau = taubin_smooth(v, f, iterations=12, lamb=0.5, mu=-0.53)
        lap_shrink = before - _mean_radius(lap)
        tau_shrink = before - _mean_radius(tau)
        assert lap_shrink > 0, "Laplacian should shrink a tube; the fixture is wrong"
        assert tau_shrink < lap_shrink / 2, (
            f"Taubin shrank {tau_shrink:.3f} vs Laplacian {lap_shrink:.3f}; "
            f"it is not preserving volume")

    def test_zero_iterations_is_a_no_op(self):
        v, f = _tube()
        assert np.allclose(taubin_smooth(v, f, iterations=0), v)

    def test_a_vertex_with_no_neighbours_does_not_produce_nan(self):
        """An isolated vertex has no neighbour centroid; it must stay put, not divide
        by zero. Marching cubes output can contain these."""
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [50.0, 50.0, 50.0]])
        f = np.array([[0, 1, 2]], dtype=np.int64)
        out = taubin_smooth(v, f, iterations=3)
        assert np.isfinite(out).all()
        assert np.allclose(out[3], v[3])


class TestWritePly:
    def test_roundtrips_counts_and_header(self, tmp_path):
        v, f = _tube(n_rings=4, per_ring=6)
        p = write_ply(tmp_path / "t.ply", v, f)
        text = p.read_text(encoding="utf-8").splitlines()
        assert text[0] == "ply"
        assert f"element vertex {len(v)}" in text
        assert f"element face {len(f)}" in text
        body = text[text.index("end_header") + 1:]
        assert len(body) == len(v) + len(f)
        assert body[len(v)].split()[0] == "3", "faces must be written as triangles"

    def test_refuses_an_empty_mesh(self, tmp_path):
        """An empty mesh means the neuron had no masks. Writing a 0-vertex PLY makes
        that look like a successful export of nothing."""
        with pytest.raises(ValueError):
            write_ply(tmp_path / "e.ply", np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
