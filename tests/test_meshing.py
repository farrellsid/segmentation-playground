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
        assert "property list uchar int vertex_indices" in text
        body = text[text.index("end_header") + 1:]
        assert len(body) == len(v) + len(f)
        assert body[len(v)].split()[0] == "3", "faces must be written as triangles"

    def test_refuses_an_empty_mesh(self, tmp_path):
        """An empty mesh means the neuron had no masks. Writing a 0-vertex PLY makes
        that look like a successful export of nothing."""
        with pytest.raises(ValueError):
            write_ply(tmp_path / "e.ply", np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))


from sam2_utils.meshing import PRESETS, volume_to_mesh


def _blob(shape=(12, 20, 20)):
    """A solid box inside an empty volume, so the surface area is predictable."""
    vol = np.zeros(shape, dtype=np.uint8)
    vol[3:9, 6:14, 6:14] = 1
    return vol


class TestVolumeToMesh:
    def test_vertices_come_out_in_nanometres_in_xyz_order(self):
        """spacing is (z, y, x) because that is the array axis order, but Blender wants
        x, y, z. Getting this backwards produces a mesh that looks plausible and is
        wrong by 2.5x in one direction."""
        vol = _blob()
        verts, faces = volume_to_mesh(vol, spacing=(50.0, 128.0, 128.0))
        assert verts.shape[1] == 3 and len(faces) > 0
        # the box spans 6..14 in x and y (8 voxels at 128nm) and 3..9 in z (6 at 50nm)
        span = verts.max(axis=0) - verts.min(axis=0)
        assert 800 < span[0] < 1200, f"x span {span[0]:.0f}nm, expected about 1024"
        assert 800 < span[1] < 1200, f"y span {span[1]:.0f}nm, expected about 1024"
        assert 200 < span[2] < 400, f"z span {span[2]:.0f}nm, expected about 300"

    def test_presets_differ_in_triangle_count(self):
        vol = _blob()
        counts = {}
        for name in ("faithful", "balanced", "smooth"):
            _v, f = volume_to_mesh(vol, spacing=(50.0, 128.0, 128.0), preset=name)
            counts[name] = len(f)
        assert counts["smooth"] < counts["faithful"], (
            f"smooth ({counts['smooth']}) should coarsen via step_size vs "
            f"faithful ({counts['faithful']})")

    def test_faithful_is_the_default(self):
        vol = _blob()
        a = volume_to_mesh(vol, spacing=(50.0, 128.0, 128.0))
        b = volume_to_mesh(vol, spacing=(50.0, 128.0, 128.0), preset="faithful")
        assert len(a[1]) == len(b[1])

    def test_unknown_preset_is_refused_by_name(self):
        with pytest.raises(ValueError) as e:
            volume_to_mesh(_blob(), spacing=(50.0, 128.0, 128.0), preset="pretty")
        assert "pretty" in str(e.value)
        assert "faithful" in str(e.value), "the error should list the valid presets"

    def test_empty_volume_returns_no_geometry_rather_than_raising(self):
        """The caller decides what an empty neuron means; this is just geometry."""
        verts, faces = volume_to_mesh(np.zeros((8, 8, 8), np.uint8),
                                      spacing=(50.0, 128.0, 128.0))
        assert len(verts) == 0 and len(faces) == 0

    def test_preset_table_matches_the_spec(self):
        assert PRESETS["faithful"] == {"step_size": 1, "iterations": 2}
        assert PRESETS["balanced"] == {"step_size": 1, "iterations": 10}
        assert PRESETS["smooth"] == {"step_size": 2, "iterations": 25}

    def test_balanced_smooths_further_than_faithful_at_matched_vertex_count(self):
        """Regression test for a bug where volume_to_mesh silently ignores
        cfg["iterations"] and always smooths with the default iterations count.

        `faithful` (iterations=2) and `balanced` (iterations=10) share step_size=1, so
        marching cubes produces the same base mesh for both: same vertex count and the
        same vertex-to-vertex correspondence. That means triangle/vertex counts alone
        (as in test_presets_differ_in_triangle_count and
        test_preset_table_matches_the_spec) cannot tell them apart; a bug that used the
        `faithful` iterations count for every preset except the one already covered by
        test_faithful_is_the_default would pass every other test in this file.

        Instead this compares actual vertex positions. More Taubin iterations should
        keep moving vertices away from where marching cubes originally placed them, so
        `balanced` must land further from `faithful` than two `faithful` runs land
        from each other. Two `faithful` runs are a repeat of the same preset
        (iterations forced to 2 on both sides), so their displacement is a
        deterministic noise floor of zero; if `balanced` matched that floor, iterations
        would not be doing anything.
        """
        vol = _blob()
        spacing = (50.0, 128.0, 128.0)
        v_faithful, f_faithful = volume_to_mesh(vol, spacing=spacing, preset="faithful")
        v_faithful_again, _f_again = volume_to_mesh(vol, spacing=spacing, preset="faithful")
        v_balanced, f_balanced = volume_to_mesh(vol, spacing=spacing, preset="balanced")

        assert v_faithful.shape == v_balanced.shape, (
            "faithful and balanced share step_size, so they should produce the same "
            "vertex count; if they do not, marching cubes itself is nondeterministic "
            "here and this test's premise does not hold")
        assert len(f_faithful) == len(f_balanced), "faces should match too, same reason"

        repeat_displacement = float(
            np.linalg.norm(v_faithful - v_faithful_again, axis=1).mean())
        displacement = float(np.linalg.norm(v_balanced - v_faithful, axis=1).mean())

        assert repeat_displacement == 0.0, (
            "volume_to_mesh should be deterministic for a fixed preset on a fixed "
            "volume; the fixture or the smoothing is not, so this comparison is not "
            "meaningful")
        assert displacement > repeat_displacement, (
            f"balanced (iterations=10) produced vertices no further from faithful "
            f"(iterations=2) than two faithful runs are from each other (mean "
            f"displacement {displacement:.6f}nm); this means volume_to_mesh is not "
            f"actually using cfg['iterations']")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
