"""Turn a binary volume into a mesh a person can open in Blender.

Pure geometry: numpy only, no project IO and no knowledge of chains, bundles or
trees. Callers hand in vertices and faces and get vertices and faces back. Marching
cubes, which turns a volume into that starting mesh, is built on top of this module
in a later step; this module only smooths and writes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    """Every triangle edge, both directions, as an (M, 2) array of vertex indices."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.vstack([e, e[:, ::-1]])


def _toward_neighbour_centroid(v: np.ndarray, edges: np.ndarray, factor: float) -> np.ndarray:
    """Move every vertex `factor` of the way to the mean of its neighbours.

    A vertex with no neighbours keeps a count of 1 and a centroid equal to itself, so
    it stays put instead of dividing by zero. Marching cubes output can contain
    isolated vertices, and a single NaN propagates through every later step.
    """
    sums = np.zeros_like(v)
    counts = np.zeros(v.shape[0])
    np.add.at(sums, edges[:, 0], v[edges[:, 1]])
    np.add.at(counts, edges[:, 0], 1.0)
    lonely = counts == 0
    counts[lonely] = 1.0
    sums[lonely] = v[lonely]
    centroid = sums / counts[:, None]
    return v + factor * (centroid - v)


def taubin_smooth(verts: np.ndarray, faces: np.ndarray, *, iterations: int = 2,
                  lamb: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin lambda/mu smoothing, which does NOT shrink the surface the way plain
    Laplacian smoothing does.

    Each iteration is a positive step toward the neighbour centroid followed by a
    slightly larger negative one. The negative pass undoes most of the shrinkage the
    positive pass causes while leaving the high-frequency noise removed, which is why
    this is the right choice for neurites: they are thin tubes, and Laplacian
    smoothing eats thin tubes. Passing ``mu=0.0`` reduces this to Laplacian, which the
    tests use as the comparison case.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    if iterations <= 0 or len(faces) == 0:
        return v
    edges = _directed_edges(np.asarray(faces, dtype=np.int64))
    for _ in range(int(iterations)):
        v = _toward_neighbour_centroid(v, edges, lamb)
        if mu:
            v = _toward_neighbour_centroid(v, edges, mu)
    return v


def write_ply(path, verts: np.ndarray, faces: np.ndarray) -> Path:
    """Write an ASCII PLY. Hand-written because the format is a short header plus two
    arrays, and a dependency for that would not earn itself."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError(
            "refusing to write an empty mesh: no surface was produced, which means the "
            "neuron had no masks rather than that the export succeeded")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format ascii 1.0",
              f"element vertex {len(verts)}",
              "property float x", "property float y", "property float z",
              f"element face {len(faces)}",
              "property list uchar int vertex_indices",
              "end_header"]
    lines = header
    lines += [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


#: Detail presets. `faithful` is the default because the reason these meshes exist is
#: to find mistakes, and smoothing removes exactly the z-to-z jitter that marks a bad
#: slice. Triangle reduction is via marching cubes `step_size`, which is cruder than
#: quadric decimation; proper decimation needs trimesh or open3d, which the reviewer's
#: install does not have. That is a known limitation, not an oversight.
PRESETS = {
    "faithful": {"step_size": 1, "iterations": 2},
    "balanced": {"step_size": 1, "iterations": 10},
    "smooth": {"step_size": 2, "iterations": 25},
}


def volume_to_mesh(vol: np.ndarray, *, spacing: Tuple[float, float, float],
                   preset: str = "faithful") -> Tuple[np.ndarray, np.ndarray]:
    """Surface a binary volume, with the anisotropy baked into the coordinates.

    `spacing` is `(z, y, x)` in nanometres because that is the array's own axis order.
    Vertices are returned in **x, y, z** order, which is what Blender expects, so the
    axes are reversed on the way out. At `_sam` scale 8 the correct spacing is
    `(50, 128, 128)`: full res is 16nm in xy and 50nm in z, so xy coarsens to 128nm
    while z does not change. That makes z the FINER axis, the opposite of the usual EM
    intuition, and getting it backwards yields a mesh that looks plausible and is wrong
    by 2.5x in one direction.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, choose one of {sorted(PRESETS)}")
    vol = np.asarray(vol)
    if not vol.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    from skimage.measure import marching_cubes
    cfg = PRESETS[preset]
    verts_zyx, faces, _normals, _values = marching_cubes(
        vol.astype(np.float32), level=0.5, spacing=spacing,
        step_size=cfg["step_size"], allow_degenerate=False)
    verts_zyx = taubin_smooth(verts_zyx, faces, iterations=cfg["iterations"])
    verts_xyz = verts_zyx[:, ::-1].copy()      # (z, y, x) -> (x, y, z) for Blender
    return verts_xyz, np.asarray(faces, dtype=np.int64)
