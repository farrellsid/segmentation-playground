"""Turn a binary volume into a mesh a person can open in Blender.

Pure geometry: numpy only, no project IO and no knowledge of chains, bundles or
trees. Callers hand in vertices and faces and get vertices and faces back. Marching
cubes, which turns a volume into that starting mesh, is built on top of this module
in a later step; this module only smooths and writes.
"""
from __future__ import annotations

from pathlib import Path

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
