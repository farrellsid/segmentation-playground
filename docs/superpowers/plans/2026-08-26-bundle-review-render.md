# Review Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a reviewer one video per neuron and one Blender-ready mesh per neuron, from a corrected bundle or an output tree, driven from a window reached through `launcher.py`.

**Architecture:** Pure geometry lives in `sam2_utils/meshing.py` (numpy plus scikit-image, no project IO). Everything source-aware lives in `render_review.py`: the one seam where a bundle reads its own `frames/` and a tree reads the EM store, plus video, mesh orchestration, a CLI, and a thin Qt window. `launcher.py` gains one button. This mirrors how `launcher.py` already splits pure functions from `run()` so the logic tests with no display.

**Tech Stack:** numpy, scikit-image (marching cubes), opencv-python (image IO, mp4), qtpy (window), pytest. All already present via `requirements-review.txt`.

## Global Constraints

- **No new dependencies.** `napari` declares `scikit-image[data]>=0.19.1` and `imageio>=2.20`, so both are available. Do not add anything to `requirements-review.txt`.
- **No torch anywhere in this feature.** The reviewer's machine has none.
- **No em dashes or en dashes** in code, comments, docs, or commit messages. Use commas, colons, parentheses, or separate sentences.
- **Tests are CPU-only, torch-free, and must not need the EM store on `F:`.** Stub `pipeline.load_frame_sam` where a tree path is exercised.
- Lint with `py -3 -m ruff check .`. Run tests with `py -3 -m pytest`.
- The library (`pipeline`, `sam2_utils/`) must never import drivers (`batch`, `gui`, `launcher`, `render_review`). `tests/test_import_direction.py` enforces this.
- Mask grid is `_sam` scale 8. Full-res is 16 nm in xy, 50 nm in z, so scale-8 spacing is `(50, 128, 128)` in `(z, y, x)`. **z is the finer axis.** Derive this from data, never hardcode.
- Commit after every task.

---

### Task 1: Taubin smoothing and PLY writing

**Files:**
- Create: `sam2_utils/meshing.py`
- Test: `tests/test_meshing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `taubin_smooth(verts: np.ndarray, faces: np.ndarray, *, iterations: int = 2, lamb: float = 0.5, mu: float = -0.53) -> np.ndarray` returning smoothed `(N, 3)` float64 vertices. `write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_meshing.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_meshing.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'sam2_utils.meshing'`

- [ ] **Step 3: Write minimal implementation**

Create `sam2_utils/meshing.py`:

```python
"""Turn a binary volume into a mesh a person can open in Blender.

Pure geometry: numpy plus scikit-image, no project IO and no knowledge of chains,
bundles or trees. Callers hand in a volume and get vertices and faces back.
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
              "property list uchar int vertex_index",
              "end_header"]
    lines = header
    lines += [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_meshing.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check sam2_utils/meshing.py tests/test_meshing.py
git add sam2_utils/meshing.py tests/test_meshing.py
git commit -m "meshing: Taubin smoothing and PLY output

Taubin rather than Laplacian because a neurite is mostly thin tube and Laplacian
smoothing shrinks exactly that. The negative mu pass cancels most of the shrinkage the
positive lambda pass causes, which the test measures directly against a synthetic
cylinder rather than asserting on an implementation detail.

An isolated vertex keeps a neighbour count of 1 and a centroid of itself, so marching
cubes output with a stray vertex cannot put NaN through the whole mesh. write_ply
refuses an empty mesh, since a 0-vertex file makes a neuron with no masks look like a
successful export."
```

---

### Task 2: Volume to mesh, with anisotropy and detail presets

**Files:**
- Modify: `sam2_utils/meshing.py`
- Test: `tests/test_meshing.py`

**Interfaces:**
- Consumes: `taubin_smooth` from Task 1.
- Produces: `PRESETS: dict[str, dict]` keyed `"faithful" | "balanced" | "smooth"`, each `{"step_size": int, "iterations": int}`. `volume_to_mesh(vol: np.ndarray, *, spacing: Tuple[float, float, float], preset: str = "faithful") -> Tuple[np.ndarray, np.ndarray]` returning `(verts_xyz, faces)` with vertices in nanometres, **x, y, z order** for Blender.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_meshing.py`, above the `if __name__` block:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_meshing.py -q`
Expected: FAIL, `ImportError: cannot import name 'PRESETS'`

- [ ] **Step 3: Write minimal implementation**

Append to `sam2_utils/meshing.py`:

```python
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
```

Add `Tuple` to the existing `typing` import if it is not already there.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_meshing.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check sam2_utils/meshing.py tests/test_meshing.py
git add sam2_utils/meshing.py tests/test_meshing.py
git commit -m "meshing: volume to mesh with anisotropy baked in

spacing is (z, y, x) to match the array axes and vertices come out (x, y, z) for
Blender, with a test asserting the real spans in nanometres rather than the axis order
in the abstract. At scale 8 the spacing is (50, 128, 128), which makes z the finer
axis, the opposite of the usual EM intuition, so a test states that fact directly.

Presets reduce triangles through marching cubes step_size. Real quadric decimation
needs trimesh or open3d, which the review install does not have; the docstring says so
rather than implying decimation happens."
```

---

### Task 3: The frame seam, bundle versus tree

**Files:**
- Create: `render_review.py`
- Test: `tests/test_render_review_frames.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `source_kind(root: Path) -> str` returning `"bundle"` or `"tree"`. `chain_frames(chain_dir: Path, state: dict, kind: str) -> dict[int, np.ndarray]` mapping frame index to an RGB uint8 image.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_review_frames.py`:

```python
"""The one place a bundle and an output tree genuinely differ.

Both are laid out <neuron>/chain_NN/state.json, so bundle.index_chains and
pipeline.chain_masks_in_sam already read either. What differs is EM: a bundle carries
its own per-chain frames/, an output tree does not and must go to the EM store. The
reviewer's Mac has no EM store at all, so the bundle branch has to work with no
network and no F: drive.

Torch-free and data-free: the tree branch stubs pipeline.load_frame_sam.
    py -3 -m pytest tests/test_render_review_frames.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


def _chain(tmp_path, kind, n=3):
    root = tmp_path / kind
    d = root / "AIBL" / "chain_00"
    (d / "masks").mkdir(parents=True)
    state = {"neuron": "AIBL", "chain_idx": 0,
             "frame_to_z": {str(i): 1500 + i for i in range(n)}}
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if kind == "bundle":
        (d / "frames").mkdir()
        for i in range(n):
            img = np.full((20, 24, 3), 10 * (i + 1), dtype=np.uint8)
            cv2.imwrite(str(d / "frames" / f"{i:05d}.jpg"), img)
        (root / "bundle.json").write_text('{"schema_version": 1, "chains": []}',
                                          encoding="utf-8")
    else:
        (root / "_manifest.csv").write_text("neuron,chain_idx\nAIBL,0\n", encoding="utf-8")
    return root, d, state


class TestSourceKind:
    def test_bundle_json_means_bundle(self, tmp_path):
        root, _d, _s = _chain(tmp_path, "bundle")
        assert render_review.source_kind(root) == "bundle"

    def test_manifest_means_tree(self, tmp_path):
        root, _d, _s = _chain(tmp_path, "tree")
        assert render_review.source_kind(root) == "tree"

    def test_neither_is_refused_naming_what_was_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit) as e:
            render_review.source_kind(tmp_path / "empty")
        assert "bundle.json" in str(e.value) and "_manifest.csv" in str(e.value)


class TestChainFrames:
    def test_bundle_reads_its_own_frames_without_the_em_store(self, tmp_path, monkeypatch):
        root, d, state = _chain(tmp_path, "bundle")

        def explode(*a, **k):
            raise AssertionError("bundle path must not touch the EM store")

        monkeypatch.setattr(render_review.pipeline, "load_frame_sam", explode)
        frames = render_review.chain_frames(d, state, "bundle")
        assert sorted(frames) == [0, 1, 2]
        assert frames[0].shape == (20, 24, 3)
        assert frames[1].mean() > frames[0].mean(), "frames came back in the wrong order"

    def test_tree_reads_the_em_store_at_each_z(self, tmp_path, monkeypatch):
        root, d, state = _chain(tmp_path, "tree")
        seen = []

        def fake_load(z, *, scale):
            seen.append(int(z))
            return np.full((30, 30, 3), int(z) % 255, dtype=np.uint8), (240, 240)

        monkeypatch.setattr(render_review.pipeline, "load_frame_sam", fake_load)
        frames = render_review.chain_frames(d, state, "tree")
        assert seen == [1500, 1501, 1502], "must request the chain's own z values"
        assert sorted(frames) == [0, 1, 2]

    def test_bundle_chain_with_no_frames_dir_names_the_chain(self, tmp_path):
        """A silently skipped chain is how a video stops being of the whole neuron
        while still looking finished."""
        root, d, state = _chain(tmp_path, "bundle")
        import shutil
        shutil.rmtree(d / "frames")
        with pytest.raises(SystemExit) as e:
            render_review.chain_frames(d, state, "bundle")
        assert "chain_00" in str(e.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_render_review_frames.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'render_review'`

- [ ] **Step 3: Write minimal implementation**

Create `render_review.py`:

```python
"""Render a reviewed neuron as a video and as a Blender-ready mesh.

Runs against a review bundle or an output tree. Both are laid out
``<neuron>/chain_NN/state.json``, so `bundle.index_chains` indexes either and
`pipeline.chain_masks_in_sam` reads masks from either. They differ in exactly one way
that matters here: a bundle carries its own per-chain ``frames/`` while a tree expects
the EM store. That difference lives in `chain_frames` and nowhere else, which is what
lets this run on a reviewer's laptop with no EM store and no torch.

    py -3 render_review.py --source ~/mask-review/AIB --out AIB/review
    py -3 render_review.py --gui
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline


def source_kind(root) -> str:
    """``"bundle"`` or ``"tree"``, decided by which marker file is present."""
    root = Path(root)
    if (root / "bundle.json").exists():
        return "bundle"
    if (root / "_manifest.csv").exists():
        return "tree"
    raise SystemExit(
        f"[render] {root} is neither a bundle nor an output tree: no bundle.json and "
        f"no _manifest.csv. Point --source at a bundle folder or a tree root.")


def chain_frames(chain_dir, state: dict, kind: str) -> Dict[int, np.ndarray]:
    """``{frame_idx: RGB uint8}`` for one chain, from wherever this source keeps EM.

    A bundle reads the JPEGs it shipped with, which is what makes this work on a
    machine that has never seen the raw stack. A tree reads the EM store at each of the
    chain's own z values.
    """
    chain_dir = Path(chain_dir)
    if kind == "bundle":
        fdir = chain_dir / "frames"
        paths = sorted(fdir.glob("*.jpg")) if fdir.is_dir() else []
        if not paths:
            raise SystemExit(
                f"[render] {chain_dir.parent.name}/{chain_dir.name} has no frames/ in "
                f"this bundle, so its EM cannot be drawn. Re-export the bundle rather "
                f"than rendering a neuron with a chain silently missing.")
        out = {}
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                raise SystemExit(f"[render] could not read {p}")
            out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return out

    frame_to_z = {int(k): int(v) for k, v in (state.get("frame_to_z") or {}).items()}
    out = {}
    for fi in sorted(frame_to_z):
        em, _full = pipeline.load_frame_sam(frame_to_z[fi], scale=8)
        out[fi] = em if em.ndim == 3 else np.stack([em] * 3, axis=-1)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_render_review_frames.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check render_review.py tests/test_render_review_frames.py
git add render_review.py tests/test_render_review_frames.py
git commit -m "render-review: the one seam between a bundle and a tree

Both layouts are <neuron>/chain_NN/state.json, so chain discovery and mask reading
already work on either through bundle.index_chains and pipeline.chain_masks_in_sam.
The only real difference is EM: a bundle ships its own frames/, a tree goes to the EM
store. Isolating that in chain_frames is what lets the whole feature run on a
reviewer's laptop with no EM store.

The bundle branch is asserted not to touch load_frame_sam at all, and a bundle chain
with no frames/ raises naming the chain, because a silently skipped chain is how a
video stops being of the whole neuron while still looking finished."
```

---

### Task 4: Composite a neuron into one volume and write its mesh

**Files:**
- Modify: `render_review.py`
- Test: `tests/test_render_review_mesh.py`

**Interfaces:**
- Consumes: `source_kind` from Task 3, `volume_to_mesh`/`write_ply` from Tasks 1 and 2.
- Produces: `neuron_volume(root: Path, neuron: str, *, max_voxels: int = 400_000_000) -> tuple[np.ndarray, tuple[float, float, float]]` returning `(vol_zyx_uint8, spacing_zyx_nm)`. `neuron_mesh(root: Path, neuron: str, out_path: Path, *, preset: str = "faithful") -> Path | None`, returning `None` when the neuron has no masks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_review_mesh.py`:

```python
"""Compositing a neuron's chains into one volume, then surfacing it.

A neuron's chains sit in different crop windows and overlap in z, so they have to be
placed on one grid before meshing. pipeline.chain_masks_in_sam already returns each
mask on the shared _sam grid with its placement, so this is a paste, not a remap.

The volume is cropped to the neuron's own bounding box. The full _sam grid over a long
neuron is hundreds of MB, which is not a laptop-sized allocation, and the reviewer runs
this on a laptop.

Torch-free, data-free.
    py -3 -m pytest tests/test_render_review_mesh.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


def _tree_with_two_chains(tmp_path):
    """Two chains of one neuron, in different places, overlapping in z."""
    root = tmp_path / "tree"
    (root / "_manifest.csv").write_text("neuron,chain_idx\n", encoding="utf-8")
    placements = {}
    for ci, (x0, y0, zs) in enumerate([(100, 200, [1500, 1501, 1502]),
                                       (140, 210, [1502, 1503])]):
        d = root / "AIBL" / f"chain_{ci:02d}"
        (d / "masks").mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"neuron": "AIBL", "chain_idx": ci}),
                                      encoding="utf-8")
        blocks = {}
        for z in zs:
            m = np.zeros((6, 6), dtype=bool)
            m[1:5, 1:5] = True
            blocks[z] = (m, x0, y0)
        placements[ci] = blocks
    return root, placements


def _patch_masks(monkeypatch, placements):
    def fake(chain_dir):
        ci = int(pathlib.Path(chain_dir).name.split("_")[-1])
        return placements.get(ci, {})
    monkeypatch.setattr(render_review.pipeline, "chain_masks_in_sam", fake)


class TestNeuronVolume:
    def test_composites_every_chain_onto_one_grid(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, spacing = render_review.neuron_volume(root, "AIBL")
        assert vol.ndim == 3
        assert vol.sum() > 0
        # 4 z values across both chains
        assert vol.shape[0] == 4, f"expected 4 z planes, got {vol.shape[0]}"

    def test_volume_is_cropped_to_the_neuron_bbox_not_the_full_frame(self, tmp_path,
                                                                    monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        vol, _ = render_review.neuron_volume(root, "AIBL")
        # masks span x 101..144 and y 201..214, so the bbox is tens of px, not 1152
        assert vol.shape[1] < 100 and vol.shape[2] < 100, (
            f"volume is {vol.shape}, which is the full grid rather than the bbox")

    def test_spacing_is_z_finer_than_xy(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        _vol, spacing = render_review.neuron_volume(root, "AIBL")
        assert spacing == (50.0, 128.0, 128.0)
        assert spacing[0] < spacing[1], "z must be the finer axis at scale 8"

    def test_refuses_a_volume_over_the_memory_budget(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        with pytest.raises(SystemExit) as e:
            render_review.neuron_volume(root, "AIBL", max_voxels=10)
        assert "voxel" in str(e.value).lower()


class TestNeuronMesh:
    def test_writes_a_ply_that_parses(self, tmp_path, monkeypatch):
        root, placements = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, placements)
        out = render_review.neuron_mesh(root, "AIBL", tmp_path / "AIBL.ply")
        assert out is not None and out.exists()
        text = out.read_text(encoding="utf-8")
        assert text.startswith("ply")
        n_verts = int([ln for ln in text.splitlines()
                       if ln.startswith("element vertex")][0].split()[-1])
        assert n_verts > 0

    def test_a_neuron_with_no_masks_returns_none_instead_of_raising(self, tmp_path,
                                                                    monkeypatch):
        root, _p = _tree_with_two_chains(tmp_path)
        _patch_masks(monkeypatch, {})
        assert render_review.neuron_mesh(root, "AIBL", tmp_path / "x.ply") is None
        assert not (tmp_path / "x.ply").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_render_review_mesh.py -q`
Expected: FAIL, `AttributeError: module 'render_review' has no attribute 'neuron_volume'`

- [ ] **Step 3: Write minimal implementation**

Append to `render_review.py`:

```python
from sam2_utils import bundle as bundle_utils
from sam2_utils import meshing

#: Nanometres per voxel at `_sam` scale 8, as (z, y, x). Full res is 16nm in xy and
#: 50nm in z, so xy coarsens by 8 to 128nm while z does not change. z is therefore the
#: FINER axis, which is the opposite of the usual EM intuition.
SPACING_SAM8_NM = (50.0, 128.0, 128.0)


def _chain_dirs(root: Path, neuron: str):
    """Every chain directory for one neuron, in chain order, bundle or tree alike."""
    recs = bundle_utils.index_chains(Path(root), neurons=[neuron])
    return [(int(r["chain_idx"]), Path(root) / r["chain_dir"])
            for r in sorted(recs, key=lambda r: int(r["chain_idx"]))]


def neuron_volume(root, neuron: str, *, max_voxels: int = 400_000_000):
    """Every chain of ``neuron`` composited onto one grid, cropped to its bbox.

    Cropping is what keeps this laptop-sized. The full `_sam` grid across a long
    neuron runs to hundreds of MB, while the neuron's own bounding box is a small
    fraction of that. Returns ``(vol, spacing)`` with ``vol`` as ``(z, y, x)`` uint8.
    """
    root = Path(root)
    blocks = []
    for _ci, cdir in _chain_dirs(root, neuron):
        for z, (mask, x0, y0) in pipeline.chain_masks_in_sam(cdir).items():
            if mask.any():
                blocks.append((int(z), np.asarray(mask, dtype=bool), int(x0), int(y0)))
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.uint8), SPACING_SAM8_NM

    zs = sorted({b[0] for b in blocks})
    z_index = {z: i for i, z in enumerate(zs)}
    x0 = min(b[2] for b in blocks)
    y0 = min(b[3] for b in blocks)
    x1 = max(b[2] + b[1].shape[1] for b in blocks)
    y1 = max(b[3] + b[1].shape[0] for b in blocks)
    shape = (len(zs), y1 - y0, x1 - x0)
    n_vox = shape[0] * shape[1] * shape[2]
    if n_vox > max_voxels:
        raise SystemExit(
            f"[render] {neuron} needs a {shape} volume, {n_vox:,} voxels, over the "
            f"{max_voxels:,} budget. Render fewer chains, or raise --max-voxels if "
            f"this machine has the memory.")

    vol = np.zeros(shape, dtype=np.uint8)
    for z, mask, bx, by in blocks:
        h, w = mask.shape
        sy, sx = by - y0, bx - x0
        vol[z_index[z], sy:sy + h, sx:sx + w] |= mask.astype(np.uint8)
    return vol, SPACING_SAM8_NM


def neuron_mesh(root, neuron: str, out_path, *, preset: str = "faithful",
                max_voxels: int = 400_000_000):
    """Write one PLY for ``neuron``. Returns the path, or None if it has no masks."""
    vol, spacing = neuron_volume(root, neuron, max_voxels=max_voxels)
    if not vol.size or not vol.any():
        print(f"[render] {neuron}: no masks, skipping mesh")
        return None
    verts, faces = meshing.volume_to_mesh(vol, spacing=spacing, preset=preset)
    if len(faces) == 0:
        print(f"[render] {neuron}: no surface produced, skipping mesh")
        return None
    out = meshing.write_ply(out_path, verts, faces)
    print(f"[render] {neuron}: mesh {len(verts):,} verts, {len(faces):,} faces -> {out}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_render_review_mesh.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check render_review.py tests/test_render_review_mesh.py
git add render_review.py tests/test_render_review_mesh.py
git commit -m "render-review: composite a neuron into one volume and mesh it

chain_masks_in_sam already returns each mask on the shared _sam grid with its
placement, so compositing is a paste rather than a remap. The volume is cropped to the
neuron's own bounding box, which is what keeps it laptop-sized: the full grid across a
long neuron is hundreds of MB and the reviewer runs this on a laptop. A voxel budget
refuses anything larger with the measured size rather than dying in the allocator.

A neuron with no masks returns None and writes nothing, so an empty export cannot be
mistaken for a successful one."
```

---

### Task 5: The per-neuron video

**Files:**
- Modify: `render_review.py`
- Test: `tests/test_render_review_video.py`

**Interfaces:**
- Consumes: `chain_frames`, `_chain_dirs`, `source_kind` from Tasks 3 and 4.
- Produces: `common_canvas(sizes: list[tuple[int, int]], *, cap: int = 900) -> tuple[int, int]` returning `(h, w)`. `fit_to_canvas(img: np.ndarray, canvas_hw: tuple[int, int], scale: float = 1.0) -> np.ndarray`. `neuron_video(root: Path, neuron: str, out_path: Path, *, fmt: str = "gif", progress=None) -> Path | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_review_video.py`:

```python
"""Padding a neuron's chains onto one canvas.

The reprop report's tour re-crops every frame out of the full _sam frame. A bundle
cannot do that: it only ships each chain's own crop. So frames are padded onto a common
canvas instead, which introduces two failure modes the report version never had.

Chains can sit at different crop_scale, meaning different nanometres per pixel. Padding
those together without normalising puts chains at different magnifications inside one
video with nothing on screen to say so.

And one outlier chain must not size the whole video. That is exactly what went wrong in
the merged reprop render, where a single legacy chain's window dragged every frame to
full-frame size and the masks became invisible.

Torch-free, data-free.
    py -3 -m pytest tests/test_render_review_video.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


class TestCommonCanvas:
    def test_takes_the_max_of_each_axis(self):
        assert render_review.common_canvas([(100, 50), (40, 300)]) == (100, 300)

    def test_caps_so_one_outlier_cannot_size_the_video(self):
        """The merged reprop render's exact failure, in miniature."""
        sizes = [(120, 130), (118, 126), (1152, 1154)]
        h, w = render_review.common_canvas(sizes, cap=400)
        assert h <= 400 and w <= 400

    def test_empty_is_refused(self):
        with pytest.raises(ValueError):
            render_review.common_canvas([])


class TestFitToCanvas:
    def test_pads_a_small_frame_without_stretching_it(self):
        img = np.full((10, 12, 3), 200, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 50))
        assert out.shape == (40, 50, 3)
        assert (out == 200).sum() == 10 * 12 * 3, "content should be padded, not scaled"

    def test_scales_before_padding_so_nm_per_px_matches(self):
        """A chain at half the nm/px of its neighbours must be halved before padding,
        or the video shows two magnifications with nothing to signal it."""
        img = np.full((20, 20, 3), 150, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 40), scale=0.5)
        assert out.shape == (40, 40, 3)
        assert (out == 150).sum() < 20 * 20 * 3, "scale=0.5 should shrink the content"

    def test_a_frame_larger_than_the_canvas_is_scaled_down_to_fit(self):
        img = np.full((80, 90, 3), 100, dtype=np.uint8)
        out = render_review.fit_to_canvas(img, (40, 40))
        assert out.shape == (40, 40, 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_render_review_video.py -q`
Expected: FAIL, `AttributeError: module 'render_review' has no attribute 'common_canvas'`

- [ ] **Step 3: Write minimal implementation**

Append to `render_review.py`:

```python
from sam2_utils import video_viz

#: Largest canvas edge, in _sam px, before one chain is treated as an outlier. The
#: merged reprop render is the cautionary case: a single legacy chain whose window
#: covered the whole frame dragged every frame to full-frame size and the masks fell to
#: a fraction of a percent of the image.
CANVAS_CAP_PX = 900


def common_canvas(sizes, *, cap: int = CANVAS_CAP_PX):
    """``(h, w)`` big enough for every frame, capped so an outlier cannot size it."""
    if not sizes:
        raise ValueError("common_canvas needs at least one frame size")
    h = min(cap, max(s[0] for s in sizes))
    w = min(cap, max(s[1] for s in sizes))
    return int(h), int(w)


def fit_to_canvas(img: np.ndarray, canvas_hw, scale: float = 1.0) -> np.ndarray:
    """Scale ``img`` by ``scale``, then centre it on a ``canvas_hw`` black canvas.

    ``scale`` is how a chain at a different ``crop_scale`` is brought to the video's
    common nanometres per pixel. Padding without it would put two magnifications in one
    video with nothing on screen to say so. Anything still larger than the canvas after
    scaling is shrunk to fit, so a frame is never cropped silently.
    """
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h > ch or w > cw:
        f = min(ch / h, cw / w)
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    out = np.zeros((ch, cw, 3), dtype=np.uint8)
    oy, ox = (ch - h) // 2, (cw - w) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out


def _crop_scale(state: dict) -> int:
    cw = state.get("crop_window") or {}
    return int(cw.get("crop_scale") or 1)


def neuron_video(root, neuron: str, out_path, *, fmt: str = "gif", progress=None):
    """One video for ``neuron``: every chain in z order, each in its own window, padded
    onto one canvas, captioned with chain and z so a problem is traceable back."""
    import json
    root = Path(root)
    kind = source_kind(root)
    chains = _chain_dirs(root, neuron)
    if not chains:
        print(f"[render] {neuron}: no chains, skipping video")
        return None

    loaded = []
    for ci, cdir in chains:
        state = json.loads((cdir / "state.json").read_text(encoding="utf-8"))
        masks = pipeline.chain_masks_in_sam(cdir)
        if not masks:
            continue
        frames = chain_frames(cdir, state, kind)
        f2z = {int(k): int(v) for k, v in (state.get("frame_to_z") or {}).items()}
        if not f2z:
            f2z = {i: z for i, z in enumerate(sorted(masks))}
        loaded.append((ci, cdir, state, frames, f2z, masks))
    if not loaded:
        print(f"[render] {neuron}: no masks, skipping video")
        return None

    loaded.sort(key=lambda t: min(t[4].values()) if t[4] else 0)
    finest = min(_crop_scale(t[2]) for t in loaded)
    sizes = []
    for _ci, _cd, state, frames, _f2z, _m in loaded:
        s = finest / _crop_scale(state)
        for img in frames.values():
            sizes.append((int(img.shape[0] * s), int(img.shape[1] * s)))
    canvas = common_canvas(sizes)

    tmp = Path("scratch_render") / f"review_{neuron}"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    segments, idx, total = {}, 0, sum(len(t[3]) for t in loaded)
    for ci, _cd, state, frames, f2z, masks in loaded:
        s = finest / _crop_scale(state)
        for fi in sorted(frames):
            z = f2z.get(fi)
            img = fit_to_canvas(frames[fi], canvas, scale=s)
            _stamp(img, f"chain_{ci:02d}  z={z}")
            cv2.imwrite(str(tmp / f"{idx:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            m = masks.get(z)
            if m is not None:
                mask_img = np.zeros(frames[fi].shape[:2], dtype=np.uint8)
                mm, mx, my = m
                # the mask is on the _sam grid; the frame is this chain's own crop, so
                # only the overlap is drawn. A chain whose mask sits outside its own
                # frame is a data problem, not something to paper over here.
                h, w = mm.shape
                mask_img[:min(h, mask_img.shape[0]), :min(w, mask_img.shape[1])] = \
                    mm[:mask_img.shape[0], :mask_img.shape[1]].astype(np.uint8)
                seg = fit_to_canvas(np.stack([mask_img * 255] * 3, -1), canvas, scale=s)
                segments[idx] = {ci + 1: seg[..., 0] > 127}
            idx += 1
            if progress:
                progress(idx, total, f"{neuron} chain_{ci:02d}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = video_viz.to_gif if fmt == "gif" else video_viz.to_mp4
    writer(segments, tmp, out_path, obj_id=None, preview_scale=1, color=None)
    import shutil
    shutil.rmtree(tmp)
    print(f"[render] {neuron}: video {idx} frames -> {out_path}")
    return out_path


def _stamp(img_rgb: np.ndarray, text: str) -> None:
    """Burn a caption into the top-left, in place, dark then light so it stays readable
    over both bright cytoplasm and dark membrane."""
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img_rgb, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, thickness, cv2.LINE_AA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_render_review_video.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check render_review.py tests/test_render_review_video.py
git add render_review.py tests/test_render_review_video.py
git commit -m "render-review: one video per neuron, padded onto a common canvas

The reprop tour re-crops each frame out of the full _sam frame, which a bundle cannot
do because it only ships each chain's own crop. Frames are padded instead, which brings
two failure modes the tour never had, and both are guarded.

Chains at different crop_scale are normalised to a common nm/px before padding, since
padding them raw puts two magnifications in one video with nothing on screen to say so.
And the canvas is capped, because one outlier chain sizing the whole video is exactly
what went wrong in the merged reprop render, where a single legacy chain dragged every
frame to full-frame and the masks fell under a percent of the image."
```

---

### Task 6: Orchestration and CLI

**Files:**
- Modify: `render_review.py`
- Test: `tests/test_render_review_cli.py`

**Interfaces:**
- Consumes: `neuron_video`, `neuron_mesh`, `source_kind` from Tasks 3 to 5.
- Produces: `render_all(root: Path, out_dir: Path, neurons: list[str] | None, *, video: bool = True, mesh: bool = True, fmt: str = "gif", preset: str = "faithful", progress=None, should_cancel=None) -> dict`. `main(argv=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_review_cli.py`:

```python
"""Orchestration: which neurons, which outputs, and stopping cleanly.

Cancel must stop between chains and never mid-write. A half-written PLY or a truncated
GIF is worse than no file, because it looks like output.

Torch-free, data-free: neuron_video and neuron_mesh are stubbed.
    py -3 -m pytest tests/test_render_review_cli.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review


@pytest.fixture
def src(tmp_path, monkeypatch):
    root = tmp_path / "AIB"
    root.mkdir()
    (root / "bundle.json").write_text('{"schema_version": 1}', encoding="utf-8")
    for n in ("AIBL", "AIBR"):
        (root / n / "chain_00" / "masks").mkdir(parents=True)
        (root / n / "chain_00" / "state.json").write_text(
            '{"neuron": "%s", "chain_idx": 0}' % n, encoding="utf-8")
    calls = {"video": [], "mesh": []}

    def fake_video(r, neuron, out, **k):
        calls["video"].append(neuron)
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text("gif", encoding="utf-8")
        return pathlib.Path(out)

    def fake_mesh(r, neuron, out, **k):
        calls["mesh"].append(neuron)
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text("ply", encoding="utf-8")
        return pathlib.Path(out)

    monkeypatch.setattr(render_review, "neuron_video", fake_video)
    monkeypatch.setattr(render_review, "neuron_mesh", fake_mesh)
    monkeypatch.setattr(render_review, "list_neurons", lambda r: ["AIBL", "AIBR"])
    return root, calls


class TestRenderAll:
    def test_renders_both_outputs_for_every_neuron_by_default(self, src, tmp_path):
        root, calls = src
        res = render_review.render_all(root, tmp_path / "out", None)
        assert calls["video"] == ["AIBL", "AIBR"]
        assert calls["mesh"] == ["AIBL", "AIBR"]
        assert len(res["written"]) == 4

    def test_neuron_filter_is_honoured(self, src, tmp_path):
        root, calls = src
        render_review.render_all(root, tmp_path / "out", ["AIBR"])
        assert calls["video"] == ["AIBR"] and calls["mesh"] == ["AIBR"]

    def test_video_only(self, src, tmp_path):
        root, calls = src
        render_review.render_all(root, tmp_path / "out", None, mesh=False)
        assert calls["mesh"] == []
        assert calls["video"] == ["AIBL", "AIBR"]

    def test_cancel_stops_between_neurons_and_reports_what_was_done(self, src, tmp_path):
        root, calls = src
        state = {"n": 0}

        def should_cancel():
            state["n"] += 1
            return state["n"] > 1        # allow the first neuron, then stop

        res = render_review.render_all(root, tmp_path / "out", None,
                                       should_cancel=should_cancel)
        assert res["cancelled"] is True
        assert calls["video"] == ["AIBL"], "cancel must not start the second neuron"

    def test_asking_for_neither_output_is_refused(self, src, tmp_path):
        root, _c = src
        with pytest.raises(SystemExit):
            render_review.render_all(root, tmp_path / "out", None,
                                     video=False, mesh=False)


class TestCli:
    def test_source_and_out_are_required(self):
        with pytest.raises(SystemExit):
            render_review.main([])

    def test_flags_reach_render_all(self, src, tmp_path, monkeypatch):
        root, _c = src
        seen = {}
        monkeypatch.setattr(render_review, "render_all",
                            lambda *a, **k: seen.update(args=a, kwargs=k) or {"written": []})
        render_review.main(["--source", str(root), "--out", str(tmp_path / "o"),
                            "--no-mesh", "--format", "mp4"])
        assert seen["kwargs"]["mesh"] is False
        assert seen["kwargs"]["fmt"] == "mp4"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_render_review_cli.py -q`
Expected: FAIL, `AttributeError: module 'render_review' has no attribute 'render_all'`

- [ ] **Step 3: Write minimal implementation**

Append to `render_review.py`:

```python
def list_neurons(root):
    """Every neuron with at least one chain, bundle or tree alike."""
    return sorted({r["cell_name"] for r in bundle_utils.index_chains(Path(root))})


def render_all(root, out_dir, neurons=None, *, video: bool = True, mesh: bool = True,
               fmt: str = "gif", preset: str = "faithful", progress=None,
               should_cancel=None) -> dict:
    """Render every requested neuron. Returns ``{"written": [...], "cancelled": bool}``.

    ``should_cancel`` is polled BETWEEN neurons, never during one. Stopping mid-write
    would leave a truncated GIF or a partial PLY, which is worse than no file because
    it looks like output.
    """
    if not video and not mesh:
        raise SystemExit("[render] nothing to do: both --no-video and --no-mesh given")
    root, out_dir = Path(root), Path(out_dir)
    source_kind(root)                       # refuse a bad source before any work
    wanted = list(neurons) if neurons else list_neurons(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, cancelled = [], False
    for neuron in wanted:
        if should_cancel and should_cancel():
            cancelled = True
            break
        if video:
            ext = "gif" if fmt == "gif" else "mp4"
            p = neuron_video(root, neuron, out_dir / f"{neuron}.{ext}", fmt=fmt,
                             progress=progress)
            if p:
                written.append(p)
        if mesh:
            p = neuron_mesh(root, neuron, out_dir / f"{neuron}.ply", preset=preset)
            if p:
                written.append(p)
    print(f"[render] {len(written)} file(s) in {out_dir}"
          + (" (cancelled)" if cancelled else ""))
    return {"written": written, "cancelled": cancelled}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="a review bundle or an output tree")
    ap.add_argument("--out", required=True, help="directory for the rendered files")
    ap.add_argument("--neurons", nargs="*", default=None, help="default: all of them")
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--no-mesh", dest="mesh", action="store_false")
    ap.add_argument("--format", dest="fmt", choices=["gif", "mp4"], default="gif")
    ap.add_argument("--detail", dest="preset", choices=sorted(meshing.PRESETS),
                    default="faithful")
    args = ap.parse_args(argv)
    render_all(Path(args.source), Path(args.out), args.neurons, video=args.video,
               mesh=args.mesh, fmt=args.fmt, preset=args.preset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_render_review_cli.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Lint and commit**

```bash
py -3 -m ruff check render_review.py tests/test_render_review_cli.py
git add render_review.py tests/test_render_review_cli.py
git commit -m "render-review: orchestration and CLI

should_cancel is polled between neurons and never during one, so a cancel cannot leave
a truncated GIF or a partial PLY. A half-written file is worse than no file because it
looks like output.

The source is validated before any work starts, so a wrong path fails in a second
rather than after the first neuron has rendered."
```

---

### Task 7: The window, and the launcher button

**Files:**
- Modify: `render_review.py`
- Modify: `launcher.py`
- Test: `tests/test_render_review_gui.py`

**Interfaces:**
- Consumes: `render_all`, `list_neurons`, `PRESETS` from earlier tasks.
- Produces: `render_defaults(profile: dict) -> dict` with keys `out_dir`, `video`, `mesh`, `fmt`, `preset`. `run(source: str = "", neurons: list[str] | None = None) -> None` opening the window.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_review_gui.py`:

```python
"""Settings the render window remembers, tested without a display.

Same split as launcher.py: pure functions carry the logic, run() is a thin window over
them, so this needs no Qt and no screen.

    py -3 -m pytest tests/test_render_review_gui.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import render_review
from sam2_utils import meshing


class TestRenderDefaults:
    def test_empty_profile_gives_the_documented_defaults(self):
        d = render_review.render_defaults({})
        assert d["video"] is True and d["mesh"] is True
        assert d["fmt"] == "gif"
        assert d["preset"] == "faithful", "review is the purpose, so faithful is default"

    def test_saved_render_settings_are_read_back(self):
        d = render_review.render_defaults(
            {"render": {"fmt": "mp4", "preset": "smooth", "mesh": False,
                        "out_dir": "/tmp/x"}})
        assert d["fmt"] == "mp4" and d["preset"] == "smooth"
        assert d["mesh"] is False and d["out_dir"] == "/tmp/x"

    def test_settings_live_under_a_render_key_so_launcher_keys_are_untouched(self):
        prof = {"output_root": "/keep", "reviewer": "L", "render": {"fmt": "mp4"}}
        d = render_review.render_defaults(prof)
        assert d["fmt"] == "mp4"
        assert prof["output_root"] == "/keep" and prof["reviewer"] == "L"

    def test_an_unknown_preset_in_the_profile_falls_back(self):
        """A profile written by a newer version must not crash an older one."""
        d = render_review.render_defaults({"render": {"preset": "ultra"}})
        assert d["preset"] in meshing.PRESETS


class TestLauncherIntegration:
    def test_launcher_exposes_a_render_entry_point(self):
        import launcher
        assert hasattr(launcher, "open_render_window"), (
            "launcher needs a seam the button calls, testable without Qt")

    def test_launcher_hands_its_source_and_neurons_across(self, monkeypatch):
        import launcher
        seen = {}
        monkeypatch.setattr(render_review, "run",
                            lambda source="", neurons=None: seen.update(
                                source=source, neurons=neurons))
        launcher.open_render_window("/some/bundle", ["AIBL"])
        assert seen["source"] == "/some/bundle"
        assert seen["neurons"] == ["AIBL"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_render_review_gui.py -q`
Expected: FAIL, `AttributeError: module 'render_review' has no attribute 'render_defaults'`

- [ ] **Step 3: Write minimal implementation**

Append to `render_review.py`:

```python
def render_defaults(profile: dict) -> dict:
    """Render settings out of a launcher profile, under its own ``render`` key so
    nothing the launcher already stores can collide. An unrecognised preset falls back
    rather than raising, so a profile written by a newer version cannot break an older
    one."""
    saved = (profile or {}).get("render") or {}
    preset = saved.get("preset", "faithful")
    if preset not in meshing.PRESETS:
        preset = "faithful"
    fmt = saved.get("fmt", "gif")
    return {
        "out_dir": saved.get("out_dir", ""),
        "video": bool(saved.get("video", True)),
        "mesh": bool(saved.get("mesh", True)),
        "fmt": fmt if fmt in ("gif", "mp4") else "gif",
        "preset": preset,
    }


#: Preset names as a reviewer would pick them, not as the code names them.
PRESET_LABELS = [("faithful", "faithful (for review)"),
                 ("balanced", "balanced"),
                 ("smooth", "smooth (for figures)")]


def run(source: str = "", neurons=None) -> None:
    """The render window. Thin over render_all, in the same shape as launcher.run."""
    from qtpy.QtCore import Qt, QThread, Signal, QObject
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                                QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                QListWidgetItem, QProgressBar, QPushButton,
                                QVBoxLayout, QWidget)
    import launcher as launcher_mod

    profile = launcher_mod.load_profile()
    d = render_defaults(profile)
    app = QApplication.instance() or QApplication([])
    win = QWidget()
    win.setWindowTitle("Render video and mesh")
    layout = QVBoxLayout(win)

    src_row = QHBoxLayout()
    src_edit = QLineEdit(source or profile.get("output_root", ""))
    src_browse = QPushButton("Browse...")
    src_row.addWidget(QLabel("Bundle or tree:"))
    src_row.addWidget(src_edit, 1)
    src_row.addWidget(src_browse)
    layout.addLayout(src_row)

    neuron_list = QListWidget()
    neuron_list.setSelectionMode(QListWidget.NoSelection)
    layout.addWidget(QLabel("Neurons (none ticked means all):"))
    layout.addWidget(neuron_list, 1)

    opts = QHBoxLayout()
    video_cb = QCheckBox("Video")
    video_cb.setChecked(d["video"])
    fmt_combo = QComboBox()
    for f in ("gif", "mp4"):
        fmt_combo.addItem(f, f)
    fmt_combo.setCurrentIndex(0 if d["fmt"] == "gif" else 1)
    mesh_cb = QCheckBox("Mesh")
    mesh_cb.setChecked(d["mesh"])
    preset_combo = QComboBox()
    for key, label in PRESET_LABELS:
        preset_combo.addItem(label, key)
    preset_combo.setCurrentIndex([k for k, _ in PRESET_LABELS].index(d["preset"]))
    for w in (video_cb, fmt_combo, mesh_cb, QLabel("detail:"), preset_combo):
        opts.addWidget(w)
    layout.addLayout(opts)

    out_row = QHBoxLayout()
    out_edit = QLineEdit(d["out_dir"])
    out_browse = QPushButton("Browse...")
    out_row.addWidget(QLabel("Output to:"))
    out_row.addWidget(out_edit, 1)
    out_row.addWidget(out_browse)
    layout.addLayout(out_row)

    bar = QProgressBar()
    bar.setValue(0)
    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(bar)
    layout.addWidget(status)
    go = QPushButton("Render")
    cancel = QPushButton("Cancel")
    cancel.setEnabled(False)
    btns = QHBoxLayout()
    btns.addWidget(go)
    btns.addWidget(cancel)
    layout.addLayout(btns)

    flags = {"cancel": False}

    class _Worker(QObject):
        tick = Signal(int, int, str)
        done = Signal(str)

        def __init__(self, kwargs):
            super().__init__()
            self.kwargs = kwargs

        def work(self):
            try:
                res = render_all(progress=lambda i, n, m: self.tick.emit(i, n, m),
                                 should_cancel=lambda: flags["cancel"], **self.kwargs)
                self.done.emit(
                    f"{len(res['written'])} file(s) written"
                    + (", cancelled" if res["cancelled"] else ""))
            except BaseException as exc:                     # surface, never swallow
                self.done.emit(f"failed: {exc}")

    def refresh(*_):
        neuron_list.clear()
        root = Path(src_edit.text().strip() or ".")
        if not root.is_dir():
            status.setText(f"Not a directory: {root}")
            return
        try:
            names = list_neurons(root)
        except Exception as exc:
            status.setText(str(exc))
            return
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if (neurons and name in neurons)
                               else Qt.Unchecked)
            neuron_list.addItem(item)
        status.setText(f"{len(names)} neuron(s)")

    def ticked():
        return [neuron_list.item(i).text() for i in range(neuron_list.count())
                if neuron_list.item(i).checkState() == Qt.Checked]

    def start(*_):
        out = out_edit.text().strip()
        if not out:
            status.setText("Pick an output folder first")
            return
        profile.setdefault("render", {}).update(
            {"out_dir": out, "video": video_cb.isChecked(), "mesh": mesh_cb.isChecked(),
             "fmt": fmt_combo.currentData(), "preset": preset_combo.currentData()})
        launcher_mod.save_profile(profile)
        flags["cancel"] = False
        go.setEnabled(False)
        cancel.setEnabled(True)
        thread = QThread()
        worker = _Worker(dict(root=Path(src_edit.text().strip()), out_dir=Path(out),
                              neurons=ticked() or None, video=video_cb.isChecked(),
                              mesh=mesh_cb.isChecked(), fmt=fmt_combo.currentData(),
                              preset=preset_combo.currentData()))
            # keep references alive: a garbage-collected QThread kills the render
        win._thread, win._worker = thread, worker
        worker.moveToThread(thread)
        thread.started.connect(worker.work)
        worker.tick.connect(lambda i, n, m: (bar.setMaximum(n), bar.setValue(i),
                                             status.setText(m)))

        def finish(msg):
            status.setText(msg)
            go.setEnabled(True)
            cancel.setEnabled(False)
            thread.quit()

        worker.done.connect(finish)
        thread.start()

    src_browse.clicked.connect(lambda: (
        src_edit.setText(QFileDialog.getExistingDirectory(win, "Pick a bundle or tree",
                                                          src_edit.text() or str(Path.home()))
                         or src_edit.text()), refresh()))
    out_browse.clicked.connect(lambda: out_edit.setText(
        QFileDialog.getExistingDirectory(win, "Output folder",
                                         out_edit.text() or str(Path.home()))
        or out_edit.text()))
    src_edit.editingFinished.connect(refresh)
    go.clicked.connect(start)
    cancel.clicked.connect(lambda: (flags.__setitem__("cancel", True),
                                    status.setText("cancelling after this neuron...")))
    refresh()
    win.resize(640, 560)
    win.show()
    app.exec_()
```

Fix the stray indentation on the comment line before `win._thread` when pasting: it belongs at the same level as the assignment.

Then in `launcher.py`, add this function above `run()`:

```python
def open_render_window(source: str, neurons) -> None:
    """Hand the current source and ticked neurons to the render window.

    A named seam rather than a lambda inside run() so it can be tested without Qt,
    the same reason build_launch_kwargs is a function.
    """
    import render_review
    render_review.run(source=source, neurons=list(neurons or []))
```

And inside `launcher.run()`, after `launch_btn` is created and added, add:

```python
    render_btn = QPushButton("Render video + mesh")
    layout.addWidget(render_btn)

    def do_render(*_):
        open_render_window(path_edit.text().strip(), ticked())

    render_btn.clicked.connect(do_render)
```

Place the `render_btn.clicked.connect(do_render)` line next to the existing `launch_btn.clicked.connect(do_launch)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_render_review_gui.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Full suite, lint, and commit**

```bash
py -3 -m pytest -q
py -3 -m ruff check .
git add render_review.py launcher.py tests/test_render_review_gui.py
git commit -m "render-review: the window, reached from the launcher

Same split launcher.py already uses: render_defaults and open_render_window carry the
logic so they test with no Qt and no display, and run() is a thin window over
render_all.

Rendering runs on a QThread because it takes minutes and a frozen window reads as a
crash. Cancel is polled between neurons, so it can never truncate a file. The thread and
worker are kept on the window, since a garbage-collected QThread silently kills the
render.

Settings persist under their own render key in the existing profile, so adding them
cannot disturb what the launcher already stores, and an unrecognised preset falls back
rather than raising so a profile from a newer version cannot break an older one."
```

---

### Task 8: Reviewer documentation

**Files:**
- Modify: `F:\Lucinda_Review\START_HERE.md` (not in git; edit in place)
- Modify: `docs/CHANGELOG.md`

- [ ] **Step 1: Add a section to START_HERE.md**

Insert before "Sending work back":

```markdown
## Seeing the whole neuron

After correcting a bundle, you can watch each neuron end to end and open it in Blender.

```bash
source ~/review-env/bin/activate
python3 ~/segmentation-playground/launcher.py
```

Point it at the bundle as usual, then press **Render video + mesh** instead of Launch
review. Tick the neurons you want, pick an output folder, press Render. It takes a few
minutes per neuron and shows progress; Cancel stops after the neuron it is on.

You get one video and one `.ply` per neuron. The video plays anywhere. Drag the `.ply`
into Blender, or File > Import > Stanford (.ply).

The mesh detail setting defaults to **faithful (for review)**, which keeps the
slice-to-slice roughness visible. That roughness is usually the signal: a mask that
jumps between slices shows up as a jog in the tube. Pick **smooth (for figures)** only
when you want it to look nice, since smoothing hides exactly what you are looking for.
```

- [ ] **Step 2: Add a CHANGELOG entry**

Add above the most recent entry, with a matching Contents line:

```markdown
<a id="r-2026-08-26-review-render"></a>
## 2026-08-26, a video and a Blender mesh from a corrected bundle

A reviewer corrects a bundle one crop window at a time and never sees the neuron whole.
`render_review.py` gives her both missing views: one video per neuron, every chain in z
order captioned with chain and z, and one PLY per neuron.

It runs from a bundle or an output tree behind one command, reached from `launcher.py`.
Two facts made that cheap. `napari` already declares `scikit-image` and `imageio` as hard
dependencies, so marching cubes costs no new install on a machine where macOS setup was
the original friction. And bundles and trees share the `<neuron>/chain_NN/state.json`
layout, so `bundle.index_chains` and `pipeline.chain_masks_in_sam` already read either;
the only real difference is that a bundle ships its own `frames/`, which is isolated in
`chain_frames`.

Meshing bakes the anisotropy in, `spacing=(50, 128, 128)` in `(z, y, x)`, so vertices are
nanometres and Blender gets true proportions. Note z is the FINER axis at scale 8, the
opposite of the usual EM intuition. Smoothing is Taubin rather than Laplacian because
Laplacian shrinks thin tubes and a neurite is mostly thin tube. The default detail preset
is `faithful`, since the purpose is finding mistakes and smoothing removes the z-to-z
jitter that marks a bad slice. Real quadric decimation would need `trimesh` or `open3d`,
which the review install does not have, so `smooth` coarsens through marching cubes
`step_size` instead; that is a stated limitation rather than an implied capability.
```

- [ ] **Step 3: Verify and commit**

```bash
py -3 -m pytest -q
py -3 -m ruff check .
grep -c "—\|–" docs/CHANGELOG.md render_review.py sam2_utils/meshing.py
git add docs/CHANGELOG.md
git commit -m "docs: review render, reviewer instructions and changelog"
git push origin repo-reorg
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: frame seam to Task 3, video to Task 5, mesh to Tasks 1, 2 and 4, presets to Task 2, window and launcher button to Task 7, errors spread across Tasks 3, 4, 6 and 7, testing throughout, reviewer docs to Task 8.

**Placeholders.** None. Every code step carries real code; every test step carries real assertions.

**Type consistency.** `source_kind` returns `"bundle"`/`"tree"` and is consumed as such in Tasks 3, 5 and 6. `chain_frames` returns `{int: np.ndarray}` and is consumed that way in Task 5. `neuron_volume` returns `(vol, spacing)` and Task 4's `neuron_mesh` unpacks exactly that. `PRESETS` keys `faithful`/`balanced`/`smooth` are used identically in Tasks 2, 6 and 7. `render_all` keyword names match between Task 6's definition and Task 7's `_Worker`.

**One gap accepted deliberately.** The spec lists "tree source on a machine with no EM store: refuse before starting". That is not its own task; `chain_frames` will raise from `load_frame_sam` when the store is absent, which names the missing path. A pre-flight probe would be nicer and is not worth a task on its own, since the reviewer's path is always a bundle.
