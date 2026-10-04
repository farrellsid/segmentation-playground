"""build_layer_specs / build_mesh_specs: per-neuron napari layer specs from a
tree or bundle root, 2D labelmap and 3D mesh respectively.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, these two functions here) is tested directly; the real
napari wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import view_neurons


def _fake_neuron_volume(volumes):
    def fake(root, neuron, **kw):
        return volumes[neuron]
    return fake


def _asymmetric_box_volume():
    """A solid box, a different voxel extent on each axis (z=2, y=4, x=6), with a
    1-voxel margin on every side so marching cubes produces a single closed
    surface. The asymmetry matters: with spacing=(50, 128, 128), the real-world
    span ends up strictly x > y > z, so a test can tell a correctly axis-reversed
    vertex array (z, y, x) apart from volume_to_mesh's native (x, y, z) one by
    which column has the largest span, not by exact coordinates."""
    vol = np.zeros((6, 8, 10), dtype=np.uint8)
    vol[1:3, 1:5, 1:7] = 1
    return vol


def test_assigns_distinct_ids_in_selection_order(monkeypatch):
    # A real mask volume is mostly background (0) with a foreground blob (1), not
    # all-foreground: the fixture needs a 0 voxel for {0, N} to be a reachable
    # unique-value set after the per-neuron ID multiply.
    vol_a = np.ones((2, 3, 3), dtype=np.uint8)
    vol_a[0, 0, 0] = 0
    vol_b = np.ones((2, 3, 3), dtype=np.uint8)
    vol_b[0, 0, 0] = 0
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol_a, (100, 0, 0), (50.0, 128.0, 128.0)),
        "AIAR": (vol_b, (100, 0, 0), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["AIAL", "AIAR"])

    assert [s["name"] for s in specs] == ["AIAL", "AIAR"]
    assert set(np.unique(specs[0]["data"]).tolist()) == {0, 1}, (
        "AIAL (selected first) must be label ID 1")
    assert set(np.unique(specs[1]["data"]).tolist()) == {0, 2}, (
        "AIAR (selected second) must be label ID 2, not the same ID as AIAL")


def test_translate_is_origin_converted_to_real_world_nm(monkeypatch):
    vol = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (10, 20, 30), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["AIAL"])

    assert specs[0]["translate"] == (500.0, 2560.0, 3840.0), (
        "translate must be origin * spacing elementwise (z=10*50, y=20*128, x=30*128), "
        "in real-world nm, not raw voxel indices, since scale=spacing is set too")
    assert specs[0]["scale"] == (50.0, 128.0, 128.0)


def test_a_neuron_with_no_masks_is_skipped_not_an_empty_layer(monkeypatch):
    empty = np.zeros((0, 0, 0), dtype=np.uint8)
    has_masks = np.ones((1, 1, 1), dtype=np.uint8)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "EMPTYN": (empty, (0, 0, 0), (50.0, 128.0, 128.0)),
        "AIAL": (has_masks, (0, 0, 0), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_layer_specs("fake-root", ["EMPTYN", "AIAL"])

    assert [s["name"] for s in specs] == ["AIAL"], (
        "a neuron with no masks must be skipped entirely, not added as an empty layer")


def test_id_volume_does_not_overflow_for_many_neurons(monkeypatch):
    # 300 neurons would overflow uint8 (max 255) if the ID multiply stayed narrow.
    volumes = {f"N{i}": (np.ones((1, 1, 1), dtype=np.uint8), (0, 0, 0),
                         (50.0, 128.0, 128.0)) for i in range(300)}
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume",
                        _fake_neuron_volume(volumes))

    specs = view_neurons.build_layer_specs("fake-root", list(volumes))

    assert specs[-1]["data"].max() == 300, (
        "the 300th neuron's label ID must survive as 300, not wrap around a narrow dtype")


def test_mesh_vertices_are_reversed_back_to_z_y_x_order(monkeypatch):
    vol = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume",
                        _fake_neuron_volume({"AIAL": (vol, (0, 0, 0), spacing)}))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL"])

    assert len(specs) == 1
    verts = specs[0]["vertices"]
    z_span, y_span, x_span = (np.ptp(verts[:, i]) for i in range(3))
    assert x_span > y_span > z_span, (
        f"expected x-span > y-span > z-span (the fixture box is widest in x, "
        f"narrowest in z), got z={z_span:.1f} y={y_span:.1f} x={x_span:.1f}; this "
        f"fails if vertices are still in volume_to_mesh's native (x, y, z) order")


def test_mesh_translate_is_origin_converted_to_real_world_nm(monkeypatch):
    vol = _asymmetric_box_volume()
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol, (10, 20, 30), (50.0, 128.0, 128.0)),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL"])

    assert specs[0]["translate"] == (500.0, 2560.0, 3840.0), (
        "translate must be origin * spacing elementwise, in real-world nm, no scale "
        "kwarg is used for mesh layers since volume_to_mesh already bakes spacing in")


def test_each_neuron_mesh_gets_a_distinct_color_from_the_existing_palette(monkeypatch):
    vol_a = _asymmetric_box_volume()
    vol_b = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "AIAL": (vol_a, (0, 0, 0), spacing),
        "AIAR": (vol_b, (0, 0, 0), spacing),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["AIAL", "AIAR"])

    expected_a = view_neurons.video_viz._color_for(1)
    expected_b = view_neurons.video_viz._color_for(2)
    assert not np.allclose(expected_a, expected_b), (
        "fixture sanity check: the palette must actually give IDs 1 and 2 different "
        "colors, or this test cannot tell a real bug from a coincidence")
    assert np.allclose(specs[0]["vertex_colors"][0], expected_a)
    assert np.allclose(specs[1]["vertex_colors"][0], expected_b)
    assert specs[0]["vertex_colors"].shape == (len(specs[0]["vertices"]), 3)


def test_a_neuron_with_no_masks_is_skipped_not_an_empty_mesh(monkeypatch):
    empty = np.zeros((0, 0, 0), dtype=np.uint8)
    has_masks = _asymmetric_box_volume()
    spacing = (50.0, 128.0, 128.0)
    monkeypatch.setattr(view_neurons.render_review, "neuron_volume", _fake_neuron_volume({
        "EMPTYN": (empty, (0, 0, 0), spacing),
        "AIAL": (has_masks, (0, 0, 0), spacing),
    }))

    specs = view_neurons.build_mesh_specs("fake-root", ["EMPTYN", "AIAL"])

    assert [s["name"] for s in specs] == ["AIAL mesh"], (
        "a neuron with no masks must be skipped entirely, not added as an empty mesh")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
