"""build_layer_specs / build_mesh_specs / save_package / load_package: per-neuron
napari layer specs from a tree, bundle, or saved package, 2D labelmap, 3D mesh,
and the VAST-style package round-trip respectively.

Pure, no napari import, following the same split gui.py already uses: the logic
(_peek_chain there, these functions here) is tested directly; the real napari
wiring around it is not.

    py -3 -m pytest tests/test_view_neurons.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import view_neurons
from eval.groundtruth import _read_label_png


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


def _two_specs(*, overlap: bool = False):
    """Two small, hand-built layer specs in the exact shape build_layer_specs()
    returns, for testing save_package/load_package without needing a real tree
    or neuron_volume() at all. With overlap=True, the second neuron's volume
    occupies the exact same block as the first's, so the overlap policy
    (last-in-list wins) is directly testable. With overlap=False, the second
    neuron is offset two voxels along Z and one along X: both have z-depth 2,
    so z=0..1 is AIAL-only and z=2..3 is AIAR-only, no shared z-slice; the
    1-voxel X offset also makes the union canvas one voxel wider than either
    neuron's own 3-voxel block, so AIAL's own z=0 slice has a real background
    column at the far edge, not just foreground wall to wall. That's what lets
    test_saved_png_is_readable_by_the_existing_vast_gt_reader genuinely check
    both "background is 0" and "AIAL's id is present" on one real slice,
    instead of a slice that happens to have no background pixel at all."""
    spacing = (50.0, 128.0, 128.0)
    vol_a = np.ones((2, 3, 3), dtype=np.uint16)
    data_a = vol_a * np.uint16(1)
    spec_a = {"name": "AIAL", "data": data_a,
             "translate": (0.0, 0.0, 0.0), "scale": spacing}

    vol_b = np.ones((2, 3, 3), dtype=np.uint16)
    data_b = vol_b * np.uint16(2)
    if overlap:
        # same origin as AIAL, same shape, so every voxel overlaps
        spec_b = {"name": "AIAR", "data": data_b,
                 "translate": (0.0, 0.0, 0.0), "scale": spacing}
    else:
        # z: offset by 2 voxels (real-world 2*50=100nm), so AIAL occupies
        # z=0..1 and AIAR occupies z=2..3, no shared z-slice.
        # x: offset by 1 voxel (real-world 1*128=128nm), so the union canvas
        # is 4 voxels wide in x while each neuron's own block is only 3,
        # leaving a real background column in every slice.
        spec_b = {"name": "AIAR", "data": data_b,
                 "translate": (100.0, 0.0, 128.0), "scale": spacing}
    return [spec_a, spec_b]


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


def test_save_and_load_package_round_trips_names_and_placement(tmp_path):
    specs = _two_specs()

    out_dir = view_neurons.save_package(
        specs, tmp_path / "pkg", built_from={"tree": "fake-root", "neurons": ["AIAL", "AIAR"]})
    loaded = view_neurons.load_package(out_dir)

    assert [s["name"] for s in loaded] == ["AIAL", "AIAR"]
    for spec in loaded:
        assert spec["scale"] == (50.0, 128.0, 128.0)
        assert np.unique(spec["data"]).tolist()[0] == 0, "background must still be 0"


def test_loaded_data_matches_each_neurons_own_id_and_shape(tmp_path):
    specs = _two_specs()

    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    loaded = view_neurons.load_package(out_dir)

    aial = next(s for s in loaded if s["name"] == "AIAL")
    aiar = next(s for s in loaded if s["name"] == "AIAR")
    assert set(np.unique(aial["data"]).tolist()) == {0, 1}
    assert set(np.unique(aiar["data"]).tolist()) == {0, 2}
    # both neurons' layers share the package's one full shape
    assert aial["data"].shape == aiar["data"].shape


def test_overlap_uses_the_later_spec_in_the_list(tmp_path):
    specs = _two_specs(overlap=True)

    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})
    loaded = view_neurons.load_package(out_dir)

    aial = next(s for s in loaded if s["name"] == "AIAL")
    aiar = next(s for s in loaded if s["name"] == "AIAR")
    # Both specs occupied the exact same voxels. AIAR was second in the input
    # list, so it must have won every one of them: AIAL's reconstructed layer
    # has nothing left (id 1 never appears), AIAR's has the full block (id 2
    # everywhere the original block was).
    assert not np.any(aial["data"] == 1), "AIAL must have lost every overlapping voxel"
    assert np.sum(aiar["data"] == 2) == 2 * 3 * 3, "AIAR must have kept the whole block"


def test_a_later_neurons_own_background_does_not_erase_an_earlier_neurons_foreground(tmp_path):
    # A literal "later entry wins on every pixel in its bounding box" policy
    # would let a neuron's own empty crop margin blank out a completely
    # unrelated neuron's real mask content, just because the two bounding
    # boxes happen to overlap, even when the two neurons' actual shapes never
    # touch. save_package must overwrite only on the later entry's own
    # foreground (non-zero) pixels, never on its background ones. AIAR here
    # shares AIAL's exact origin and block shape (so every voxel's bounding
    # box overlaps), but AIAR has one background voxel at (0, 0, 0), exactly
    # where AIAL has foreground.
    spacing = (50.0, 128.0, 128.0)
    data_a = np.ones((2, 3, 3), dtype=np.uint16) * np.uint16(1)
    spec_a = {"name": "AIAL", "data": data_a,
             "translate": (0.0, 0.0, 0.0), "scale": spacing}

    data_b = np.ones((2, 3, 3), dtype=np.uint16) * np.uint16(2)
    data_b[0, 0, 0] = 0  # AIAR's own background voxel, same bbox slot as AIAL's foreground
    spec_b = {"name": "AIAR", "data": data_b,
             "translate": (0.0, 0.0, 0.0), "scale": spacing}

    out_dir = view_neurons.save_package([spec_a, spec_b], tmp_path / "pkg", built_from={})
    loaded = view_neurons.load_package(out_dir)

    aial = next(s for s in loaded if s["name"] == "AIAL")
    aiar = next(s for s in loaded if s["name"] == "AIAR")
    assert aial["data"][0, 0, 0] == 1, (
        "AIAR's own background voxel must not erase AIAL's real foreground "
        "underneath it; a literal bounding-box overwrite would blank this to 0")
    # Control: AIAR's genuine foreground elsewhere in the same shared bbox
    # still wins, so this isn't passing merely because nothing overwrote.
    assert aiar["data"][0, 1, 1] == 2, "AIAR's real foreground must still win where it has one"
    assert aiar["data"][0, 0, 0] == 0, "AIAR has no claim of its own at this voxel"


def test_saved_png_is_readable_by_the_existing_vast_gt_reader(tmp_path):
    specs = _two_specs()
    out_dir = view_neurons.save_package(specs, tmp_path / "pkg", built_from={})

    # Real interop check, not just round-tripping through our own loader: this
    # repo's existing VAST-GT reader (eval/groundtruth.py, unrelated feature,
    # unmodified) must be able to read a package PNG directly.
    arr = _read_label_png(out_dir / "labels" / "000000.png")
    assert arr.dtype == np.uint16
    assert set(np.unique(arr).tolist()) == {0, 1}, (
        "z-slice 0 should hold only background (0) and AIAL's id (1), "
        "AIAR starts one slice later in this fixture's un-overlapped layout")


def test_main_rejects_package_combined_with_tree(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(tmp_path),
        "--tree", "fake-root", "--neurons", "AIAL"])
    with pytest.raises(SystemExit, match="cannot be combined with --tree"):
        view_neurons.main()


def test_main_rejects_package_combined_with_mesh(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(tmp_path), "--mesh"])
    with pytest.raises(SystemExit, match="cannot be combined with --mesh"):
        view_neurons.main()


def test_main_rejects_package_combined_with_save_package(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "view_neurons.py", "--package", str(tmp_path),
        "--save-package", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="only applies when building from"):
        view_neurons.main()


def test_main_requires_package_or_tree_and_neurons(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view_neurons.py"])
    with pytest.raises(SystemExit, match="need either --package"):
        view_neurons.main()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
