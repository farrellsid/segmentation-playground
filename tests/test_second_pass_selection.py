import importlib

prop = importlib.import_module("pipeline.propagate")


def test_select_flags_empty_foreign_and_not_own_contained():
    records = [
        {"z": 10, "own_contained": True, "n_foreign": 0, "empty": False},
        {"z": 11, "own_contained": True, "n_foreign": 0, "empty": True},   # dropout
        {"z": 12, "own_contained": True, "n_foreign": 2, "empty": False},  # foreign
        {"z": 13, "own_contained": False, "n_foreign": 0, "empty": False}, # lost own node
    ]
    assert prop.select_second_pass_frames(records) == {11, 12, 13}


def test_select_returns_empty_set_when_nothing_flagged():
    records = [{"z": 10, "own_contained": True, "n_foreign": 0, "empty": False}]
    assert prop.select_second_pass_frames(records) == set()


def test_find_neighbour_picks_nearest_unflagged():
    areas = {10: 100.0, 11: 100.0, 12: 5.0, 13: 5.0, 14: 100.0}
    flagged = {12, 13}
    assert prop.find_second_pass_neighbour(12, flagged, areas, min_area_ratio=0.5) == 11


def test_find_neighbour_skips_nucleus_capture_sized_candidate():
    # z=9 is the naive-nearest unflagged neighbour to z=10, but its area (3.0) is far
    # below the chain's unflagged median (100.0), simulating a nucleus-capture mask that
    # own_contained/n_foreign/empty all missed. The guard must skip it for z=8 instead.
    areas = {8: 100.0, 9: 3.0, 10: 5.0, 13: 100.0}
    flagged = {10}
    assert prop.find_second_pass_neighbour(10, flagged, areas, min_area_ratio=0.5) == 8


def test_find_neighbour_returns_none_when_nothing_unflagged():
    areas = {5: 5.0, 6: 5.0}
    flagged = {5, 6}
    assert prop.find_second_pass_neighbour(5, flagged, areas, min_area_ratio=0.5) is None
