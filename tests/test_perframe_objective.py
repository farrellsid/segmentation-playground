from eval.perframe_score import objective


def test_objective_rewards_coverage_penalises_bleed_and_overlap():
    good = {"own_coverage": 1.0, "foreign_frame_rate": 0.0, "total_foreign": 0,
            "overlap_fraction": 0.0, "mean_boundary_on_membrane": 0.9, "spanning_rate": 0.0}
    bleed = {**good, "total_foreign": 20, "foreign_frame_rate": 0.5, "spanning_rate": 0.4}
    undercover = {**good, "own_coverage": 0.3}
    assert objective(good) > objective(bleed)
    assert objective(good) > objective(undercover)
    # None membrane fields must not crash
    nomem = {**good, "mean_boundary_on_membrane": None, "spanning_rate": None}
    assert isinstance(objective(nomem), float)
