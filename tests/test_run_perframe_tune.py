import json

import numpy as np
import pandas as pd

import pipeline
import run_perframe
from eval import merge_metric


class FakeAMG:
    """Stand-in for SAM2AutomaticMaskGenerator: ignores its build params entirely, so this
    exercises the --tune loop's grid-building and bookkeeping without running real SAM2."""
    def __init__(self, masks):
        self._masks = masks

    def generate(self, image):
        return [{"segmentation": m, "area": int(m.sum()), "predicted_iou": 0.9,
                 "stability_score": 0.95} for m in self._masks]


def _disk(cx, cy, r, shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_tune_grid_combos_is_a_cartesian_product():
    combos = run_perframe._tune_grid_combos({"a": (1, 2), "b": (3,)})
    assert combos == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_tune_loop_builds_grid_writes_trials_and_logs_winner(tmp_path, monkeypatch):
    """Dry structural check of --tune (no real SAM2/torch): fakes the model build, the AMG
    build (same two masks for every grid point, since the point here is the loop's plumbing,
    not real AMG output), the frame loader, and the node table, then asserts the grid was
    fully walked, trials.csv has one row per grid point, and the winner's montages/config and
    a log row landed. The real experiments log is untouched: _EXPERIMENT_LOG_PATH is
    monkeypatched to a scratch file so this test cannot corrupt the committed doc.
    """
    frame = np.full((40, 40, 3), 128, np.uint8)
    node_table = pd.DataFrame({
        "node_id": ["a", "b"], "cell_name": ["AVAL", "AVAR"],
        "z": [1400, 1400], "x_tif": [80.0, 240.0], "y_tif": [80.0, 240.0],
    })
    monkeypatch.setattr(pipeline, "load_frame_sam", lambda z, scale: (frame, (40, 40)))
    monkeypatch.setattr(merge_metric, "load_node_table", lambda: node_table)
    monkeypatch.setattr(run_perframe, "_build_sam2_model",
                        lambda model_size: ("fake-model", "cpu"))

    build_calls = []

    def fake_build_amg(sam2_model, **params):
        build_calls.append(params)
        return FakeAMG([_disk(10, 10, 5), _disk(30, 30, 5)])

    monkeypatch.setattr(run_perframe, "build_amg", fake_build_amg)

    log_path = tmp_path / "scratch-experiments.md"
    monkeypatch.setattr(run_perframe, "_EXPERIMENT_LOG_PATH", log_path)

    out_dir = tmp_path / "tune_out"
    grid = {"pred_iou_thresh": [0.7, 0.8], "stability_score_thresh": [0.9],
            "points_per_side": [32]}
    args = run_perframe._parse([
        "--tune", "--frames", "1400", "--scale", "8", "--model-size", "tiny",
        "--out", str(out_dir), "--tune-grid", json.dumps(grid),
    ])

    run_perframe._run_tune(args)

    # the 2-point grid override was walked exactly once each, one AMG built per trial plus
    # one rebuild for the winner.
    assert len(build_calls) == 3
    assert {c["pred_iou_thresh"] for c in build_calls[:2]} == {0.7, 0.8}

    trials = pd.read_csv(out_dir / "trials.csv")
    assert len(trials) == 2
    assert set(trials["pred_iou_thresh"]) == {0.7, 0.8}
    assert "mean_objective" in trials.columns
    assert "per_frame_scores" in trials.columns
    # per_frame_scores round-trips as JSON with one entry (one cached frame) per trial.
    assert len(json.loads(trials.iloc[0]["per_frame_scores"])) == 1

    assert (out_dir / "config.json").exists()
    assert (out_dir / "scores.csv").exists()
    assert (out_dir / "montages" / "1400.png").exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "tuned: best" in log_text
    assert "objective can be gamed" in log_text
