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
