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

    def test_progress_reports_neuron_progress_for_a_mesh_only_run(self, src, tmp_path):
        """video=False must not mean the progress callback goes silent: marching
        cubes has no incremental hook of its own, so render_all is the only thing
        that can still tell a caller which neuron is being worked on."""
        root, _calls = src
        events = []
        render_review.render_all(root, tmp_path / "out", None, video=False,
                                 progress=lambda i, n, m: events.append((i, n, m)))
        assert events, "a mesh-only run must still report progress"
        for neuron in ("AIBL", "AIBR"):
            assert any(neuron in m for _i, _n, m in events), (
                f"no progress message named {neuron}")
        totals = {n for _i, n, _m in events}
        assert totals == {2}, f"total should stay pinned to the neuron count, got {totals}"
        assert all(0 <= i <= 2 for i, _n, _m in events)

    def test_frame_level_progress_does_not_change_the_bars_scale(self, tmp_path, monkeypatch):
        """neuron_video reports its own progress(idx, total_frames, message) on a
        frame-count scale, unrelated to render_all's neuron-count scale. If that
        were forwarded straight through, the bar's total would flip between two
        unrelated numbers on every tick, exactly the jumping this design avoids.
        render_all instead pins current/total to the neuron scale for every call and
        carries the frame detail in the message text only."""
        root = tmp_path / "AIB"
        root.mkdir()
        (root / "bundle.json").write_text('{"schema_version": 1}', encoding="utf-8")

        def fake_video(r, neuron, out, progress=None, **k):
            if progress:
                progress(1, 50, f"{neuron} chain_00")
                progress(50, 50, f"{neuron} chain_00")
            pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(out).write_text("gif", encoding="utf-8")
            return pathlib.Path(out)

        monkeypatch.setattr(render_review, "neuron_video", fake_video)
        monkeypatch.setattr(render_review, "neuron_mesh", lambda *a, **k: None)
        monkeypatch.setattr(render_review, "list_neurons", lambda r: ["AIBL", "AIBR"])

        events = []
        render_review.render_all(root, tmp_path / "out", None, mesh=False,
                                 progress=lambda i, n, m: events.append((i, n, m)))
        totals = {n for _i, n, _m in events}
        assert totals == {2}, (
            f"the frame-level total (50) must never leak into the bar, got {totals}")
        assert any("chain_00" in m for _i, _n, m in events), (
            "frame-level detail should still reach the message text")


class TestCli:
    def test_source_and_out_are_required(self):
        with pytest.raises(SystemExit):
            render_review.main([])

    def test_default_flags_reach_render_all(self, src, tmp_path, monkeypatch):
        root, _c = src
        seen = {}
        monkeypatch.setattr(render_review, "render_all",
                            lambda *a, **k: seen.update(args=a, kwargs=k) or {"written": []})
        out = tmp_path / "o"
        render_review.main(["--source", str(root), "--out", str(out)])
        assert seen["args"] == (root, out, None)
        assert seen["kwargs"]["video"] is True
        assert seen["kwargs"]["mesh"] is True
        assert seen["kwargs"]["fmt"] == "gif"
        assert seen["kwargs"]["preset"] == "faithful"

    def test_every_flag_reaches_render_all_with_its_own_value(self, src, tmp_path, monkeypatch):
        """Every flag main accepts, set to a non-default value, positionals included.
        A typo such as preset=args.fmt, or a dropped --neurons passthrough, must fail
        this test."""
        root, _c = src
        seen = {}
        monkeypatch.setattr(render_review, "render_all",
                            lambda *a, **k: seen.update(args=a, kwargs=k) or {"written": []})
        out = tmp_path / "o"
        render_review.main(["--source", str(root), "--out", str(out),
                            "--neurons", "AIBL", "AIBR",
                            "--no-video", "--format", "mp4", "--detail", "smooth"])
        assert seen["args"] == (root, out, ["AIBL", "AIBR"])
        assert seen["kwargs"]["video"] is False
        assert seen["kwargs"]["mesh"] is True
        assert seen["kwargs"]["fmt"] == "mp4"
        assert seen["kwargs"]["preset"] == "smooth"

    def test_no_mesh_is_wired_independently_of_no_video(self, src, tmp_path, monkeypatch):
        root, _c = src
        seen = {}
        monkeypatch.setattr(render_review, "render_all",
                            lambda *a, **k: seen.update(args=a, kwargs=k) or {"written": []})
        render_review.main(["--source", str(root), "--out", str(tmp_path / "o"), "--no-mesh"])
        assert seen["kwargs"]["mesh"] is False
        assert seen["kwargs"]["video"] is True

    def test_detail_flag_reaches_preset_and_not_format(self, src, tmp_path, monkeypatch):
        """Guards specifically against a preset/fmt cross-wiring such as
        preset=args.fmt."""
        root, _c = src
        seen = {}
        monkeypatch.setattr(render_review, "render_all",
                            lambda *a, **k: seen.update(args=a, kwargs=k) or {"written": []})
        render_review.main(["--source", str(root), "--out", str(tmp_path / "o"),
                            "--detail", "balanced", "--format", "gif"])
        assert seen["kwargs"]["preset"] == "balanced"
        assert seen["kwargs"]["fmt"] == "gif"
        assert seen["kwargs"]["preset"] != seen["kwargs"]["fmt"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
