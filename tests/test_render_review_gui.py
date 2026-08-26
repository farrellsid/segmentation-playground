"""Settings the render window remembers, tested without a display.

Same split `launcher.py` already uses: pure functions carry the logic and `run()` is a
thin window over them, so none of this needs Qt or a screen. That split is also why
`tests/test_launcher_config.py` can exist at all.

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
        assert d["out_dir"] == ""

    def test_saved_render_settings_are_read_back(self):
        d = render_review.render_defaults(
            {"render": {"fmt": "mp4", "preset": "smooth", "mesh": False,
                        "video": False, "out_dir": "/tmp/x"}})
        assert d["fmt"] == "mp4" and d["preset"] == "smooth"
        assert d["mesh"] is False and d["video"] is False
        assert d["out_dir"] == "/tmp/x"

    def test_settings_live_under_a_render_key_so_launcher_keys_are_untouched(self):
        """The launcher owns this profile file. Adding render settings must not
        disturb the keys it already stores."""
        prof = {"output_root": "/keep", "reviewer": "L", "neurons": ["AIBL"],
                "render": {"fmt": "mp4"}}
        d = render_review.render_defaults(prof)
        assert d["fmt"] == "mp4"
        assert prof["output_root"] == "/keep"
        assert prof["reviewer"] == "L"
        assert prof["neurons"] == ["AIBL"]

    def test_an_unknown_preset_in_the_profile_falls_back(self):
        """A profile written by a newer version must not break an older one."""
        d = render_review.render_defaults({"render": {"preset": "ultra"}})
        assert d["preset"] in meshing.PRESETS

    def test_an_unknown_format_in_the_profile_falls_back(self):
        d = render_review.render_defaults({"render": {"fmt": "webm"}})
        assert d["fmt"] in ("gif", "mp4")

    def test_a_missing_profile_does_not_raise(self):
        assert render_review.render_defaults(None)["preset"] == "faithful"

    def test_every_preset_key_is_offered_by_a_label(self):
        """The window offers presets by what they are for, not by their code name.
        Every preset must still be reachable, or one becomes dead configuration."""
        offered = {key for key, _label in render_review.PRESET_LABELS}
        assert offered == set(meshing.PRESETS)

    def test_preset_labels_say_what_they_are_for(self):
        labels = dict(render_review.PRESET_LABELS)
        assert "review" in labels["faithful"].lower()
        assert "figure" in labels["smooth"].lower()


class TestLauncherIntegration:
    def test_launcher_exposes_a_render_entry_point(self):
        """A named seam rather than a closure inside run(), for the same reason
        build_launch_kwargs is a function: it can be tested without Qt."""
        import launcher
        assert hasattr(launcher, "open_render_window")

    def test_launcher_hands_its_source_and_neurons_across(self, monkeypatch):
        """The reviewer picks a bundle once, in the launcher. The render window must
        inherit that choice rather than asking again."""
        import launcher
        seen = {}
        monkeypatch.setattr(render_review, "run",
                            lambda source="", neurons=None: seen.update(
                                source=source, neurons=neurons))
        launcher.open_render_window("/some/bundle", ["AIBL"])
        assert seen["source"] == "/some/bundle"
        assert seen["neurons"] == ["AIBL"]

    def test_no_ticked_neurons_passes_an_empty_list_not_none(self, monkeypatch):
        """render_all treats a falsy neuron list as 'all of them', so an empty tick
        list must not be confused with an explicit selection."""
        import launcher
        seen = {}
        monkeypatch.setattr(render_review, "run",
                            lambda source="", neurons=None: seen.update(neurons=neurons))
        launcher.open_render_window("/some/bundle", None)
        assert seen["neurons"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
