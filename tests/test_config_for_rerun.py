"""A re-run must reproduce the chain, not the GUI's defaults.

gui._recrop_to_window used to build a fresh PipelineConfig from ReviewContext's defaults,
so recropping a chain silently changed how it was segmented. Measured on a real chain in
AIA_for_lucinda: produced with backend "sam3" and seed_negatives True, re-run as "sam2"
with no negatives. A chain from a mask-seed tree lost its mask seed the same way. Nothing
in the GUI showed it, so the cost compounded with every recrop.

    py -3 -m pytest tests/test_config_for_rerun.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline
from pipeline.config import PipelineConfig


def _recorded():
    """What a chain from the sam3 tier-2 tree actually carries, paths included."""
    return PipelineConfig(
        backend="sam3", seed_negatives=True, seed_mask=True, model_size="large",
        chain_crop=True, multimask_generous=True,
        output_root=pathlib.Path("/scratch/narval/output_masks/target_sam3"),
        frames_root=pathlib.Path("/localscratch/12345/frames"))


def _base(tmp_path):
    """What the machine doing the re-run knows: its own paths, its own defaults."""
    return PipelineConfig(output_root=tmp_path / "out", frames_root=tmp_path / "frames")


class TestModelSettingsAreInherited:
    def test_the_backend_comes_from_the_chain_not_the_gui(self, tmp_path):
        cfg = pipeline.config_for_rerun(_recorded(), base=_base(tmp_path))
        assert cfg.backend == "sam3", "the whole bug: sam3 chains came back as sam2"

    def test_the_seeding_comes_from_the_chain(self, tmp_path):
        cfg = pipeline.config_for_rerun(_recorded(), base=_base(tmp_path))
        assert cfg.seed_negatives is True and cfg.seed_mask is True

    def test_settings_the_caller_never_names_still_carry_over(self, tmp_path):
        """Inherit by default, exclude by exception. The failure here is DROPPING a
        setting, so a field nobody thought to list must not revert to a default."""
        cfg = pipeline.config_for_rerun(_recorded(), base=_base(tmp_path))
        assert cfg.multimask_generous is True


class TestPathsComeFromThisMachine:
    def test_the_recorded_paths_are_not_reused(self, tmp_path):
        """A state.json can come from a bundle or from the cluster, so its paths name
        directories that do not exist here."""
        cfg = pipeline.config_for_rerun(_recorded(), base=_base(tmp_path))
        assert cfg.output_root == tmp_path / "out"
        assert cfg.frames_root == tmp_path / "frames"
        assert "narval" not in str(cfg.output_root)


class TestOverridesAndEdges:
    def test_an_explicit_override_wins_over_both(self, tmp_path):
        cfg = pipeline.config_for_rerun(_recorded(), base=_base(tmp_path),
                                        chain_crop_fallback=False, seed_negatives=False)
        assert cfg.chain_crop_fallback is False and cfg.seed_negatives is False

    def test_no_recorded_config_falls_back_to_the_base(self, tmp_path):
        """A chain with no state.json has nothing to inherit, and the caller's own
        defaults are the only answer available."""
        cfg = pipeline.config_for_rerun(None, base=_base(tmp_path))
        assert cfg.backend == "sam2" and cfg.output_root == tmp_path / "out"

    def test_neither_input_is_mutated(self, tmp_path):
        recorded, base = _recorded(), _base(tmp_path)
        pipeline.config_for_rerun(recorded, base=base, backend="sam2")
        assert recorded.backend == "sam3"
        # Compare the Path, not its string: on Windows a posix-looking literal comes back
        # with backslashes, which made this assertion fail for a reason unrelated to
        # mutation.
        assert recorded.output_root == pathlib.Path("/scratch/narval/output_masks/target_sam3")
        assert base.backend == "sam2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
