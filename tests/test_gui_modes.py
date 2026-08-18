"""Review mode omits every model control. Full mode is exactly today's surface.

Tested against the pure mode functions, not a live viewer, so this runs with no
napari and no torch (the same approach as tests/test_gui_box.py).
"""
import pytest

import gui


def test_modes_are_the_two_expected():
    assert gui.UI_MODES == (gui.UI_MODE_REVIEW, gui.UI_MODE_FULL)


def test_full_mode_includes_every_panel():
    assert gui.panels_for_mode(gui.UI_MODE_FULL) == gui.PANELS


def test_review_mode_drops_only_the_model_panel():
    panels = gui.panels_for_mode(gui.UI_MODE_REVIEW)
    assert "model" not in panels
    assert set(panels) == set(gui.PANELS) - {"model"}


def test_review_mode_keeps_drawing_and_verdict():
    panels = gui.panels_for_mode(gui.UI_MODE_REVIEW)
    assert "drawing" in panels and "verdict" in panels and "navigation" in panels


def test_full_mode_registers_every_key():
    assert gui.keys_for_mode(gui.UI_MODE_FULL) == gui.ALL_KEYS


def test_review_mode_registers_no_model_key():
    keys = gui.keys_for_mode(gui.UI_MODE_REVIEW)
    assert keys & gui.MODEL_KEYS == frozenset()
    assert keys == gui.ALL_KEYS - gui.MODEL_KEYS


def test_review_mode_keeps_lasso_save_and_verdict_keys():
    """Lasso is a drawing tool, not a model prompt, so it must survive."""
    keys = gui.keys_for_mode(gui.UI_MODE_REVIEW)
    for k in ("l", "s", "z", "w", "o", "a", "x", ",", ".", "Control-Z"):
        assert k in keys


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        gui.panels_for_mode("nonsense")
    with pytest.raises(ValueError):
        gui.keys_for_mode("nonsense")


def test_model_keys_are_the_documented_seven():
    assert gui.MODEL_KEYS == frozenset({"p", "n", "b", "r", "g", "c", "f"})
