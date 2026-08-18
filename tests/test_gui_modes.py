"""Review mode omits every model control. Full mode is exactly today's surface.

Most tests below exercise the pure mode functions, not a live viewer, so this
runs with no napari and no torch (the same approach as tests/test_gui_box.py).

The last two behaviours (_bind_keys, _build_widgets) are exercised for real,
against a stand-in ReviewGUI (object.__new__, the same tactic tests/test_gui_lasso.py
uses to skip __init__'s real chain/viewer/torch requirements). Neither napari nor
Qt is a declared test dependency (see pyproject.toml's `test` extra: pytest, scipy,
scikit-image only), so magicgui.widgets and qtpy.QtWidgets are swapped for
headless stand-ins before _build_widgets runs; gui.py imports both lazily inside
each method, so the swap is invisible to the production code under test.
"""
import sys
import types

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


# ---------------------------------------------------------------------------
# Stand-ins for the real collaborators _bind_keys / _build_widgets touch.
#
# These do not test panels_for_mode / keys_for_mode again (the tests above
# already pin those); they test that ReviewGUI's own methods actually USE
# them the way the docstrings claim. Two real mutations pass every test above
# while breaking this at runtime: (a) skipping _build_model_panel's call in
# review mode (AttributeError the first time a review-mode session reads
# self._prompt_mode or self._grow_spin), and (b) dropping a single @_bind(...)
# block from _bind_keys (that key silently never fires, in every mode).
# ---------------------------------------------------------------------------

class _StubWindow:
    """Stand-in for napari's ViewerWindow: only add_dock_widget is touched."""

    def __init__(self):
        self.docked = []

    def add_dock_widget(self, widget, area=None, name=None):
        self.docked.append((widget, area, name))


class _StubViewer:
    """Stand-in for napari.Viewer: records every key _bind_keys registers.

    Real Viewer.bind_key(key, overwrite=True) returns a decorator; _bind_keys
    calls it as `v.bind_key(key, overwrite=True)(fn)`, so this mimics that
    shape exactly, but only records the key. It never imports napari.
    """

    def __init__(self):
        self.bound_keys = []
        self.window = _StubWindow()

    def bind_key(self, key, overwrite=False):
        self.bound_keys.append(key)

        def _decorator(fn):
            return fn

        return _decorator


class _FakeQueue:
    """Stand-in for gui.ReviewQueue: enough for _populate_selectors to run
    against an empty picker (no chain is open on this stand-in GUI)."""

    def pending(self, include_in_review=True):
        return []

    def all_chains(self):
        return []

    def chain_status(self, neuron, idx):
        return "flagged"


class _FakeSignal:
    """Stand-in for magicgui's psygnal-backed `.changed`: connect is a no-op,
    matching that _build_widgets never fires these callbacks, only wires them."""

    def connect(self, fn):
        pass


class _FakeWidget:
    """Stand-in for every magicgui.widgets class the panel builders construct
    (PushButton, ComboBox, Label, LineEdit, FloatSpinBox, CheckBox, SpinBox).
    Accepts any kwargs the real widgets take (label, text, choices, min, max,
    step, ...) and exposes `.value` / `.changed.connect` / `.native`, which is
    everything the builders read or call on a freshly-built widget."""

    def __init__(self, *, value=None, **kwargs):
        self.value = value
        self.changed = _FakeSignal()
        self.native = object()
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeContainer(_FakeWidget):
    """Stand-in for magicgui.widgets.Container, built from the panels' widgets."""

    def __init__(self, *, widgets=None, **kwargs):
        super().__init__(**kwargs)
        self.widgets = list(widgets or [])


class _FakeScrollArea:
    """Stand-in for qtpy.QtWidgets.QScrollArea, the dock's outer wrapper."""

    def __init__(self, *args, **kwargs):
        pass

    def setWidgetResizable(self, *args, **kwargs):
        pass

    def setWidget(self, *args, **kwargs):
        pass


def _install_fake_gui_toolkit(monkeypatch):
    """Seed sys.modules with headless magicgui.widgets / qtpy.QtWidgets stand-ins.

    gui.py's panel builders do `from magicgui.widgets import ...` (and
    _build_widgets does the same for qtpy.QtWidgets.QScrollArea) as LOCAL
    imports inside each method, precisely so `import gui` itself stays
    napari/Qt-free. Pre-seeding the dotted module names in sys.modules makes
    that local import resolve to these stand-ins instead of doing a real
    import, so the real production method bodies run unmodified with no PyQt
    involved at all.
    """
    widgets_mod = types.ModuleType("magicgui.widgets")
    for name in ("PushButton", "ComboBox", "Label", "LineEdit",
                 "FloatSpinBox", "CheckBox", "SpinBox"):
        setattr(widgets_mod, name, _FakeWidget)
    widgets_mod.Container = _FakeContainer
    monkeypatch.setitem(sys.modules, "magicgui.widgets", widgets_mod)

    qtwidgets_mod = types.ModuleType("qtpy.QtWidgets")
    qtwidgets_mod.QScrollArea = _FakeScrollArea
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", qtwidgets_mod)


def _stub_gui(ui_mode):
    """A ReviewGUI stand-in with only the attributes _bind_keys / _build_widgets
    (and what they call) read. object.__new__ skips ReviewGUI.__init__, which
    wants a real chain on disk, a napari viewer, and torch (same tactic as
    tests/test_gui_lasso.py's _build_headless_gui)."""
    g = gui.ReviewGUI.__new__(gui.ReviewGUI)
    g.ui_mode = ui_mode
    g.viewer = _StubViewer()
    g.reviewer = ""
    g.point_size = 4.0
    g.auto_zoom = True
    g.zoom_pad = 3.0
    g.neuron = None
    g.chain_idx = None
    g.queue = _FakeQueue()
    return g


# ---------------------------------------------------------------------------
# _bind_keys: registers exactly keys_for_mode(mode), no more, no fewer
# ---------------------------------------------------------------------------

def test_bind_keys_registers_exactly_keys_for_full_mode():
    g = _stub_gui(gui.UI_MODE_FULL)
    g._bind_keys()
    assert set(g.viewer.bound_keys) == gui.keys_for_mode(gui.UI_MODE_FULL)
    assert len(g.viewer.bound_keys) == len(gui.keys_for_mode(gui.UI_MODE_FULL))


def test_bind_keys_registers_exactly_keys_for_review_mode():
    g = _stub_gui(gui.UI_MODE_REVIEW)
    g._bind_keys()
    assert set(g.viewer.bound_keys) == gui.keys_for_mode(gui.UI_MODE_REVIEW)
    assert len(g.viewer.bound_keys) == len(gui.keys_for_mode(gui.UI_MODE_REVIEW))


# ---------------------------------------------------------------------------
# _build_widgets: every panel builder runs in every mode, so the model-panel
# instance attributes always exist, even when the model panel is left out of
# the dock (panels_for_mode's job, not the builders').
# ---------------------------------------------------------------------------

def test_build_widgets_runs_every_panel_builder_in_full_mode(monkeypatch):
    _install_fake_gui_toolkit(monkeypatch)
    g = _stub_gui(gui.UI_MODE_FULL)

    g._build_widgets()

    assert hasattr(g, "_reviewer_edit")   # navigation
    assert hasattr(g, "_size_spin")       # drawing
    assert hasattr(g, "_prompt_mode")     # model
    assert hasattr(g, "_grow_spin")       # model
    assert hasattr(g, "_err_mode")        # verdict


def test_build_widgets_runs_every_panel_builder_in_review_mode(monkeypatch):
    """The critical case: review mode's dock never shows the model panel, but
    _build_model_panel must still run so self._prompt_mode / self._grow_spin
    exist for the methods (e.g. _set_prompt_label) that read them
    unconditionally. A builder skipped here is an AttributeError waiting for
    the first review-mode session that touches one of those methods."""
    _install_fake_gui_toolkit(monkeypatch)
    g = _stub_gui(gui.UI_MODE_REVIEW)

    g._build_widgets()

    assert hasattr(g, "_reviewer_edit")   # navigation
    assert hasattr(g, "_size_spin")       # drawing
    assert hasattr(g, "_prompt_mode")     # model, built even though hidden
    assert hasattr(g, "_grow_spin")       # model, built even though hidden
    assert hasattr(g, "_err_mode")        # verdict
