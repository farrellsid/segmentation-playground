"""Torch must be imported before Qt, and a torch that will not load must say so.

Reported from a real session: the launcher's preflight said "torch: not installed" on
the Windows box that runs SAM2 every day. Measured there, in three fresh processes:

    import torch                          -> ok, 2.12.0+cu130
    import qtpy.QtWidgets; import torch    -> OSError [WinError 1114], c10.dll
    import torch; import qtpy.QtWidgets    -> ok, cuda available

Qt ships DLLs that shadow torch's, so torch imported AFTER Qt fails. The preflight ran
after `run()` had already imported qtpy, so it was telling the truth about a process
that was already poisoned. The same trap sat under the GUI: `launch()` imported napari
first and torch lazily at the first R/G, which is where it would have failed.

Two guards here. The ordering one runs the real thing in a subprocess, because the bug
IS the import order and nothing short of a real process proves it. The reporting one is
pure.

    py -3 -m pytest tests/test_torch_before_qt.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import launcher

REPO = pathlib.Path(__file__).resolve().parent.parent


def _torch_installed() -> bool:
    out = subprocess.run([sys.executable, "-c", "import torch"],
                         capture_output=True, cwd=str(REPO))
    return out.returncode == 0


class TestOrdering:
    @pytest.mark.skipif(not _torch_installed(), reason="needs torch installed")
    def test_torch_survives_a_later_qt_import_when_probed_first(self):
        """The fix, end to end: probe torch (which imports it), then bring Qt up, then
        use torch. This is the order `launcher.run` and `gui.launch` now follow."""
        code = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from sam2_utils import setup\n"
            "assert setup.torch_problem() is None, setup.torch_problem()\n"
            "from qtpy.QtWidgets import QApplication\n"
            "import torch; torch.zeros(2).sum()\n"
            "print('OK')\n" % REPO)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(REPO))
        assert out.returncode == 0, out.stderr[-600:]
        assert "OK" in out.stdout

    def test_run_probes_torch_before_it_imports_qt(self):
        """A source-order assertion, deliberately. The invariant IS textual order: once
        qtpy is imported, the probe can only report the damage, not avoid it. Nothing
        headless can catch this, since `run()` opens a window."""
        src = (REPO / "launcher.py").read_text(encoding="utf-8")
        body = src[src.index("def run() -> None:"):]
        assert body.index("torch_problem()") < body.index("from qtpy"), \
            "launcher.run must call torch_problem() before importing qtpy"

    def test_launch_probes_torch_before_it_imports_napari(self):
        """Same invariant on the other entry point: `py -3 gui.py` never touches the
        launcher, and napari imports Qt."""
        src = (REPO / "gui.py").read_text(encoding="utf-8")
        body = src[src.index("def launch("):]
        assert body.index("torch_problem()") < body.index("import napari"), \
            "gui.launch must probe torch before importing napari"


class TestReporting:
    def test_a_missing_torch_still_reads_as_not_installed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(launcher, "torch_problem",
                            lambda: "ModuleNotFoundError: No module named 'torch'")
        check = [c for c in launcher.machine_checks({}) if c.name == "torch"][0]
        assert check.ok is False
        # Not an exact string: the detail also names the interpreter now (see
        # TestDiagnosingTheWrongInterpreter). What must hold is that a genuinely absent
        # torch does NOT get reported as the installed-but-unloadable case.
        assert "not importable" in check.detail
        assert "failed to load" not in check.detail
        assert "pip install" in check.fix

    def test_an_installed_but_unloadable_torch_does_not_say_not_installed(
            self, monkeypatch):
        """The reported bug. Saying "not installed" sent a real user to reinstall a
        torch that was already there and working."""
        monkeypatch.setattr(
            launcher, "torch_problem",
            lambda: "OSError: [WinError 1114] A dynamic link library (DLL) "
                    "initialization routine failed. Error loading c10.dll")
        check = [c for c in launcher.machine_checks({}) if c.name == "torch"][0]
        assert check.ok is False
        assert "not installed" not in check.detail
        assert "failed to load" in check.detail
        assert "c10.dll" in check.detail, "the real error must survive into the report"
        assert "Qt" in check.fix

    def test_torch_available_is_the_probe_not_a_fresh_import(self, monkeypatch):
        """torch_available must answer from the cached probe, so the answer taken
        before Qt is the one that sticks for the session."""
        monkeypatch.setattr(launcher, "torch_problem", lambda: None)
        assert launcher.torch_available() is True
        monkeypatch.setattr(launcher, "torch_problem", lambda: "OSError: boom")
        assert launcher.torch_available() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestDiagnosingTheWrongInterpreter:
    """The common real cause of "torch: not installed" is not a missing torch, it is
    torch installed into a DIFFERENT interpreter than the one running the launcher.
    The reviewer's documented setup makes that likely: her review venv is deliberately
    torch-free, so any torch she adds has to land in the same one she launches with."""

    def test_the_install_fix_is_runnable_on_this_platform(self, monkeypatch):
        """It used to say `py -3 ...`, the Windows launcher, which does not exist on
        macOS, so the advice printed on her machine could not be run at all."""
        monkeypatch.setattr(launcher, "torch_problem",
                            lambda: "ModuleNotFoundError: No module named 'torch'")
        check = [c for c in launcher.machine_checks({}) if c.name == "torch"][0]
        assert "py -3" not in check.fix
        assert sys.executable in check.fix

    def test_a_missing_torch_names_the_interpreter_it_is_missing_from(self, monkeypatch):
        monkeypatch.setattr(launcher, "torch_problem",
                            lambda: "ModuleNotFoundError: No module named 'torch'")
        check = [c for c in launcher.machine_checks({}) if c.name == "torch"][0]
        assert sys.executable in check.detail, \
            "without this, 'not installed' cannot be told apart from 'installed elsewhere'"
