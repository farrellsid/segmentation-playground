"""A window for picking a bundle, the neurons to review, and the mode.

Every review setting used to be a command-line flag, and the paths lived in
tracked source, so a second machine meant reading documentation and editing
sam2_utils/config.py. This puts the same choices in front of the person making
them, and keeps their answers in ~/.sam2review/profile.json, outside the repo,
where nothing they configure can collide with git.

Built on qtpy, which napari already depends on, so it adds no dependency and
runs on macOS.

    py -3 launcher.py
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sam2_utils import bundle

#: Per-user settings, deliberately outside the repo.
PROFILE_PATH = Path.home() / ".sam2review" / "profile.json"

#: Review mode is the default: the common case is a reviewer with no GPU, and
#: full mode on such a machine only offers controls that cannot work.
DEFAULT_PROFILE = {
    "output_root": "",
    "frames_root": "",
    "worm_path": "",
    "checkpoint_dir": "",
    "neurons": [],
    "reviewer": "",
    "ui_mode": "review",
    "recents": [],
}


def torch_problem() -> Optional[str]:
    """None when torch is usable in this process, else a one-line reason.

    Delegates to :func:`sam2_utils.setup.torch_problem`, which owns the rule and the
    cache. Kept as a name here so the checks and the tests have one seam to replace.
    """
    from sam2_utils import setup
    return setup.torch_problem()


def torch_available() -> bool:
    """True when torch imports, which is what the model actions need.

    Kept as a function rather than a module constant so a test can replace it, and so
    the answer comes from the cached probe rather than from a fresh import attempt
    made at some arbitrary later moment. WHEN the probe first runs matters: see
    :func:`sam2_utils.setup.torch_problem` for the Qt DLL ordering this depends on.
    """
    return torch_problem() is None


@dataclass
class MachineCheck:
    """One preflight verdict, rendered as a line in the window.

    ``fix`` is what to do about it, phrased as the field to fill or the command to
    run, and is empty when ``ok``. Kept as data rather than printed text so the tests
    assert on the verdict rather than on wording.
    """
    name: str
    ok: bool
    detail: str
    fix: str = ""


@lru_cache(maxsize=1)
def _device_name() -> str:
    """The device SAM2 would run on: 'cuda', 'mps' or 'cpu'.

    A named module function rather than an inline call so a test can replace it
    without a GPU, and so the torch import stays inside it. Cached because the mode
    gate re-runs the checks on every path edit, while the device cannot change within
    a session, and ``setup_device`` is not free: it initialises CUDA and enters a
    process-global autocast context.
    """
    from sam2_utils import setup
    return setup.setup_device(verbose=False).type


def _writable(path: Path) -> Optional[str]:
    """None when ``path`` is a directory that can be written to, else why not.

    Proved by writing and deleting a file rather than by reading permission bits,
    which are unreliable on Windows and say nothing about a full or read-only volume.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sam2_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return str(exc)
    return None


def machine_checks(profile: dict) -> list:
    """What this machine can do, one MachineCheck per requirement.

    Ordered from most fundamental to most specific, which is also the order a reviewer
    fixes them in. Each check is caught independently: one raising must not hide the
    rest, the same reason bundle.validate_bundle returns a list of problems instead of
    raising on the first.

    Parameters
    ----------
    profile : dict
        A profile from :func:`load_profile`. Its ``worm_path``, ``checkpoint_dir`` and
        ``frames_root`` are read; an empty one falls back to the config default, which
        is what an unconfigured machine looks like.

    Returns
    -------
    list of MachineCheck
        One verdict per requirement, in fix order.
    """
    from pipeline.config import PipelineConfig
    from sam2_utils import config as cfg_mod

    checks = []

    def _check(name, fn):
        try:
            checks.append(fn())
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            checks.append(MachineCheck(name, False, f"check failed: {exc}",
                                       "this is a bug; report it with the message above"))

    def _torch():
        problem = torch_problem()
        if problem is None:
            import torch
            return MachineCheck("torch", True, f"version {torch.__version__}")
        if "No module named" in problem:
            return MachineCheck("torch", False, "not installed",
                                "install the full requirements: py -3 -m pip install -r "
                                "requirements.txt")
        # Installed but not loadable is a different problem and needs a different fix.
        # Reporting it as "not installed" sent a real user to reinstall a torch that was
        # already there. The usual cause on Windows is Qt getting imported first: Qt
        # ships DLLs that shadow torch's, and torch then fails on c10.dll.
        return MachineCheck("torch", False, f"installed, but it failed to load: {problem}",
                            "usually means Qt was imported before torch in this process. "
                            "Start from launcher.py, which loads torch first, and report "
                            "it if this persists")

    def _device():
        # Checked here rather than let _device_name() raise into the generic _check()
        # wrapper: that would report the device as ok=False with "this is a bug;
        # report it", which is wrong on both counts. A missing torch is the torch
        # check's job to report, not a bug, and the spec promises device is never
        # ok=False.
        if not torch_available():
            return MachineCheck("device", True,
                                "unknown: torch is not installed, so the device this "
                                "machine would run on cannot be probed yet")
        name = _device_name()
        if name == "cuda":
            detail = "cuda, the fast path"
        elif name == "mps":
            detail = ("mps (Apple Silicon). Propagation will be slow, and SAM2 on MPS is "
                      "preliminary upstream, so masks may differ from a CUDA run")
        else:
            detail = "cpu. Propagation will be slow; expect minutes per chain"
        return MachineCheck("device", True, detail)

    def _checkpoint():
        # No fallback to cfg_mod.CHECKPOINT_DIR: that is this repo's own default
        # (relative "checkpoints", resolved against wherever the launcher happens to
        # be started from), not a path the reviewer chose. Reporting it as if she had
        # is the same lie _raw_em avoids below: an empty field means unset, full stop.
        ckpt_str = (profile.get("checkpoint_dir") or "").strip()
        if not ckpt_str:
            return MachineCheck("checkpoint", False, "not set",
                                "point 'Checkpoints' at a folder you can write to")
        ckpt_dir = Path(ckpt_str)
        problem = _writable(ckpt_dir)
        if problem:
            return MachineCheck("checkpoint", False, f"{ckpt_dir}: {problem}",
                                "point 'Checkpoints' at a folder you can write to")
        size = PipelineConfig().model_size
        _url, filename, _model_cfg = cfg_mod.SAM2_CHECKPOINTS[size]
        if (ckpt_dir / filename).exists():
            return MachineCheck("checkpoint", True, f"{filename} present in {ckpt_dir}")
        return MachineCheck("checkpoint", True,
                            f"{filename} is missing and will download on first use "
                            f"(about 2.4 GB for {size}) into {ckpt_dir}")

    def _raw_em():
        from pipeline import raw_em_problem
        worm = profile.get("worm_path") or ""
        # Pass "" rather than None when unset: None falls back to config.WORM_PATH, this
        # repo's Windows default, which would pass on a machine that happens to have it
        # and report a path the reviewer never chose.
        problem = raw_em_problem(worm)
        if problem:
            return MachineCheck("raw EM", False, problem,
                                "set 'Raw EM (tif stack)'; needed for recrop only, not "
                                "for redrawing or re-propagating")
        return MachineCheck("raw EM", True, f"tif stack at {worm}")

    def _frames():
        # Same reasoning as _checkpoint above: cfg_mod.FRAMES_ROOT is this repo's
        # Windows default (F:\ZhenLab\Data). Falling back to it silently created a
        # directory of that literal name in the working directory on a Mac and
        # reported it as writable, which is M8 in the review notes.
        root_str = (profile.get("frames_root") or "").strip()
        if not root_str:
            return MachineCheck("frames cache", False, "not set",
                                "point 'Frames cache' at a folder you can write to")
        root = Path(root_str)
        problem = _writable(root)
        if problem:
            return MachineCheck("frames cache", False, f"{root}: {problem}",
                                "point 'Frames cache' at a folder you can write to")
        return MachineCheck("frames cache", True, f"writable at {root}")

    _check("torch", _torch)
    _check("device", _device)
    _check("checkpoint", _checkpoint)
    _check("raw EM", _raw_em)
    _check("frames cache", _frames)
    return checks


def can_run_model(checks: list) -> bool:
    """Whether full mode is honest on this machine: torch and a usable checkpoint dir.

    The raw EM is deliberately NOT required. Recrop needs it; re-predict and resume
    propagation do not, and refusing the model outright over a missing tif stack would
    take away the two controls that still work.
    """
    needed = {"torch", "checkpoint"}
    return all(c.ok for c in checks if c.name in needed)


def load_profile(path: Optional[Path] = None) -> dict:
    """Load the per-user profile, filling anything absent from the defaults.

    Parameters
    ----------
    path : Path, optional
        Profile location. Defaults to :data:`PROFILE_PATH`.

    Returns
    -------
    dict
        A profile with every key in :data:`DEFAULT_PROFILE` present, so callers
        never have to guard on a missing key after an older version wrote the file.
    """
    path = Path(path) if path is not None else PROFILE_PATH
    if not path.exists():
        return dict(DEFAULT_PROFILE)
    stored = json.loads(path.read_text(encoding="utf-8"))
    return {**DEFAULT_PROFILE, **stored}


def save_profile(profile: dict, path: Optional[Path] = None) -> Path:
    """Write the profile, creating its directory if needed.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path) if path is not None else PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def build_launch_kwargs(profile: dict) -> dict:
    """Turn a profile into keyword arguments for :func:`gui.launch`.

    Parameters
    ----------
    profile : dict
        A profile from :func:`load_profile`.

    Returns
    -------
    dict
        ``output_root``, ``neurons``, ``reviewer`` and ``ui_mode``.

    Raises
    ------
    ValueError
        If no output root is set, or if full mode is requested on a machine that
        cannot honestly run it (no torch, or no writable checkpoint folder). Failing
        here is the point: the alternative is an exception several minutes into a
        session, after the frames have loaded and the predictor build fails.

    Notes
    -----
    This is the decision point's own refusal, not just the mode combo's. The combo
    is re-gated on every path edit (see launcher.run's refresh_mode_gate), but a
    disabled item can still be the CURRENT one in a stale window (a saved profile
    from a machine that could run full mode, reopened on one that cannot), so the
    check is repeated here rather than trusted from the widget.
    """
    root = str(profile.get("output_root") or "").strip()
    if not root:
        raise ValueError("no output root set; pick a bundle or an output tree first")
    mode = profile.get("ui_mode", DEFAULT_PROFILE["ui_mode"])
    if mode == "full":
        if not torch_available():
            raise ValueError("full mode needs torch, which is not installed on this "
                             "machine. Use review mode to redraw, or install the full "
                             "requirements.")
        checks = machine_checks(profile)
        if not can_run_model(checks):
            failing = ", ".join(c.name for c in checks
                                if not c.ok and c.name in ("torch", "checkpoint"))
            raise ValueError(f"full mode is not runnable on this machine ({failing} "
                             f"failing); use 'Check this machine' for details, or use "
                             f"review mode.")
    neurons = list(profile.get("neurons") or [])
    return {
        "output_root": Path(root),
        "neurons": neurons or None,   # an empty tick list means every neuron
        "reviewer": profile.get("reviewer", ""),
        "ui_mode": mode,
    }


def apply_profile_env(profile: dict) -> None:
    """Export the profile's paths as the env vars sam2_utils.config reads, and make
    config see them.

    This is what replaces editing tracked source: config.OUTPUT_ROOT, FRAMES_ROOT,
    WORM_PATH and CHECKPOINT_DIR pick these up on import. An empty field exports
    nothing, so config keeps its own default rather than reading an empty string as
    a path.

    Setting the env vars is only half the job. `sam2_utils.config` reads them at
    MODULE IMPORT time, and by the time this runs, `machine_checks` has already
    imported it once (at launcher startup, to build the preflight checks), so the
    module is already sitting in `sys.modules` with its Path constants computed
    from whatever the environment held at that first import: the repo's Windows
    defaults on a machine that has never set anything. Exporting a new env var
    after that point changes nothing on its own. `gui.py` and `pipeline` hold the
    SAME module object (`from sam2_utils import config`), not a copy of its
    values, so reloading it in place here is what makes them see the new paths:
    an import elsewhere just returns the same, now-updated, object from
    sys.modules. Without this, a reviewer who fills in the profile and launches
    still gets refused with "Set 'Raw EM (tif stack)'" naming the field she just
    set, because config.WORM_PATH was never told.
    """
    if profile.get("output_root"):
        os.environ["SAM2_OUTPUT_ROOT"] = str(profile["output_root"])
    if profile.get("frames_root"):
        os.environ["SAM2_FRAMES_ROOT"] = str(profile["frames_root"])
    if profile.get("worm_path"):
        os.environ["SAM2_WORM_PATH"] = str(profile["worm_path"])
    if profile.get("checkpoint_dir"):
        os.environ["SAM2_CHECKPOINT_DIR"] = str(profile["checkpoint_dir"])
    from sam2_utils import config as cfg_mod
    importlib.reload(cfg_mod)


def _summary_text(root: Path) -> str:
    """Human-readable neuron/chain/progress summary for a picked tree or bundle."""
    # Frames ARE required here: this is the picker a reviewer is about to open a
    # chain from, so a clone whose frames were never placed should say so now
    # rather than fail confusingly on the first chain.
    problems = bundle.validate_bundle(root) if bundle.is_bundle(root) else []
    progress = bundle.review_progress(root)
    if not progress:
        return f"No chains found under {root}"
    lines = [f"{name}: {v['reviewed']}/{v['total']} reviewed"
             for name, v in sorted(progress.items())]
    if problems:
        lines.append("")
        lines.append("Bundle problems:")
        lines.extend(f"  {p}" for p in problems)
    return "\n".join(lines)


def open_render_window(source: str, neurons) -> None:
    """Hand the currently picked source and ticked neurons to the render window.

    A named function rather than a closure inside `run()`, for the same reason
    `build_launch_kwargs` is one: it can be tested without Qt or a display. The
    reviewer picks a bundle once here, and the render window inherits that choice
    instead of asking again.
    """
    import render_review
    render_review.run(source=source, neurons=list(neurons or []))


def run() -> None:
    """Open the launcher window, then hand off to the review GUI."""
    # Torch BEFORE Qt, and this line must stay first. On Windows, Qt ships DLLs that
    # shadow torch's, so a torch imported after qtpy fails with WinError 1114 on
    # c10.dll on a machine where torch is fine. Asked here, the answer is cached for
    # the process AND torch is left in sys.modules for the predictor build later.
    # Measured: qtpy-then-torch fails, torch-then-qtpy works with CUDA available.
    torch_problem()
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                                QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                QListWidgetItem, QPushButton, QTextEdit, QVBoxLayout,
                                QWidget)
    from qtpy.QtCore import Qt

    profile = load_profile()
    # Apply the saved profile before anything below (machine_checks, first paint of
    # the mode combo) runs, and before it gets its own chance to import
    # sam2_utils.config for the first time. Not the only place this is called (see
    # refresh_mode_gate and do_launch), but the earliest one matters too: a reviewer
    # who reopens the launcher with a profile already saved should not need to
    # touch a field before config reflects her paths.
    apply_profile_env(profile)
    app = QApplication.instance() or QApplication([])
    win = QWidget()
    win.setWindowTitle("SAM2 review launcher")
    layout = QVBoxLayout(win)

    path_row = QHBoxLayout()
    path_edit = QLineEdit(profile.get("output_root", ""))
    browse = QPushButton("Browse...")
    path_row.addWidget(QLabel("Bundle or output tree:"))
    path_row.addWidget(path_edit, 1)
    path_row.addWidget(browse)
    layout.addLayout(path_row)

    def _path_row(label: str, value: str, caption: str):
        """A label + line edit + Browse button, added to the layout. Returns the edit.

        Four of these now exist; without a helper the window is the same six lines
        four times over, and the fourth one drifts."""
        row = QHBoxLayout()
        edit = QLineEdit(value)
        btn = QPushButton("Browse...")

        def _pick(*_):
            chosen = QFileDialog.getExistingDirectory(win, caption,
                                                      edit.text() or str(Path.home()))
            if chosen:
                edit.setText(chosen)
                # setText() emits textChanged, never editingFinished, which is all
                # the mode gate below is wired to. Without this, Browse-ing a good
                # checkpoint folder left full mode disabled (it can now run), and
                # Browse-ing an unwritable one after full mode was already enabled
                # left it enabled and selected: the exact "selected but not
                # runnable" state the gate exists to prevent. refresh_mode_gate is
                # defined later in this function but resolved at call time, by
                # which point it exists (this closure only runs after a click).
                refresh_mode_gate()

        btn.clicked.connect(_pick)
        row.addWidget(QLabel(label))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit

    layout.addWidget(QLabel("This machine (needed for recrop and SAM2):"))
    worm_edit = _path_row("Raw EM (tif stack):", profile.get("worm_path", ""),
                          "Pick the raw EM tif stack")
    frames_edit = _path_row("Frames cache:", profile.get("frames_root", ""),
                            "Pick a folder for prepared frames")
    ckpt_edit = _path_row("Checkpoints:", profile.get("checkpoint_dir", ""),
                          "Pick the SAM2 checkpoint folder")

    neuron_list = QListWidget()
    neuron_list.setSelectionMode(QListWidget.NoSelection)
    layout.addWidget(QLabel("Neurons (none ticked means all):"))
    layout.addWidget(neuron_list, 1)

    summary = QTextEdit()
    summary.setReadOnly(True)
    layout.addWidget(summary, 1)

    opts = QHBoxLayout()
    reviewer_edit = QLineEdit(profile.get("reviewer", ""))
    mode_combo = QComboBox()
    mode_combo.addItem("Redraw only (no model)", "review")
    mode_combo.addItem("Enable SAM2/SAM3 reprop", "full")

    def _machine_profile():
        return {**profile,
                "frames_root": frames_edit.text().strip(),
                "worm_path": worm_edit.text().strip(),
                "checkpoint_dir": ckpt_edit.text().strip()}

    def refresh_mode_gate(*_):
        """Enable full mode only when this machine can honestly run it.

        Torch alone is not enough: a machine with torch and no checkpoint used to be
        offered the reprop controls and failed minutes later, at predictor build.

        Also the one place, besides do_launch, that pushes the edited paths into
        sam2_utils.config: this runs on every path-field edit (wired below) and on
        every Browse pick, so config stays in step with what is on screen rather
        than only catching up at the moment Launch is pressed.
        """
        prof = _machine_profile()
        apply_profile_env(prof)
        checks = machine_checks(prof)
        ok = can_run_model(checks)
        mode_combo.model().item(1).setEnabled(ok)
        if not ok:
            missing = ", ".join(c.name for c in checks
                                if not c.ok and c.name in ("torch", "checkpoint"))
            mode_combo.setToolTip(f"Reprop needs {missing} on this machine.")
            # Fall back to review when the saved profile asks for full on a machine that
            # cannot run it. Without this the combo would show the disabled reprop item
            # as the selected one, offering a choice this machine cannot honour.
            mode_combo.setCurrentIndex(0)
        else:
            mode_combo.setToolTip("")
        return checks

    want_full = profile.get("ui_mode", "review") == "full"
    mode_combo.setCurrentIndex(1 if (want_full and can_run_model(
        machine_checks(_machine_profile()))) else 0)
    remember = QCheckBox("Remember these settings")
    remember.setChecked(True)
    opts.addWidget(QLabel("Reviewer:"))
    opts.addWidget(reviewer_edit)
    opts.addWidget(QLabel("Mode:"))
    opts.addWidget(mode_combo)
    opts.addWidget(remember)
    layout.addLayout(opts)

    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(status)
    launch_btn = QPushButton("Launch review")
    layout.addWidget(launch_btn)
    render_btn = QPushButton("Render video + mesh")
    layout.addWidget(render_btn)
    check_btn = QPushButton("Check this machine")
    layout.addWidget(check_btn)

    def do_check(*_):
        """Report every machine check into the summary pane, fixes included.

        Reuses refresh_mode_gate's own checks rather than running machine_checks a
        second time: the two used to run the write-probe / checkpoint-scan pass
        twice on every click, once here and once inside refresh_mode_gate.
        """
        checks = refresh_mode_gate()
        lines = []
        for c in checks:
            mark = "ok  " if c.ok else "NO  "
            lines.append(f"{mark}{c.name}: {c.detail}")
            if c.fix:
                lines.append(f"      fix: {c.fix}")
        summary.setPlainText("\n".join(lines))

    def refresh(*_):
        root = Path(path_edit.text().strip() or ".")
        neuron_list.clear()
        if not root.is_dir():
            summary.setPlainText(f"Not a directory: {root}")
            return
        progress = bundle.review_progress(root)
        for name in sorted(progress):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in (profile.get("neurons") or [])
                               else Qt.Unchecked)
            neuron_list.addItem(item)
        summary.setPlainText(_summary_text(root))

    def pick(*_):
        chosen = QFileDialog.getExistingDirectory(win, "Pick a bundle or output tree",
                                                  path_edit.text() or str(Path.home()))
        if chosen:
            path_edit.setText(chosen)
            refresh()

    def ticked():
        return [neuron_list.item(i).text() for i in range(neuron_list.count())
                if neuron_list.item(i).checkState() == Qt.Checked]

    def do_launch(*_):
        prof = {**profile,
                "output_root": path_edit.text().strip(),
                "frames_root": frames_edit.text().strip(),
                "worm_path": worm_edit.text().strip(),
                "checkpoint_dir": ckpt_edit.text().strip(),
                "neurons": ticked(),
                "reviewer": reviewer_edit.text().strip(),
                "ui_mode": mode_combo.currentData()}
        try:
            kwargs = build_launch_kwargs(prof)
        except ValueError as exc:
            status.setText(str(exc))
            return
        if remember.isChecked():
            save_profile(prof)
        apply_profile_env(prof)
        import gui
        win.close()
        gui.launch(**kwargs)

    def do_render(*_):
        open_render_window(path_edit.text().strip(), ticked())

    browse.clicked.connect(pick)
    path_edit.editingFinished.connect(refresh)
    launch_btn.clicked.connect(do_launch)
    render_btn.clicked.connect(do_render)
    check_btn.clicked.connect(do_check)
    # The machine rows re-gate the mode combo, but deliberately do NOT call refresh():
    # which neurons are listed depends on the picked tree, not on where the raw EM is.
    for edit in (worm_edit, frames_edit, ckpt_edit):
        edit.editingFinished.connect(refresh_mode_gate)
    refresh()
    refresh_mode_gate()

    win.resize(640, 760)
    win.show()
    app.exec_()


if __name__ == "__main__":
    run()
