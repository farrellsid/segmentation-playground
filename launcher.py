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

import json
import os
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
    "neurons": [],
    "reviewer": "",
    "ui_mode": "review",
    "recents": [],
}


def torch_available() -> bool:
    """True when torch imports, which is what the model actions need.

    Kept as a function rather than a module constant so a test can replace it and
    so the (slow) import is not paid at launcher start.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


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
        If no output root is set, or if full mode is requested on a machine with no
        torch. Failing here is the point: the alternative is an exception several
        minutes into a session, after the frames have loaded.
    """
    root = str(profile.get("output_root") or "").strip()
    if not root:
        raise ValueError("no output root set; pick a bundle or an output tree first")
    mode = profile.get("ui_mode", DEFAULT_PROFILE["ui_mode"])
    if mode == "full" and not torch_available():
        raise ValueError("full mode needs torch, which is not installed on this machine. "
                         "Use review mode to redraw, or install the full requirements.")
    neurons = list(profile.get("neurons") or [])
    return {
        "output_root": Path(root),
        "neurons": neurons or None,   # an empty tick list means every neuron
        "reviewer": profile.get("reviewer", ""),
        "ui_mode": mode,
    }


def apply_profile_env(profile: dict) -> None:
    """Export the profile's paths as the env vars sam2_utils.config reads.

    This is what replaces editing tracked source: config.OUTPUT_ROOT and
    config.FRAMES_ROOT pick these up on import.
    """
    if profile.get("output_root"):
        os.environ["SAM2_OUTPUT_ROOT"] = str(profile["output_root"])
    if profile.get("frames_root"):
        os.environ["SAM2_FRAMES_ROOT"] = str(profile["frames_root"])


def _summary_text(root: Path) -> str:
    """Human-readable neuron/chain/progress summary for a picked tree or bundle."""
    problems = bundle.validate_bundle(root) if (root / bundle.BUNDLE_MANIFEST).exists() else []
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


def run() -> None:
    """Open the launcher window, then hand off to the review GUI."""
    from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                                QHBoxLayout, QLabel, QLineEdit, QListWidget,
                                QListWidgetItem, QPushButton, QTextEdit, QVBoxLayout,
                                QWidget)
    from qtpy.QtCore import Qt

    profile = load_profile()
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
    if not torch_available():
        mode_combo.model().item(1).setEnabled(False)
        mode_combo.setToolTip("Reprop needs torch, which is not installed on this machine.")
    mode_combo.setCurrentIndex(0 if profile.get("ui_mode", "review") == "review" else 1)
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

    browse.clicked.connect(pick)
    path_edit.editingFinished.connect(refresh)
    launch_btn.clicked.connect(do_launch)
    refresh()

    win.resize(640, 620)
    win.show()
    app.exec_()


if __name__ == "__main__":
    run()
