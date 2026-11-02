# Launcher Machine Setup and Recrop Round Trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a reviewer on her own Mac record the raw EM, frames cache and checkpoint paths from the launcher, see honestly what her machine can do, recrop a chain inside a review bundle, and have that recrop's geometry come home when she returns the bundle.

**Architecture:** Pure functions carry the logic and the Qt window stays a thin shell over them, the split `launcher.py` already uses for `build_launch_kwargs`. Three new seams: `pipeline.frames.raw_em_problem` owns the "is this the tif stack" rule for both the launcher's preflight and the GUI's recrop guard; `bundle.adopt_chain_frames` keeps a recropped bundle self-contained; `bundle.merge_geometry` is the allowlist that carries a recrop home without carrying the reviewer's machine paths with it.

**Tech Stack:** Python 3.13, qtpy (via napari), pytest, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-27-launcher-machine-setup-and-recrop-round-trip-design.md`

## Global Constraints

- **No em dashes or en dashes anywhere**, in code, comments, docstrings, commit messages or docs. This is a hard repo rule.
- No `Co-Authored-By: Claude` trailer in commit messages.
- Run tests with `py -3 -m pytest`, lint with `ruff check .`. Both must pass before every commit.
- The library must not import drivers: `sam2_utils/` and `pipeline/` never import `gui`, `launcher`, `batch` or `import_bundle`.
- `sam2_utils/config.py` stays import-light: no torch, no cv2, no network calls at import time.
- Never merge the `config` block of a `state.json` across machines. It contains `output_root` and `frames_root`.
- `BUNDLE_SCHEMA_VERSION` stays `1`. The bundles already delivered are version 1 and must keep working.
- Docstrings follow the surrounding numpydoc style, and say WHY, not just what.

---

### Task 1: Machine paths in the profile, and a checkpoint dir that is not relative to the shell

**Files:**
- Modify: `sam2_utils/config.py:22` (the `CHECKPOINT_DIR` constant)
- Modify: `launcher.py:26-35` (`DEFAULT_PROFILE`), `launcher.py:122-131` (`apply_profile_env`), `launcher.py:164-281` (`run`, the window)
- Test: `tests/test_config_env.py`, `tests/test_launcher_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: profile keys `worm_path` and `checkpoint_dir` (both `str`, default `""`); env vars `SAM2_WORM_PATH` and `SAM2_CHECKPOINT_DIR`; `config.CHECKPOINT_DIR` honouring `SAM2_CHECKPOINT_DIR`.

- [ ] **Step 1: Write the failing config test**

Add to `tests/test_config_env.py`. Also add `SAM2_CHECKPOINT_DIR` to the autouse fixture's pop list, or a reload in this test leaks a tmp_path into every later test in the process:

```python
def test_checkpoint_dir_honours_env(monkeypatch, tmp_path):
    """CHECKPOINT_DIR used to be a bare relative Path("checkpoints"), so which
    checkpoint a session found depended on the directory it was launched from."""
    cfg = _reload(monkeypatch, SAM2_CHECKPOINT_DIR=str(tmp_path / "ckpts"))
    assert cfg.CHECKPOINT_DIR == Path(tmp_path / "ckpts")


def test_checkpoint_dir_default_is_unchanged(monkeypatch):
    from sam2_utils import config
    monkeypatch.delenv("SAM2_CHECKPOINT_DIR", raising=False)
    cfg = importlib.reload(config)
    assert cfg.CHECKPOINT_DIR == Path("checkpoints")
```

And in the `_restore_config_module` fixture, beside the two existing pops:

```python
    os.environ.pop("SAM2_CHECKPOINT_DIR", None)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3 -m pytest tests/test_config_env.py -v`
Expected: `test_checkpoint_dir_honours_env` FAILS, because `CHECKPOINT_DIR` ignores the environment.

- [ ] **Step 3: Make CHECKPOINT_DIR env-overridable**

In `sam2_utils/config.py`, replace the `CHECKPOINT_DIR` line and its comment with:

```python
#: Where SAM2 checkpoints are downloaded to. Override with the SAM2_CHECKPOINT_DIR env
#: var, the same pattern WORM_PATH uses above. The default is relative to the working
#: directory, so a launcher started from anywhere but the repo root would otherwise look
#: for checkpoints somewhere the user never put them, and silently re-download.
CHECKPOINT_DIR = Path(os.environ.get("SAM2_CHECKPOINT_DIR", "checkpoints"))
```

- [ ] **Step 4: Run the config tests**

Run: `py -3 -m pytest tests/test_config_env.py -v`
Expected: PASS, all five tests.

- [ ] **Step 5: Write the failing launcher tests**

Add to `tests/test_launcher_config.py`:

```python
import os


def test_machine_paths_default_to_empty():
    """Empty means 'not set', which the checks report as such. A wrong default
    would be worse than none: it reads as configured and fails later."""
    assert launcher.DEFAULT_PROFILE["worm_path"] == ""
    assert launcher.DEFAULT_PROFILE["checkpoint_dir"] == ""


def test_apply_profile_env_exports_every_machine_path(monkeypatch):
    for var in ("SAM2_OUTPUT_ROOT", "SAM2_FRAMES_ROOT",
                "SAM2_WORM_PATH", "SAM2_CHECKPOINT_DIR"):
        monkeypatch.delenv(var, raising=False)
    launcher.apply_profile_env({"output_root": "/o", "frames_root": "/f",
                                "worm_path": "/w", "checkpoint_dir": "/c"})
    assert os.environ["SAM2_OUTPUT_ROOT"] == "/o"
    assert os.environ["SAM2_FRAMES_ROOT"] == "/f"
    assert os.environ["SAM2_WORM_PATH"] == "/w"
    assert os.environ["SAM2_CHECKPOINT_DIR"] == "/c"


def test_apply_profile_env_leaves_unset_paths_alone(monkeypatch):
    """An empty field must not export an empty env var: config would then read ''
    as the path instead of falling back to its default."""
    monkeypatch.delenv("SAM2_WORM_PATH", raising=False)
    launcher.apply_profile_env({"output_root": "/o", "worm_path": ""})
    assert "SAM2_WORM_PATH" not in os.environ


def test_a_profile_written_before_these_keys_existed_still_loads(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"output_root": "/o", "reviewer": "lucinda"}),
                    encoding="utf-8")
    prof = launcher.load_profile(path)
    assert prof["worm_path"] == "" and prof["checkpoint_dir"] == ""
    assert prof["reviewer"] == "lucinda"
```

- [ ] **Step 6: Run them to verify they fail**

Run: `py -3 -m pytest tests/test_launcher_config.py -v`
Expected: FAIL with `KeyError: 'worm_path'`.

- [ ] **Step 7: Add the keys and the exports**

In `launcher.py`, `DEFAULT_PROFILE` becomes:

```python
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
```

and `apply_profile_env` gains the two new exports:

```python
def apply_profile_env(profile: dict) -> None:
    """Export the profile's paths as the env vars sam2_utils.config reads.

    This is what replaces editing tracked source: config.OUTPUT_ROOT, FRAMES_ROOT,
    WORM_PATH and CHECKPOINT_DIR pick these up on import. An empty field exports
    nothing, so config keeps its own default rather than reading an empty string as
    a path.
    """
    if profile.get("output_root"):
        os.environ["SAM2_OUTPUT_ROOT"] = str(profile["output_root"])
    if profile.get("frames_root"):
        os.environ["SAM2_FRAMES_ROOT"] = str(profile["frames_root"])
    if profile.get("worm_path"):
        os.environ["SAM2_WORM_PATH"] = str(profile["worm_path"])
    if profile.get("checkpoint_dir"):
        os.environ["SAM2_CHECKPOINT_DIR"] = str(profile["checkpoint_dir"])
```

- [ ] **Step 8: Run the launcher tests**

Run: `py -3 -m pytest tests/test_launcher_config.py -v`
Expected: PASS.

- [ ] **Step 9: Add the three path rows to the window**

In `launcher.py`'s `run()`, factor the existing browse row into a reusable local so four rows do not become four copies. Immediately after the existing `path_row` block (which ends with `layout.addLayout(path_row)`), insert:

```python
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
```

Then in `do_launch`, the profile it builds gains the three values:

```python
        prof = {**profile,
                "output_root": path_edit.text().strip(),
                "frames_root": frames_edit.text().strip(),
                "worm_path": worm_edit.text().strip(),
                "checkpoint_dir": ckpt_edit.text().strip(),
                "neurons": ticked(),
                "reviewer": reviewer_edit.text().strip(),
                "ui_mode": mode_combo.currentData()}
```

Raise the window height so the extra rows do not push the buttons under the fold:

```python
    win.resize(640, 760)
```

- [ ] **Step 10: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass, no lint findings.

- [ ] **Step 11: Commit**

```bash
git add sam2_utils/config.py launcher.py tests/test_config_env.py tests/test_launcher_config.py
git commit -m "launcher: record the raw EM, frames cache and checkpoint paths"
```

---

### Task 2: One home for "is this the raw EM stack", and a recrop that refuses without it

**Files:**
- Modify: `pipeline/frames.py` (add `raw_em_problem` after `TifFrameStore`, around line 178)
- Modify: `pipeline/__init__.py` (export it beside the other `frames` names)
- Modify: `gui.py`, `_recrop_to_window`
- Test: `tests/test_raw_em_problem.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `pipeline.raw_em_problem(worm_path=None) -> Optional[str]`, returning None when the path is the tif stack and a human-readable reason when it is not. Task 3's preflight calls it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_em_problem.py`:

```python
"""pipeline.raw_em_problem: the single rule for whether a path is the raw EM stack.

Recrop re-reads full-resolution tifs, so a wrong or unset path is the difference
between a working recrop and a failure deep inside a tif read. Both the launcher's
preflight and the GUI's recrop guard ask this one function, so the rule cannot drift
between the place that reports it and the place that enforces it.

    py -3 -m pytest tests/test_raw_em_problem.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pipeline


def _stack(tmp_path):
    """A directory shaped like the real stack: '1301____z1300.0.tif'."""
    d = tmp_path / "raw"
    d.mkdir()
    for file_z in (1300, 1301, 1302):
        (d / f"{file_z + 1}____z{file_z}.0.tif").write_bytes(b"")
    return d


def test_a_real_looking_stack_has_no_problem(tmp_path):
    assert pipeline.raw_em_problem(_stack(tmp_path)) is None


def test_an_unset_path_says_so(tmp_path):
    assert "not set" in pipeline.raw_em_problem("")


def test_a_missing_directory_is_named(tmp_path):
    problem = pipeline.raw_em_problem(tmp_path / "nope")
    assert "nope" in problem


def test_a_file_is_not_a_stack(tmp_path):
    f = tmp_path / "stack.tif"
    f.write_bytes(b"")
    assert pipeline.raw_em_problem(f) is not None


def test_an_empty_directory_is_a_problem(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert pipeline.raw_em_problem(d) is not None


def test_tifs_that_are_not_the_stack_do_not_pass(tmp_path):
    """A folder of unrelated tifs is the dangerous case: it looks configured, and
    every frame lookup then fails one at a time instead of once, up front."""
    d = tmp_path / "other"
    d.mkdir()
    (d / "screenshot.tif").write_bytes(b"")
    (d / "figure_panel.tif").write_bytes(b"")
    problem = pipeline.raw_em_problem(d)
    assert problem is not None and "z" in problem


def test_one_good_frame_among_others_is_enough(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "notes.tif").write_bytes(b"")
    (d / "1301____z1300.0.tif").write_bytes(b"")
    assert pipeline.raw_em_problem(d) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3 -m pytest tests/test_raw_em_problem.py -v`
Expected: FAIL with `AttributeError: module 'pipeline' has no attribute 'raw_em_problem'`.

- [ ] **Step 3: Implement it in pipeline/frames.py**

Add after the `TifFrameStore` class (before `load_frame_sam`):

```python
def raw_em_problem(worm_path=None) -> Optional[str]:
    """None when ``worm_path`` looks like the raw EM tif stack, else why it does not.

    The stack is a flat directory of ``..z{file_z}.tif`` files, parsed by
    :func:`_parse_file_z`. This lives here because this module owns that naming
    convention; the launcher's preflight and the GUI's recrop guard both call it, so
    the rule cannot drift between what is reported and what is enforced.

    A directory of unrelated tifs deliberately does NOT pass. That case looks
    configured and then fails one frame lookup at a time, which is the worst of both
    worlds: recrop is offered and then dies deep inside a read.

    Parameters
    ----------
    worm_path : path-like, optional
        Defaults to :data:`sam2_utils.config.WORM_PATH`.

    Returns
    -------
    str or None
        A human-readable problem, or None when the path is usable.
    """
    if worm_path is None:
        worm_path = config.WORM_PATH
    if not str(worm_path).strip():
        return "no raw EM path set"
    path = Path(worm_path)
    if not path.is_dir():
        return f"{path} is not a directory"
    for f in path.glob("*.tif"):
        try:
            _parse_file_z(f)
        except (ValueError, IndexError):
            continue
        return None
    return (f"{path} holds no ..z<number>.tif frames, so it is not the raw EM stack "
            f"(expected names like 1301____z1300.0.tif)")
```

Check the imports at the top of `pipeline/frames.py`: it must already have `Path` and `Optional` and `config` in scope. If `Optional` is absent, add it to the existing `typing` import rather than adding a new import line.

In `pipeline/__init__.py`, add `raw_em_problem` to the `from .frames import (...)` block and to `__all__`, both in alphabetical position.

- [ ] **Step 4: Run the test**

Run: `py -3 -m pytest tests/test_raw_em_problem.py -v`
Expected: PASS, seven tests.

- [ ] **Step 5: Guard recrop in the GUI**

In `gui.py`, `_recrop_to_window`, insert as the first statement of the method body, before the `from dataclasses import replace` line:

```python
        # Both recrop entry points (grow, and the region picker's confirm) funnel through
        # here, so the guard lives here rather than in each. Recrop re-reads full-res tifs;
        # without them it fails deep inside a read, minutes in, with nothing naming the
        # cause.
        problem = pipeline.raw_em_problem()
        if problem:
            print(f"[gui] recrop needs the raw EM tif stack: {problem}. Set 'Raw EM (tif "
                  f"stack)' in the launcher, then reopen this chain.")
            return
```

- [ ] **Step 6: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pipeline/frames.py pipeline/__init__.py gui.py tests/test_raw_em_problem.py
git commit -m "recrop: refuse up front when the raw EM stack is not configured"
```

---

### Task 3: A preflight that names what is missing

**Files:**
- Modify: `launcher.py` (add `MachineCheck` and `machine_checks` above `run`, wire a button and the mode gate inside `run`)
- Test: `tests/test_launcher_machine_checks.py` (create)

**Interfaces:**
- Consumes: `pipeline.raw_em_problem` (Task 2); profile keys `worm_path`, `frames_root`, `checkpoint_dir` (Task 1).
- Produces: `launcher.MachineCheck` (dataclass with `name: str`, `ok: bool`, `detail: str`, `fix: str`) and `launcher.machine_checks(profile: dict) -> list[MachineCheck]`; `launcher.can_run_model(checks) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_launcher_machine_checks.py`:

```python
"""launcher.machine_checks: what this machine can actually do, before a session starts.

Full mode used to unlock on a bare `import torch`, so a machine with torch and no
checkpoint offered the reprop controls and failed minutes later, and one with no raw EM
offered recrop and failed deep inside a tif read. These are the checks that make the
window tell the truth first.

No Qt and no GPU: machine_checks is pure, the same split that lets
tests/test_launcher_config.py exercise build_launch_kwargs without a display.

    py -3 -m pytest tests/test_launcher_machine_checks.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import launcher


def _named(checks, name):
    for c in checks:
        if c.name == name:
            return c
    raise AssertionError(f"no check named {name!r} in {[c.name for c in checks]}")


def _stack(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "1301____z1300.0.tif").write_bytes(b"")
    return d


def _good_profile(tmp_path):
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    (ckpt / "sam2.1_hiera_large.pt").write_bytes(b"")
    frames = tmp_path / "frames"
    frames.mkdir()
    return {"worm_path": str(_stack(tmp_path)), "checkpoint_dir": str(ckpt),
            "frames_root": str(frames)}


@pytest.fixture(autouse=True)
def _torch_present(monkeypatch):
    """Default every test to a machine that HAS torch, so each test states only the
    one thing it is about."""
    monkeypatch.setattr(launcher, "torch_available", lambda: True)
    monkeypatch.setattr(launcher, "_device_name", lambda: "cuda")


def test_every_check_has_a_name_and_a_verdict(tmp_path):
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert {c.name for c in checks} == {"torch", "device", "checkpoint",
                                        "raw EM", "frames cache"}
    assert all(isinstance(c.ok, bool) for c in checks)


def test_a_fully_configured_machine_passes_everything(tmp_path):
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert all(c.ok for c in checks), [c.name for c in checks if not c.ok]
    assert launcher.can_run_model(checks) is True


def test_missing_torch_fails_and_says_how_to_fix_it(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "torch_available", lambda: False)
    checks = launcher.machine_checks(_good_profile(tmp_path))
    torch_check = _named(checks, "torch")
    assert torch_check.ok is False and torch_check.fix
    assert launcher.can_run_model(checks) is False


def test_a_missing_checkpoint_is_a_pass_with_a_download_warning(tmp_path):
    """setup.ensure_checkpoint downloads rather than failing, so this is not a
    blocker. It IS a surprise on a laptop, so the size is stated."""
    prof = _good_profile(tmp_path)
    (pathlib.Path(prof["checkpoint_dir"]) / "sam2.1_hiera_large.pt").unlink()
    check = _named(launcher.machine_checks(prof), "checkpoint")
    assert check.ok is True
    assert "download" in check.detail.lower()


def test_an_uncreatable_checkpoint_dir_fails(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory", encoding="utf-8")
    prof = dict(_good_profile(tmp_path), checkpoint_dir=str(blocker))
    check = _named(launcher.machine_checks(prof), "checkpoint")
    assert check.ok is False and launcher.can_run_model(
        launcher.machine_checks(prof)) is False


def test_a_missing_raw_em_fails_with_the_reason_from_pipeline(tmp_path):
    prof = dict(_good_profile(tmp_path), worm_path=str(tmp_path / "gone"))
    check = _named(launcher.machine_checks(prof), "raw EM")
    assert check.ok is False and "gone" in check.detail


def test_raw_em_does_not_block_review_only_work(tmp_path):
    """Recrop needs the raw EM; re-predict and resume propagation do not. A missing
    stack must not disable the model outright."""
    prof = dict(_good_profile(tmp_path), worm_path="")
    assert launcher.can_run_model(launcher.machine_checks(prof)) is True


def test_an_unwritable_frames_cache_fails(tmp_path):
    blocker = tmp_path / "cache_is_a_file"
    blocker.write_text("x", encoding="utf-8")
    prof = dict(_good_profile(tmp_path), frames_root=str(blocker))
    assert _named(launcher.machine_checks(prof), "frames cache").ok is False


def test_the_writability_probe_leaves_nothing_behind(tmp_path):
    prof = _good_profile(tmp_path)
    launcher.machine_checks(prof)
    assert list(pathlib.Path(prof["frames_root"]).iterdir()) == []


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_the_device_is_reported_and_never_blocks(monkeypatch, tmp_path, device):
    monkeypatch.setattr(launcher, "_device_name", lambda: device)
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert check.ok is True and device in check.detail


def test_mps_carries_the_upstream_caveat(monkeypatch, tmp_path):
    """SAM2 on MPS is preliminary upstream and may differ from CUDA. A reviewer
    comparing her masks with the batch's needs to know that."""
    monkeypatch.setattr(launcher, "_device_name", lambda: "mps")
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert "preliminary" in check.detail.lower()


def test_slow_devices_say_so(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_device_name", lambda: "cpu")
    check = _named(launcher.machine_checks(_good_profile(tmp_path)), "device")
    assert "slow" in check.detail.lower()


def test_one_broken_check_does_not_hide_the_others(monkeypatch, tmp_path):
    """machine_checks catches per check, the same reason validate_bundle returns a
    list: a reviewer should see everything wrong at once."""
    def _boom():
        raise RuntimeError("torch import exploded")
    monkeypatch.setattr(launcher, "torch_available", _boom)
    checks = launcher.machine_checks(_good_profile(tmp_path))
    assert len(checks) == 5
    assert _named(checks, "torch").ok is False
    assert _named(checks, "raw EM").ok is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `py -3 -m pytest tests/test_launcher_machine_checks.py -v`
Expected: FAIL with `AttributeError: module 'launcher' has no attribute 'machine_checks'`.

- [ ] **Step 3: Implement the checks**

In `launcher.py`, add after `torch_available` and before `load_profile`:

```python
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


def _device_name() -> str:
    """The device SAM2 would run on: 'cuda', 'mps' or 'cpu'.

    A named module function rather than an inline call so a test can replace it
    without a GPU, and so the torch import stays inside it.
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
        if not torch_available():
            return MachineCheck("torch", False, "not installed",
                                "install the full requirements: py -3 -m pip install -r "
                                "requirements.txt")
        import torch
        return MachineCheck("torch", True, f"version {torch.__version__}")

    def _device():
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
        ckpt_dir = Path(profile.get("checkpoint_dir") or cfg_mod.CHECKPOINT_DIR)
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
        root = Path(profile.get("frames_root") or cfg_mod.FRAMES_ROOT)
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
```

Add `from dataclasses import dataclass` to the imports at the top of `launcher.py`, beside the existing `import json` block.

- [ ] **Step 4: Run the tests**

Run: `py -3 -m pytest tests/test_launcher_machine_checks.py -v`
Expected: PASS, fifteen tests.

- [ ] **Step 5: Wire the button and the mode gate into the window**

In `run()`, replace the existing torch gating block:

```python
    has_torch = torch_available()
    if not has_torch:
        mode_combo.model().item(1).setEnabled(False)
        mode_combo.setToolTip("Reprop needs torch, which is not installed on this machine.")
```

with a gate that asks the checks, and re-runs when a path changes:

```python
    def _machine_profile():
        return {**profile,
                "frames_root": frames_edit.text().strip(),
                "worm_path": worm_edit.text().strip(),
                "checkpoint_dir": ckpt_edit.text().strip()}

    def refresh_mode_gate(*_):
        """Enable full mode only when this machine can honestly run it."""
        checks = machine_checks(_machine_profile())
        ok = can_run_model(checks)
        mode_combo.model().item(1).setEnabled(ok)
        if not ok:
            missing = ", ".join(c.name for c in checks
                                if not c.ok and c.name in ("torch", "checkpoint"))
            mode_combo.setToolTip(f"Reprop needs {missing} on this machine.")
            mode_combo.setCurrentIndex(0)
        else:
            mode_combo.setToolTip("")
        return checks

    def do_check(*_):
        lines = []
        for c in machine_checks(_machine_profile()):
            mark = "ok  " if c.ok else "NO  "
            lines.append(f"{mark}{c.name}: {c.detail}")
            if c.fix:
                lines.append(f"      fix: {c.fix}")
        summary.setPlainText("\n".join(lines))
        refresh_mode_gate()

    check_btn = QPushButton("Check this machine")
    layout.addWidget(check_btn)
    check_btn.clicked.connect(do_check)
    for edit in (worm_edit, frames_edit, ckpt_edit):
        edit.editingFinished.connect(refresh_mode_gate)
```

Place the `check_btn` line beside the existing `render_btn`, and call `refresh_mode_gate()` once next to the existing `refresh()` call near the end of `run()`. Keep the existing `want_full` line, but read the gate rather than `has_torch`:

```python
    want_full = profile.get("ui_mode", "review") == "full"
    mode_combo.setCurrentIndex(1 if (want_full and can_run_model(
        machine_checks(_machine_profile()))) else 0)
```

- [ ] **Step 6: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add launcher.py tests/test_launcher_machine_checks.py
git commit -m "launcher: preflight what this machine can do, before the session"
```

---

### Task 4: Keep a recropped bundle a bundle

**Files:**
- Modify: `sam2_utils/bundle.py` (add `is_bundle` and `adopt_chain_frames`)
- Modify: `sam2_utils/chain_meta.py` (add `refresh_meta`)
- Modify: `launcher.py:139` (`_summary_text`, use `is_bundle`)
- Test: `tests/test_bundle_adopt_frames.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `bundle.is_bundle(root: Path) -> bool`; `bundle.adopt_chain_frames(chain_dir: Path, frames_dir, *, relative: str = "frames") -> str`; `chain_meta.refresh_meta(chain_dir: Path, state: dict) -> Optional[Path]`. Task 5 calls all three.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bundle_adopt_frames.py`:

```python
"""bundle.adopt_chain_frames + chain_meta.refresh_meta: what a recrop must fix up.

A recrop prepares frames in the machine's frames cache and records an absolute path.
Inside a bundle that breaks two things at once: validate_bundle rejects an absolute
frames_dir, and the bundle stops opening anywhere else. These are the two repairs.

    py -3 -m pytest tests/test_bundle_adopt_frames.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils import bundle, chain_meta


def _view(tmp_path, name="view", n=3):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"{i:05d}.jpg").write_bytes(b"jpg")
    return d


def _chain_dir(tmp_path):
    d = tmp_path / "AIAL" / "chain_00"
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "00000.jpg").write_bytes(b"old")
    return d


class TestIsBundle:
    def test_a_directory_with_a_manifest_is_a_bundle(self, tmp_path):
        (tmp_path / bundle.BUNDLE_MANIFEST).write_text("{}", encoding="utf-8")
        assert bundle.is_bundle(tmp_path) is True

    def test_a_plain_output_tree_is_not(self, tmp_path):
        assert bundle.is_bundle(tmp_path) is False


class TestAdoptChainFrames:
    def test_the_new_frames_land_inside_the_chain_dir(self, tmp_path):
        chain_dir, view = _chain_dir(tmp_path), _view(tmp_path)
        rel = bundle.adopt_chain_frames(chain_dir, view)
        assert rel == "frames"
        assert sorted(p.name for p in (chain_dir / "frames").glob("*.jpg")) == [
            "00000.jpg", "00001.jpg", "00002.jpg"]

    def test_the_old_frames_are_gone_not_mixed_in(self, tmp_path):
        """A leftover frame from the old window would be drawn under a mask from the
        new one, and nothing about the directory would look wrong."""
        chain_dir = _chain_dir(tmp_path)
        for i in range(9):
            (chain_dir / "frames" / f"{i:05d}.jpg").write_bytes(b"old")
        bundle.adopt_chain_frames(chain_dir, _view(tmp_path, n=3))
        assert len(list((chain_dir / "frames").glob("*.jpg"))) == 3

    def test_the_source_is_moved_not_copied(self, tmp_path):
        chain_dir, view = _chain_dir(tmp_path), _view(tmp_path)
        bundle.adopt_chain_frames(chain_dir, view)
        assert not view.exists()

    def test_an_empty_source_refuses(self, tmp_path):
        """A failed prep must not empty the chain's frames and leave the bundle
        unopenable."""
        chain_dir = _chain_dir(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            bundle.adopt_chain_frames(chain_dir, empty)
        assert (chain_dir / "frames" / "00000.jpg").exists()

    def test_adopting_the_frames_already_in_place_is_a_no_op(self, tmp_path):
        chain_dir = _chain_dir(tmp_path)
        rel = bundle.adopt_chain_frames(chain_dir, chain_dir / "frames")
        assert rel == "frames" and (chain_dir / "frames" / "00000.jpg").exists()


class TestRefreshMeta:
    def _write_meta(self, chain_dir, crop_window):
        meta = chain_meta.build_meta(
            {"neuron": "AIAL", "chain_idx": 0, "crop_window": crop_window,
             "config": {"save_downscale": 8}},
            neuron_id=7, source_tree="reprop_maskseed", chain_dir=chain_dir)
        chain_meta.write_meta(chain_dir, meta)

    def test_a_new_window_reaches_meta_json(self, tmp_path):
        chain_dir = _chain_dir(tmp_path)
        self._write_meta(chain_dir, {"origin_tif": [0, 0], "size_tif": [1024, 1024],
                                     "crop_scale": 2, "sam_scale": 8})
        new_state = {"neuron": "AIAL", "chain_idx": 0,
                     "crop_window": {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                                     "crop_scale": 4, "sam_scale": 8},
                     "config": {"save_downscale": 8}}
        chain_meta.refresh_meta(chain_dir, new_state)
        meta = chain_meta.read_meta(chain_dir)
        assert meta["crop_window"]["size_tif"] == [2048, 2048]
        assert meta["mask_scale"] == 4, "a recrop can change crop_scale, and an exporter "\
                                        "that kept the old one misplaces every pixel"

    def test_identity_and_provenance_survive(self, tmp_path):
        """refresh_meta must not need the registry: the chain's identity is already
        recorded, and inventing a new neuron_id would be worse than failing."""
        chain_dir = _chain_dir(tmp_path)
        self._write_meta(chain_dir, None)
        chain_meta.refresh_meta(chain_dir, {"neuron": "AIAL", "chain_idx": 0,
                                            "crop_window": None,
                                            "config": {"save_downscale": 8}})
        meta = chain_meta.read_meta(chain_dir)
        assert meta["neuron_id"] == 7
        assert meta["provenance"]["source_tree"] == "reprop_maskseed"

    def test_a_chain_with_no_meta_is_left_alone(self, tmp_path):
        """A plain output tree does not always carry meta.json, and there is nothing
        to keep in sync there."""
        chain_dir = _chain_dir(tmp_path)
        assert chain_meta.refresh_meta(chain_dir, {"neuron": "AIAL", "chain_idx": 0,
                                                   "crop_window": None,
                                                   "config": {}}) is None
        assert not (chain_dir / chain_meta.META_FILENAME).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `py -3 -m pytest tests/test_bundle_adopt_frames.py -v`
Expected: FAIL with `AttributeError: module 'sam2_utils.bundle' has no attribute 'is_bundle'`.

- [ ] **Step 3: Implement the bundle helpers**

In `sam2_utils/bundle.py`, add after `is_recorded_absolute`:

```python
def is_bundle(root) -> bool:
    """Whether ``root`` is a review bundle rather than a plain output tree.

    The manifest is the marker. Callers used to spell this inline, and the two
    behaviours that depend on it (frames stay relative, geometry comes home) are too
    important to be decided by a copy of an expression.
    """
    return (Path(root) / BUNDLE_MANIFEST).exists()


def adopt_chain_frames(chain_dir, frames_dir, *, relative: str = "frames") -> str:
    """Move a freshly prepared frame view into a bundle chain dir, returning its
    relative name.

    A recrop prepares frames in the machine's frames cache and records an absolute
    path. Inside a bundle that breaks two things at once: ``validate_bundle`` rejects an
    absolute ``frames_dir``, and the bundle stops opening on any other machine. Moving
    the view in and recording it relative keeps the bundle self-contained.

    Moves rather than copies: the source is a cache ``prepare_chain_crop_frames``
    rebuilds fresh anyway, and a duplicated tier-2 chain's frames are real disk on a
    laptop.

    Parameters
    ----------
    chain_dir : Path
        The bundle's chain directory.
    frames_dir : path-like
        The prepared view to adopt.
    relative : str, optional
        The name to give it inside the chain directory.

    Returns
    -------
    str
        ``relative``, ready to store as the state's ``frames_dir``.

    Raises
    ------
    ValueError
        If the source holds no prepared frames. Replacing a chain's frames with an
        empty directory would leave the bundle unopenable, which is worse than a
        failed recrop.
    """
    import shutil

    chain_dir, frames_dir = Path(chain_dir), Path(frames_dir)
    if not sorted(frames_dir.glob("*.jpg")):
        raise ValueError(f"{frames_dir} holds no prepared frames; refusing to replace "
                         f"{chain_dir / relative} with an empty view")
    dest = chain_dir / relative
    if dest.exists() and frames_dir.resolve() == dest.resolve():
        return relative
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(frames_dir), str(dest))
    return relative
```

In `sam2_utils/chain_meta.py`, add after `read_meta`:

```python
def refresh_meta(chain_dir: Path, state: dict) -> Optional[Path]:
    """Rewrite ``<chain_dir>/meta.json`` for a chain whose geometry just changed.

    A recrop changes ``mask_space``, ``mask_scale``, ``crop_window`` and ``z_range``,
    every one of which this record projects from ``state.json``. Identity and
    provenance are carried over from the existing record, so this never needs the
    registry and can never invent a new neuron id.

    Parameters
    ----------
    chain_dir : Path
        The chain directory.
    state : dict
        The chain's new ``state.json``, already parsed.

    Returns
    -------
    Path or None
        The written path, or None when the chain has no ``meta.json``. A plain output
        tree does not always carry one, and there is nothing to keep in sync there.
    """
    chain_dir = Path(chain_dir)
    if not (chain_dir / META_FILENAME).exists():
        return None
    old = read_meta(chain_dir)
    prov = old.get("provenance", {}) or {}
    meta = build_meta(state, neuron_id=old["neuron_id"],
                      source_tree=prov.get("source_tree", ""),
                      backend=prov.get("backend", "sam2"),
                      reprop_variant=prov.get("reprop_variant"),
                      chain_dir=chain_dir)
    return write_meta(chain_dir, meta)
```

In `launcher.py`'s `_summary_text`, use the new helper:

```python
    problems = bundle.validate_bundle(root) if bundle.is_bundle(root) else []
```

- [ ] **Step 4: Run the tests**

Run: `py -3 -m pytest tests/test_bundle_adopt_frames.py -v`
Expected: PASS, eleven tests.

- [ ] **Step 5: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sam2_utils/bundle.py sam2_utils/chain_meta.py launcher.py tests/test_bundle_adopt_frames.py
git commit -m "bundle: adopt a recrop's frames, and keep meta.json in step with them"
```

---

### Task 5: Recrop inside a bundle leaves a valid bundle

**Files:**
- Modify: `gui.py`, `_recrop_to_window` (the block after `run_chain` that saves state)
- Test: `tests/test_gui_recrop_in_bundle.py` (create)

**Interfaces:**
- Consumes: `bundle.is_bundle`, `bundle.adopt_chain_frames`, `chain_meta.refresh_meta` (Task 4); `pipeline.raw_em_problem` guard (Task 2).
- Produces: the on-disk contract Task 6 reads: after a recrop in a bundle, `state.json` has a relative `frames_dir` and the new `crop_window`, and `meta.json` matches.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_recrop_in_bundle.py`:

```python
"""A recrop inside a bundle must leave a bundle, not a broken one.

run_chain prepares frames in the machine's cache and save_state records an absolute
path. In a bundle that fails validate_bundle and stops the bundle opening anywhere
else, so the GUI adopts the frames and rewrites the record. In a plain output tree
none of that applies and behaviour is unchanged.

run_chain is stubbed: this is about the bookkeeping around it, not about SAM2.

    py -3 -m pytest tests/test_gui_recrop_in_bundle.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui
from sam2_utils import bundle, chain_meta


def _bundle_chain(tmp_path, *, manifest=True):
    root = tmp_path / "tree"
    chain_dir = root / "AIAL" / "chain_00"
    (chain_dir / "frames").mkdir(parents=True)
    (chain_dir / "frames" / "00000.jpg").write_bytes(b"old")
    if manifest:
        (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
            {"schema_version": 1, "source_tree": "t", "chains": []}), encoding="utf-8")
    meta = chain_meta.build_meta(
        {"neuron": "AIAL", "chain_idx": 0,
         "crop_window": {"origin_tif": [0, 0], "size_tif": [1024, 1024],
                         "crop_scale": 2, "sam_scale": 8},
         "config": {"save_downscale": 8}},
        neuron_id=7, source_tree="t", chain_dir=chain_dir)
    chain_meta.write_meta(chain_dir, meta)
    return root, chain_dir


def _new_view(tmp_path):
    d = tmp_path / "cache" / "AIAL_chain00_s4"
    d.mkdir(parents=True)
    (d / "00000.jpg").write_bytes(b"new")
    return d


def _harness(monkeypatch, tmp_path, root, chain_dir):
    """A ReviewGUI stand-in carrying only what _recrop_to_window touches."""
    view = _new_view(tmp_path)
    new_window = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
                  "crop_scale": 4, "sam_scale": 8}
    saved = {}

    class _State:
        neuron, chain_idx = "AIAL", 0
        frames_dir = str(view)
        crop_window = new_window

    def _run_chain(state, **kwargs):
        state.frames_dir = str(view)
        state.crop_window = new_window

    monkeypatch.setattr(gui.pipeline, "run_chain", _run_chain)
    monkeypatch.setattr(gui.pipeline, "ChainState", lambda **kw: _State())
    monkeypatch.setattr(gui.pipeline, "save_state",
                        lambda state, path: saved.update(
                            path=path, frames_dir=state.frames_dir))
    monkeypatch.setattr(gui.pipeline, "state_to_dict",
                        lambda state: {"neuron": "AIAL", "chain_idx": 0,
                                       "crop_window": state.crop_window,
                                       "config": {"save_downscale": 8}})
    monkeypatch.setattr(gui.pipeline, "raw_em_problem", lambda *a, **k: None)

    # cfg must be a real PipelineConfig: _recrop_to_window calls dataclasses.replace on
    # it, which raises on anything that is not a dataclass.
    self = types.SimpleNamespace(
        ctx=types.SimpleNamespace(output_root=root, image_predictor=None,
                                  video_predictor=None, annotate_df=None,
                                  cfg=gui.pipeline.PipelineConfig(),
                                  ensure_predictors=lambda **kw: None),
        neuron="AIAL", chain_idx=0, chain={}, _cw=None,
        queue=types.SimpleNamespace(set_status=lambda *a, **k: None),
        reviewer="lucinda",
        _close_session=lambda: None,
        open_chain=lambda *a, **k: None,
    )
    # cw_new is printed via cw_new.size_tif, so a bare object() raises before the code
    # under test is reached.
    cw_new = types.SimpleNamespace(size_tif=[2048, 2048])
    return self, saved, view, cw_new


def test_a_recrop_in_a_bundle_records_a_relative_frames_dir(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved["frames_dir"] == "frames"


def test_the_new_frames_are_inside_the_bundle(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert (chain_dir / "frames" / "00000.jpg").read_bytes() == b"new"
    assert not view.exists()


def test_meta_json_follows_the_new_window(monkeypatch, tmp_path):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    meta = chain_meta.read_meta(chain_dir)
    assert meta["crop_window"]["size_tif"] == [2048, 2048]
    assert meta["mask_scale"] == 4


def test_an_output_tree_keeps_the_absolute_path(monkeypatch, tmp_path):
    """Outside a bundle the frames cache is exactly where frames belong, and moving
    them into the tree would break every other chain that shares the cache."""
    root, chain_dir = _bundle_chain(tmp_path, manifest=False)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved["frames_dir"] == str(view)
    assert view.exists()


def test_recrop_refuses_without_the_raw_em(monkeypatch, tmp_path, capsys):
    root, chain_dir = _bundle_chain(tmp_path)
    self, saved, view, cw_new = _harness(monkeypatch, tmp_path, root, chain_dir)
    monkeypatch.setattr(gui.pipeline, "raw_em_problem",
                        lambda *a, **k: "no raw EM path set")
    gui.ReviewGUI._recrop_to_window(self, cw_new, "test")
    assert saved == {}, "nothing may be written when the recrop cannot run"
    assert "raw EM" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -3 -m pytest tests/test_gui_recrop_in_bundle.py -v`
Expected: FAIL. `saved["frames_dir"]` is the absolute cache path, because nothing adopts the frames yet.

- [ ] **Step 3: Wire it into the GUI**

In `gui.py`, `_recrop_to_window`, replace the block that runs from `chain_dir = self.ctx.output_root / ...` through `pipeline.save_state(...)` with:

```python
        chain_dir = self.ctx.output_root / self.neuron / f"chain_{self.chain_idx:02d}"
        # Inside a bundle the new frames must come with it: run_chain leaves them in the
        # machine's frames cache and records an absolute path, which validate_bundle
        # rejects and which stops the bundle opening anywhere else. In an output tree the
        # cache IS where frames belong, and moving them would break the chains that share
        # it, so this runs only for a bundle.
        if bundle_mod.is_bundle(self.ctx.output_root) and state.frames_dir:
            state.frames_dir = bundle_mod.adopt_chain_frames(chain_dir, state.frames_dir)
        pipeline.save_state(state, chain_dir / "state.json")
        # meta.json projects mask_space, mask_scale, crop_window and z_range from the
        # state, and a recrop invalidates all four.
        chain_meta.refresh_meta(chain_dir, pipeline.state_to_dict(state))
```

Add `chain_meta` to the `sam2_utils` imports at the top of `gui.py`, beside the existing `bundle as bundle_mod` import.

- [ ] **Step 4: Run the test**

Run: `py -3 -m pytest tests/test_gui_recrop_in_bundle.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass. `tests/test_gui_recrop_persists_state.py` exercises the same method; if it stubs `save_state` without `state_to_dict`, extend that stub rather than weakening this code.

- [ ] **Step 6: Commit**

```bash
git add gui.py tests/test_gui_recrop_in_bundle.py
git commit -m "gui: a recrop inside a bundle leaves a valid, self-contained bundle"
```

---

### Task 6: Carry the recrop home

**Files:**
- Modify: `sam2_utils/bundle.py` (add `GEOMETRY_FIELDS`, `geometry_of`, `merge_geometry`)
- Modify: `import_bundle.py` (the per-chain loop in `import_bundle`, and its report)
- Modify: `gui.py`, `_ensure_local_frames` (the `resolved` / `have_recorded` lines)
- Test: `tests/test_import_bundle_geometry.py` (create)

**Interfaces:**
- Consumes: the on-disk contract from Task 5.
- Produces: `bundle.GEOMETRY_FIELDS`, `bundle.geometry_of(state) -> dict`, `bundle.merge_geometry(master_state, bundle_state) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_import_bundle_geometry.py`:

```python
"""A recropped chain's geometry must come home with its masks.

import_bundle merges masks and qc.csv and deliberately never copies state.json, which
carries a bundle-relative frames_dir. A recrop, though, rewrites the chain's GEOMETRY,
so masks-only means the master tree keeps describing the old window and every consumer
places the new pixels at the old offsets, silently.

    py -3 -m pytest tests/test_import_bundle_geometry.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sam2_utils import bundle

OLD_WINDOW = {"origin_tif": [0, 0], "size_tif": [1024, 1024],
              "crop_scale": 2, "sam_scale": 8}
NEW_WINDOW = {"origin_tif": [10, 10], "size_tif": [2048, 2048],
              "crop_scale": 4, "sam_scale": 8}


def _state(window, *, frames_dir, frame_to_z=None, n_frames=3, anchor=1, extra=None):
    state = {"neuron": "AIAL", "chain_idx": 0, "crop_window": window,
             "frames_dir": frames_dir, "n_frames": n_frames,
             "anchor_frame_idx": anchor,
             "frame_to_z": frame_to_z or {"0": 1500, "1": 1501, "2": 1502},
             "config": {"output_root": "/master", "frames_root": "/master/frames"},
             "qc_summary": {"flagged": 2}}
    if extra:
        state.update(extra)
    return state


class TestGeometryOf:
    def test_it_reads_exactly_the_recrop_sensitive_fields(self):
        assert set(bundle.GEOMETRY_FIELDS) == {
            "crop_window", "frame_to_z", "n_frames", "anchor_frame_idx"}

    def test_two_states_with_the_same_window_compare_equal(self):
        a = _state(OLD_WINDOW, frames_dir="/a")
        b = _state(OLD_WINDOW, frames_dir="/b")
        assert bundle.geometry_of(a) == bundle.geometry_of(b), \
            "frames_dir differs on every machine and must not read as a recrop"

    def test_a_new_window_compares_different(self):
        assert bundle.geometry_of(_state(OLD_WINDOW, frames_dir="/a")) != \
            bundle.geometry_of(_state(NEW_WINDOW, frames_dir="/a"))


class TestMergeGeometry:
    def test_the_new_window_replaces_the_old(self):
        merged = bundle.merge_geometry(_state(OLD_WINDOW, frames_dir="/master/f"),
                                       _state(NEW_WINDOW, frames_dir="frames"))
        assert merged["crop_window"] == NEW_WINDOW

    def test_the_config_block_never_crosses_machines(self):
        """state.json embeds the whole PipelineConfig, including output_root and
        frames_root. Copying it back writes the reviewer's Mac paths into the master
        tree."""
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        incoming = _state(NEW_WINDOW, frames_dir="frames")
        incoming["config"] = {"output_root": "/Users/lucinda/bundle",
                              "frames_root": "/Users/lucinda/frames"}
        merged = bundle.merge_geometry(master, incoming)
        assert merged["config"] == master["config"]

    def test_the_stale_frames_dir_is_cleared(self):
        """The _pcrop view dir is namespaced by neuron, chain and crop scale, so a new
        window at the same scale REUSES the directory name. The master's recorded dir
        still exists and still holds the old window's frames."""
        merged = bundle.merge_geometry(_state(OLD_WINDOW, frames_dir="/master/f"),
                                       _state(NEW_WINDOW, frames_dir="frames"))
        assert merged["frames_dir"] is None

    def test_reviewer_owned_summaries_are_not_touched_here(self):
        """qc.csv is copied as a file by the existing REVIEWER_OWNED path; the state's
        qc_summary belongs to the master's own run and is not part of this merge."""
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        incoming = _state(NEW_WINDOW, frames_dir="frames",
                          extra={"qc_summary": {"flagged": 99}})
        assert bundle.merge_geometry(master, incoming)["qc_summary"] == {"flagged": 2}

    def test_the_master_state_is_not_mutated_in_place(self):
        master = _state(OLD_WINDOW, frames_dir="/master/f")
        bundle.merge_geometry(master, _state(NEW_WINDOW, frames_dir="frames"))
        assert master["crop_window"] == OLD_WINDOW

    def test_a_sam_chain_with_no_window_merges_without_error(self):
        merged = bundle.merge_geometry(_state(None, frames_dir="/master/f"),
                                       _state(None, frames_dir="frames"))
        assert merged["crop_window"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `py -3 -m pytest tests/test_import_bundle_geometry.py -v`
Expected: FAIL with `AttributeError: module 'sam2_utils.bundle' has no attribute 'GEOMETRY_FIELDS'`.

- [ ] **Step 3: Implement the allowlist**

In `sam2_utils/bundle.py`, add after `REVIEWER_LEDGERS`:

```python
#: state.json fields a recrop changes, and the only ones import merges back. The whole
#: `config` block is deliberately absent: it carries output_root and frames_root, which
#: are the reviewer's machine's paths, not the master tree's. An allowlist rather than a
#: denylist, so a field added to ChainState later cannot quietly start crossing machines.
GEOMETRY_FIELDS = ("crop_window", "frame_to_z", "n_frames", "anchor_frame_idx")
```

and after `rewrite_state_frames_dir`:

```python
def geometry_of(state: dict) -> dict:
    """The recrop-sensitive slice of a parsed ``state.json``.

    Comparing this, rather than the whole file, is what separates "she recropped this
    chain" from "these two files were written on different machines": ``frames_dir``,
    ``config`` and the timing blocks differ on every machine and mean nothing here.
    """
    return {k: state.get(k) for k in GEOMETRY_FIELDS}


def merge_geometry(master_state: dict, bundle_state: dict) -> dict:
    """``master_state`` with the bundle's geometry merged in, as a new dict.

    Clears ``frames_dir``, which is the part that is easy to miss.
    ``prepare_chain_crop_frames`` namespaces its view directory by neuron, chain and
    crop scale, so a new window at the SAME scale reuses the same directory name: the
    master's recorded path still exists and still holds the old window's frames, and
    the GUI would draw the new masks on them with nothing missing to signal it. None
    makes the next open regenerate from the raw EM, which the master machine has.
    """
    merged = dict(master_state)
    merged.update(geometry_of(bundle_state))
    merged["frames_dir"] = None
    return merged
```

- [ ] **Step 4: Run the bundle tests**

Run: `py -3 -m pytest tests/test_import_bundle_geometry.py -v`
Expected: PASS, nine tests.

- [ ] **Step 5: Write the failing import test**

Append to `tests/test_import_bundle_geometry.py`:

```python
class TestImportCarriesItHome:
    def _tree(self, tmp_path, name, window, *, frames_dir, source_tree="t"):
        from sam2_utils import chain_meta
        root = tmp_path / name
        chain_dir = root / "AIAL" / "chain_00"
        (chain_dir / "masks").mkdir(parents=True)
        (chain_dir / "masks" / "mask_1500.png").write_bytes(b"png" + name.encode())
        (chain_dir / "qc.csv").write_text("frame,flag\n0,0\n", encoding="utf-8")
        (chain_dir / "state.json").write_text(
            json.dumps(_state(window, frames_dir=frames_dir)), encoding="utf-8")
        chain_meta.write_meta(chain_dir, chain_meta.build_meta(
            {"neuron": "AIAL", "chain_idx": 0, "crop_window": window,
             "config": {"save_downscale": 8}},
            neuron_id=7, source_tree=source_tree, chain_dir=chain_dir))
        return root, chain_dir

    def _bundle(self, tmp_path, window):
        root, chain_dir = self._tree(tmp_path, "bundle", window, frames_dir="frames")
        (chain_dir / "frames").mkdir()
        (root / bundle.BUNDLE_MANIFEST).write_text(json.dumps(
            {"schema_version": 1, "source_tree": "master",
             "chains": [{"cell_name": "AIAL", "chain_idx": 0,
                         "chain_dir": "AIAL/chain_00"}]}), encoding="utf-8")
        for rel in bundle.BUNDLE_DATA_FILES:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("[]" if rel.endswith(".json") else "node_id\n", encoding="utf-8")
        # import_bundle merges the two root ledgers row-wise after the per-chain files.
        # If _read_ledger turns out to tolerate their absence, drop these two lines. Do
        # NOT weaken import_bundle to make the test pass.
        for name in bundle.REVIEWER_LEDGERS:
            (root / name).write_text("", encoding="utf-8")
        return root

    def test_a_recropped_chain_updates_the_masters_window(self, tmp_path):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["crop_window"] == NEW_WINDOW
        assert state["frames_dir"] is None
        assert state["config"]["output_root"] == "/master", \
            "the reviewer's config must never land in the master tree"

    def test_the_masters_meta_json_is_replaced(self, tmp_path):
        import import_bundle as ib
        from sam2_utils import chain_meta
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        assert chain_meta.read_meta(master_chain)["crop_window"] == NEW_WINDOW

    def test_an_unrecropped_chain_leaves_the_state_alone(self, tmp_path):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, OLD_WINDOW)
        ib.import_bundle(b, master, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["frames_dir"] == "/master/f", \
            "a plain mask edit must not invalidate the master's frames"

    def test_a_dry_run_reports_the_recrop_and_writes_nothing(self, tmp_path, capsys):
        import import_bundle as ib
        master, master_chain = self._tree(tmp_path, "master", OLD_WINDOW,
                                          frames_dir="/master/f")
        b = self._bundle(tmp_path, NEW_WINDOW)
        ib.import_bundle(b, master, dry_run=True, allow_tree_mismatch=True)
        state = json.loads((master_chain / "state.json").read_text(encoding="utf-8"))
        assert state["crop_window"] == OLD_WINDOW
        out = capsys.readouterr().out
        assert "recrop" in out.lower() and "2048" in out
```

- [ ] **Step 6: Run it to verify it fails**

Run: `py -3 -m pytest tests/test_import_bundle_geometry.py -v -k ImportCarries`
Expected: FAIL. The master's `crop_window` is still `OLD_WINDOW` after the import.

- [ ] **Step 7: Merge the geometry in import_bundle**

In `import_bundle.py`, add this helper above `import_bundle`:

```python
def _window_label(window) -> str:
    """A crop window's size for a report line, or a name for having none."""
    if not window:
        return "_sam (no crop window)"
    w, h = window.get("size_tif", ["?", "?"])
    return f"{w}x{h} _tif at crop_scale {window.get('crop_scale', '?')}"
```

In the per-chain loop, immediately after the `for name in bundle.REVIEWER_OWNED:` block and before `if chain_changed:`, insert:

```python
        # A recrop rewrites the chain's GEOMETRY, and masks alone would land in the
        # master tree describing the window they were NOT drawn in. Merge the allowlist
        # field by field, never the file: state.json also carries the reviewer's config,
        # whose output_root and frames_root are her machine's.
        src_state_path, dst_state_path = src_dir / "state.json", dst_dir / "state.json"
        if src_state_path.exists() and dst_state_path.exists():
            src_state = json.loads(src_state_path.read_text(encoding="utf-8"))
            dst_state = json.loads(dst_state_path.read_text(encoding="utf-8"))
            if bundle.geometry_of(src_state) != bundle.geometry_of(dst_state):
                chain_changed = True
                recropped.append((rec["chain_dir"],
                                  _window_label(dst_state.get("crop_window")),
                                  _window_label(src_state.get("crop_window"))))
                if not dry_run:
                    merged = bundle.merge_geometry(dst_state, src_state)
                    dst_state_path.write_text(json.dumps(merged, indent=2),
                                              encoding="utf-8")
                    src_meta = src_dir / chain_meta.META_FILENAME
                    if src_meta.exists():
                        shutil.copy2(src_meta, dst_dir / chain_meta.META_FILENAME)
```

Declare `recropped: List[tuple] = []` beside `changed: List[dict] = []`, and add to the report, after the existing per-chain loop that prints `changed`:

```python
    if recropped:
        print(f"[import] {verb} the crop window on {len(recropped)} chain(s); their frames "
              f"will be re-prepared from the raw EM on next open")
        for chain_dir, before, after in recropped:
            print(f"  {chain_dir}: {before} -> {after}")
```

- [ ] **Step 8: Run the import tests**

Run: `py -3 -m pytest tests/test_import_bundle_geometry.py -v`
Expected: PASS, thirteen tests.

- [ ] **Step 9: Write the failing frames-regeneration test**

Create `tests/test_ensure_local_frames_none.py`:

```python
"""A cleared frames_dir means 'regenerate', not 'crash'.

import_bundle clears the master's frames_dir when a chain comes home recropped, because
the recorded view directory would otherwise be reused with the OLD window's frames. That
makes None a state the GUI now reaches on a normal path.

    py -3 -m pytest tests/test_ensure_local_frames_none.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


def test_a_cleared_frames_dir_regenerates_instead_of_raising(monkeypatch, tmp_path):
    prepared = {}

    def _prepare(chain, annotate_df, *, scale, frames_root, anchor_catmaid_z,
                 neuron, chain_idx, z_range=None):
        prepared["called"] = True
        return str(tmp_path / "fresh"), {0: 1500}, 0, 1

    monkeypatch.setattr(gui.pipeline, "prepare_video_frames", _prepare)
    frames_dir, frame_to_z, anchor_idx = gui._ensure_local_frames(
        None, {0: 1500}, 0, chain={}, cw=None,
        cfg=types.SimpleNamespace(scale=8, frames_root=tmp_path),
        annotate_df=None, anchor_catmaid_z=1500, neuron="AIAL", chain_idx=0,
        chain_dir=tmp_path)
    assert prepared["called"] is True
    assert frame_to_z == {0: 1500} and anchor_idx == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 10: Run it to verify it fails**

Run: `py -3 -m pytest tests/test_ensure_local_frames_none.py -v`
Expected: FAIL with a `TypeError` from `resolved / "00000.jpg"`, since `resolved` is None.

- [ ] **Step 11: Harden the frames resolution**

In `gui.py`, `_ensure_local_frames`, replace:

```python
    resolved = resolve_frames_dir(recorded_frames_dir, chain_dir) if chain_dir else \
        Path(recorded_frames_dir)
    have_recorded = (resolved / "00000.jpg").exists()
```

with:

```python
    # None is a real state, not a bug: import_bundle clears frames_dir when a chain comes
    # home recropped, precisely so the stale view (same directory name, old window's
    # frames) is regenerated instead of silently reused.
    if not recorded_frames_dir:
        resolved = None
    else:
        resolved = (resolve_frames_dir(recorded_frames_dir, chain_dir) if chain_dir
                    else Path(recorded_frames_dir))
    have_recorded = resolved is not None and (resolved / "00000.jpg").exists()
```

- [ ] **Step 12: Run the full suite and lint**

Run: `py -3 -m pytest -q` then `ruff check .`
Expected: all pass.

- [ ] **Step 13: Commit**

```bash
git add sam2_utils/bundle.py import_bundle.py gui.py tests/test_import_bundle_geometry.py tests/test_ensure_local_frames_none.py
git commit -m "import: carry a recropped chain's geometry home, never its config"
```

---

### Task 7: Say it in the docs a reviewer actually reads

**Files:**
- Modify: `docs/how-to/review-flagged-chains.md` (§1 Launch, and §5f Recrop)
- Modify: `docs/CHANGELOG.md` (a new dated entry plus its Contents line)
- Modify: `F:\Lucinda_Review\START_HERE.md` if F: is mounted; if it is not, say so in the commit body rather than skipping silently

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Document the machine settings in the how-to**

In `docs/how-to/review-flagged-chains.md` §1, after the launcher is introduced, add:

```markdown
### Machine settings (recrop and SAM2)

The launcher's **This machine** group holds the three paths the model needs, none of
which travel in a bundle:

- **Raw EM (tif stack)**: the full-resolution `..z<number>.tif` frames. Recrop re-reads
  them to cut a new window, so recrop is the only thing that needs this.
- **Frames cache**: where prepared JPEG frames are written. Needs to be writable and
  roomy; a tier-2 chain's frames are unique to its window and are not shared.
- **Checkpoints**: where the SAM2 weights live or land. If the checkpoint is absent it
  downloads on first use, about 2.4 GB for the `large` model.

**Check this machine** reports each one, plus whether torch is installed and which
device SAM2 would run on. On Apple Silicon that is `mps`, which works but is slow, and
which upstream still calls preliminary, so masks can differ slightly from a CUDA run.
Reprop stays disabled until torch and a usable checkpoint folder are both present.
```

- [ ] **Step 2: Document recrop in a bundle in §5f**

Append to §5f:

```markdown
**Recropping inside a review bundle.** This works, and the bundle stays a bundle: the
new frames are moved into the chain's own `frames/` folder and recorded relative, so it
still opens on any machine. When you send the bundle back, `import_bundle` notices the
new crop window and carries it home with the masks, then clears the master tree's
recorded frames so they are re-prepared in the right window. It reports every chain
whose window changed, since that is a bigger event than a mask edit.
```

- [ ] **Step 3: Add the CHANGELOG entry**

Add this Contents line at the top of the entry list in `docs/CHANGELOG.md`:

```markdown
- [2026-08-27, a second reviewer's machine, and a recrop that survives the round trip](#r-2026-08-27-machine-setup-recrop)
```

and this entry immediately above the `<a id="r-2026-08-26-gui-image-cleanup">` anchor:

```markdown
<a id="r-2026-08-27-machine-setup-recrop"></a>
## 2026-08-27, a second reviewer's machine, and a recrop that survives the round trip

A review bundle opens on a machine that has none of this project's data, and it does
that well. What never travelled is everything the model needs: the raw EM tif stack that
recrop re-reads, the frames cache it writes to, and the checkpoint. All three were env
vars only a shell could set, so recrop was unreachable on a reviewer's own machine and
full mode unlocked on a bare `import torch`, which meant a laptop with torch and no
checkpoint offered the reprop controls and failed minutes later.

The launcher now records all three and runs a preflight that says what is missing and
how to fix it, including which device SAM2 would actually use. On Apple Silicon that is
MPS, which works, is slow, and is preliminary upstream, so a reviewer comparing her
masks against the batch's deserves to be told.

**The part that was not asked for.** Recrop rewrites a chain's geometry, and geometry is
exactly what the round trip did not carry. `import_bundle` moves masks and qc.csv and
deliberately never copies state.json, because it holds a bundle-relative frames_dir.
A recrop in a bundle therefore came home as masks alone, into a master tree whose
state.json still described the old window, and every consumer placed the new pixels at
the old offsets with nothing to signal it. The same failure `_recrop_to_window`'s own
comments warn about, one level out.

Sharper still: `prepare_chain_crop_frames` namespaces its view directory by neuron,
chain and crop scale, so a new window at the same scale reuses the same directory name.
The master's recorded frames_dir would still exist, still holding the old window's
frames. So the import clears it, and a cleared frames_dir became a state the GUI has to
handle rather than crash on.

Two decisions worth keeping. **No schema bump**: the bundles already delivered are
version 1, and gating the geometry merge on a version would silently drop a recrop made
in a bundle a reviewer already holds. Import compares geometry per chain instead.
**Geometry merges, `config` never does**: state.json embeds the whole PipelineConfig,
including output_root and frames_root, so copying it back would write a reviewer's Mac
paths into the master tree. The allowlist is the mechanism that prevents that, which is
why it is a named constant with a test that asserts its exact contents.
```

- [ ] **Step 4: Update START_HERE.md if F: is mounted**

Check first: `py -3 -c "import pathlib; print(pathlib.Path(r'F:\Lucinda_Review').exists())"`.
If True, add the machine-settings section to `F:\Lucinda_Review\START_HERE.md` in the
same voice as the rest of that file. If False, do not invent the change: report it in
the commit body and in the final summary, since that file is not in git and cannot be
fixed later from history.

- [ ] **Step 5: Check the repo rule and commit**

Run: `py -3 -c "import pathlib,sys; bad=[p for p in [pathlib.Path('docs/CHANGELOG.md'), pathlib.Path('docs/how-to/review-flagged-chains.md')] if chr(8212) in p.read_text(encoding='utf-8')]; print(bad); sys.exit(1 if bad else 0)"`
Expected: `[]` and exit 0.

```bash
git add docs/CHANGELOG.md docs/how-to/review-flagged-chains.md
git commit -m "docs: machine settings, and what a recrop inside a bundle does"
```
