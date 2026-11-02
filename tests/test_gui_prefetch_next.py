"""_prefetch_next: fires a background prep of whatever `next CHAIN` would open
right now, using a stand-in for threading.Thread that runs synchronously so
the tests are deterministic. See
docs/superpowers/specs/2026-10-01-next-chain-frame-prefetch-design.md.

    py -3 -m pytest tests/test_gui_prefetch_next.py
"""
from __future__ import annotations

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gui


class _SyncThread:
    """Runs its target immediately on .start(), on the calling thread, so
    prefetch tests don't race a real background thread."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _fake_self(tmp_path, *, peek, prep_locks=None):
    return types.SimpleNamespace(
        _peek_chain=lambda direction: peek,
        ctx=types.SimpleNamespace(output_root=tmp_path, cfg="CFG", annotate_df="DF"),
        anchor_only=True, context_frames=2,
        _prep_locks=prep_locks if prep_locks is not None else {})


def test_prefetch_resolves_frames_for_the_peeked_chain(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, "STATE", None))
    monkeypatch.setattr(gui, "_resolve_chain_frames",
                        lambda state, **kw: calls.append((state, kw)) or (None, None, None))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)

    assert len(calls) == 1
    state, kw = calls[0]
    assert state == "STATE"
    assert kw["neuron"] == "AIAL" and kw["chain_idx"] == 1
    assert kw["anchor_only"] is True and kw["context_frames"] == 2
    assert ("AIAL", 1) in fake_self._prep_locks


def test_prefetch_is_a_noop_when_there_is_nothing_to_peek(monkeypatch, tmp_path):
    def _must_not_be_called(*a, **k):
        raise AssertionError("no thread should be started when _peek_chain returns None")
    monkeypatch.setattr(gui.threading, "Thread", _must_not_be_called)

    fake_self = _fake_self(tmp_path, peek=None)
    gui.ReviewGUI._prefetch_next(fake_self)   # must not raise, must not touch threading.Thread


def test_prefetch_is_a_noop_when_the_peeked_chain_has_no_state_yet(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, None, None))
    monkeypatch.setattr(gui, "_resolve_chain_frames",
                        lambda state, **kw: calls.append((state, kw)))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 9))
    gui.ReviewGUI._prefetch_next(fake_self)

    assert calls == [], "a never-run chain (no state.json) has nothing to prefetch"


def test_prefetch_swallows_errors_from_the_background_thread(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)

    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gui, "_load_chain_state_and_cw", _boom)

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)   # must not raise

    out = capsys.readouterr().out
    assert "background prefetch" in out and "AIAL" in out


def test_prefetch_setup_failure_does_not_escape_into_open_chain(monkeypatch, tmp_path, capsys):
    """_peek_chain calls self.queue.refresh() under the hood, which can raise on
    real disk flakiness this repo has hit before. open_chain has already fully
    succeeded by the time _prefetch_next runs (it's the last line of open_chain),
    so a failure here must degrade to "no prefetch this time," never escape
    _prefetch_next and fail a chain that already opened fine. No thread should
    even be started: the failure is in the setup step, before one would spawn."""
    def _must_not_be_called(*a, **k):
        raise AssertionError("no thread should be started when _peek_chain itself raises")
    monkeypatch.setattr(gui.threading, "Thread", _must_not_be_called)

    def _raising_peek(direction):
        raise RuntimeError("simulated queue.refresh() disk flakiness")

    fake_self = types.SimpleNamespace(
        _peek_chain=_raising_peek,
        ctx=types.SimpleNamespace(output_root=tmp_path, cfg="CFG", annotate_df="DF"),
        anchor_only=True, context_frames=2, _prep_locks={})

    gui.ReviewGUI._prefetch_next(fake_self)   # must not raise

    out = capsys.readouterr().out
    assert "background prefetch" in out


def test_prefetch_reuses_the_same_lock_object_for_the_same_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(gui.threading, "Thread", _SyncThread)
    monkeypatch.setattr(gui, "_load_chain_state_and_cw",
                        lambda ctx, neuron, chain_idx, chain_dir: ({"nodes": []}, "STATE", None))
    monkeypatch.setattr(gui, "_resolve_chain_frames", lambda state, **kw: (None, None, None))

    fake_self = _fake_self(tmp_path, peek=("AIAL", 1))
    gui.ReviewGUI._prefetch_next(fake_self)
    lock_first = fake_self._prep_locks[("AIAL", 1)]
    gui.ReviewGUI._prefetch_next(fake_self)
    lock_second = fake_self._prep_locks[("AIAL", 1)]

    assert lock_first is lock_second


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
