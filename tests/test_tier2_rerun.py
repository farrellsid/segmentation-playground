"""Unit tests for batch._should_tier2_rerun — the trigger for decision 1
(PIPELINE_CONTEXT §8.5 / §8.7): a chain flagged in the _sam pass is re-run once with
tier-2 per-chain cropping.

Torch-free: batch imports pipeline/sam2_utils.setup, but all of them defer torch to
call-time, so importing the module and exercising the pure decision function (and the
`replace(cfg, chain_crop=True)` override the second pass relies on) needs no GPU/torch.

Run either way:
    py -3 -m pytest tests/test_tier2_rerun.py
    py -3 tests/test_tier2_rerun.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import batch
from pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# the trigger truth table
# ---------------------------------------------------------------------------

def test_flagged_sam_chain_triggers_rerun():
    # the canonical case: flagged after a _sam first pass, knob on -> re-run as tier-2
    assert batch._should_tier2_rerun(batch.FLAGGED, cfg_chain_crop=False,
                                     tier2_on_flagged=True) is True


def test_done_chain_never_reruns():
    # a clean _sam chain is fine as-is; tier-2 is for flagged chains only
    assert batch._should_tier2_rerun(batch.DONE, cfg_chain_crop=False,
                                     tier2_on_flagged=True) is False


def test_already_tier2_chain_never_reruns():
    # cfg.chain_crop already on -> the first (only) pass WAS tier-2; do not loop
    assert batch._should_tier2_rerun(batch.FLAGGED, cfg_chain_crop=True,
                                     tier2_on_flagged=True) is False


def test_knob_off_disables_rerun():
    # legacy single-pass _sam behaviour when the knob is off
    assert batch._should_tier2_rerun(batch.FLAGGED, cfg_chain_crop=False,
                                     tier2_on_flagged=False) is False


def test_non_flagged_statuses_never_rerun():
    # only FLAGGED triggers; failed/pending/running/done all decline
    for status in (batch.FAILED, batch.PENDING, batch.RUNNING, batch.DONE):
        assert batch._should_tier2_rerun(status, cfg_chain_crop=False,
                                         tier2_on_flagged=True) is False


def test_returns_plain_bool():
    # callers compare with `is`; keep it a real bool, not a truthy object
    out = batch._should_tier2_rerun(batch.FLAGGED, False, True)
    assert isinstance(out, bool)


# ---------------------------------------------------------------------------
# tier2_all — "tier-2 everywhere" test mode (re-run every completed _sam chain)
# ---------------------------------------------------------------------------

def test_tier2_all_reruns_done_and_flagged():
    # a clean _sam chain that wouldn't normally re-run DOES under tier2_all
    assert batch._should_tier2_rerun(batch.DONE, cfg_chain_crop=False,
                                     tier2_on_flagged=False, tier2_all=True) is True
    assert batch._should_tier2_rerun(batch.FLAGGED, cfg_chain_crop=False,
                                     tier2_on_flagged=False, tier2_all=True) is True


def test_tier2_all_still_skips_incomplete_and_already_tier2():
    # never re-run a chain that didn't finish a _sam pass, or one already tier-2
    for status in (batch.FAILED, batch.PENDING, batch.RUNNING):
        assert batch._should_tier2_rerun(status, cfg_chain_crop=False,
                                         tier2_on_flagged=False, tier2_all=True) is False
    assert batch._should_tier2_rerun(batch.DONE, cfg_chain_crop=True,
                                     tier2_on_flagged=False, tier2_all=True) is False


def test_tier2_all_default_off_preserves_flagged_only():
    # with tier2_all unset, behaviour is exactly the flagged-only default
    assert batch._should_tier2_rerun(batch.DONE, cfg_chain_crop=False,
                                     tier2_on_flagged=True) is False
    assert batch._should_tier2_rerun(batch.FLAGGED, cfg_chain_crop=False,
                                     tier2_on_flagged=True) is True


# ---------------------------------------------------------------------------
# the config override the second pass uses
# ---------------------------------------------------------------------------

def test_replace_enables_chain_crop_without_mutating_original():
    """The second pass runs `replace(cfg, chain_crop=True)`; confirm it flips only
    chain_crop, leaves the original cfg untouched, and preserves every other knob
    (so the tier-2 pass keeps the same model_size, scale, qc_* thresholds, etc.)."""
    cfg = PipelineConfig(model_size="large", scale=8, save_downscale=8,
                         chain_crop=False, qc_triage_min_signals=2)
    cfg2 = dataclasses.replace(cfg, chain_crop=True)
    assert cfg.chain_crop is False                 # original not mutated
    assert cfg2.chain_crop is True                 # override applied
    # every other field carried through
    assert cfg2.model_size == cfg.model_size
    assert cfg2.scale == cfg.scale
    assert cfg2.save_downscale == cfg.save_downscale
    assert cfg2.qc_triage_min_signals == cfg.qc_triage_min_signals
    # and the fallback that makes the second pass regression-free is still on
    assert cfg2.chain_crop_fallback is True


# ---------------------------------------------------------------------------
# plain runner
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:                          # noqa: BLE001 - test runner
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
