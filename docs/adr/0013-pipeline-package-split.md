# 0013. The pipeline core is a package split by concern

Status: Accepted

## Context

ADR 0001 established the library-plus-thin-drivers shape, with the phase functions, the run-config
dataclass, the per-chain state, and the orchestrator all in a single `pipeline.py`. That file grew to
about 2,200 lines holding eight separable concerns (frame I/O, config, state + serialization, prompt
build and image prediction and mask selection, tier-2 crop and frame prep, video propagation, mask
save and postprocess, QC, and the `run_chain` orchestrator). One oversized module is the one fair
"why is this so long" reaction from a newcomer and works against "one concern, one home": to find where
a phase lives you scroll, and an edit to one concern sits next to four others.

## Decision

Make `pipeline` a package whose submodules each own one concern (`config`, `state`, `frames`, `masks`,
`predict`, `crop`, `propagate`, `qc`, `orchestrator`), and have `pipeline/__init__.py` re-export the
full public surface (plus the underscore helpers the tests and eval reach). Every consumer still does
`import pipeline` / `from pipeline import X` unchanged; the package is the public contract.

This is a pure structural move, no logic changed. Submodules import siblings by relative import in a
leaf-first acyclic order (`config` is a leaf; `orchestrator` sits on top); the lazy `import torch` stays
inside the prediction and propagation functions so importing the package stays cheap and torch-free.
`pipeline.config` (the run-knob dataclass) is kept distinct from `sam2_utils.config` (static paths and
constants), and submodules never shadow one with the other.

This applies ADR 0001 and the "one concern, one home" principle rather than reversing anything; it
supersedes no prior decision.

## Consequences

- A newcomer finds a phase by file name, not by scrolling; each submodule is small enough to hold in
  one's head. The code map names the submodule per concern.
- The public API is unchanged, so no driver, GUI, eval module, or test was edited (only the
  import-direction test's file glob, which now scans `pipeline/*.py`).
- The split is verified by the existing torch-free suite, ruff (including undefined-name checks across
  the package), and the import-direction check; behavior preservation of `run_chain` end to end is
  confirmed by a single-chain `run_aval` smoke, since the unit tests do not exercise the GPU path.
- New phase code now has an obvious home; the risk is a submodule slowly accreting unrelated helpers,
  caught by the same code-map discipline that motivated the split.
