# 0001. Library plus thin drivers

Status: Accepted

## Context

The project started as a single notebook that did everything for one chain: pull annotations, align,
seed SAM2, propagate, save, and visualize. As the work grew to cover a headless batch over thousands
of chains, a review GUI, and ground-truth evaluation, a single script could not carry all of it.
A notebook also hides execution-order state, resists testing, and cannot be imported by other tools.

## Decision

Split the code into a pure library plus thin drivers.

`pipeline.py` is the library. It holds the phase functions, a `PipelineConfig` of knobs, a
serializable `ChainState`, and `run_chain`, which runs one chain through the phases. It builds no
predictors and does no I/O setup. Running `python pipeline.py` does nothing.

Each driver imports the library and adds one entry point: `run_aval.py` for a single chain,
`batch.py` for the headless batch, `gui.py` for review, and `eval/` for scoring. Shared logic lives
in the library, never in a driver.

## Consequences

- The phase functions are small and testable. A new developer reads `run_chain` once and sees the
  whole flow.
- The single-chain driver doubles as a regression harness: it reproduces the reference notebook
  output, so a refactor can prove it changed nothing.
- The library does not import any driver. This keeps the dependency graph acyclic and is enforced in
  CI (see [0011](0011-flat-layout-over-src.md) and `tests/test_import_direction.py`).
- A reader who wants to know "what does the pipeline do" reads the library; "how do I run it" reads a
  driver. The two questions have two homes.
