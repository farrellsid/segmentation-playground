# 0011. Flat layout over src layout

Status: Accepted

## Context

Python packaging offers two common shapes. The src layout puts importable code under `src/`, which
prevents accidental imports of the working copy and is the more defensive choice for a library that
is installed and distributed. The flat layout keeps modules at the repo root and is common in
scientific Python (NumPy, SciPy, Matplotlib use it). This project is a solo, single-box lab tool. Its
drivers are root-level modules run directly as `py -3 batch.py`, and `pipeline.py` is imported as a
top-level module.

## Decision

Keep the flat layout. Do not move code under `src/`.

Requiring an editable install before any script runs is real friction for a lab tool that people run
by invoking a file. The reference report that prompted this reorganization says to adopt src layout
only when the package is already installed that way and to skip it when the editable-install step is
friction for the users, which is the case here.

Make the package cleanly installable in place instead: `pip install -e .` works, the test config puts
the repo root on the path so `import pipeline` and `import eval` resolve, and the dependency
direction (drivers and eval import the library, never the reverse) is enforced by
`tests/test_import_direction.py` in CI.

## Consequences

- A new lab member runs `py -3 run_aval.py` without an install step, and the import paths resolve.
- The "this is the real library" boundary is drawn by the dependency-direction rule and the import
  contract rather than by a directory.
- If the project ever grows multiple external consumers or gets published, revisit this with a new
  ADR; the move to src layout is mechanical but disruptive, so it is not worth doing speculatively.
