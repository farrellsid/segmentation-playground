# Contributing

This is a research codebase for one lab. The goal of these notes is that a new student can set up,
make a change, and check it without tribal knowledge. For the bigger picture, read
[docs/tutorials/getting-started.md](docs/tutorials/getting-started.md) first.

## Set up

```bash
pip install -e ".[dev]"      # library, test deps, and linters (pytest, scipy, scikit-image, ruff, import-linter)
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Install the CUDA build of torch that matches your machine from pytorch.org first; it is left out of
the dependency list on purpose. Create a `.env` with `CATMAID_TOKEN=...` and point the paths in
`sam2_utils/config.py` at your data.

## Run the tests

The suite is CPU-only and does not load a SAM2 checkpoint, so it runs on any box:

```bash
py -3 -m pytest
```

Keep new tests for pure geometry or pure pandas torch-free, so they stay in this fast suite.

## Conventions

- Docstrings follow NumPy style. The linter checks that public functions have one.
- No em dashes in code, comments, docs, or commit messages. Use commas, colons, parentheses, or
  separate sentences.
- Tag coordinates with their space suffix and route conversions through `sam2_utils/alignment.py`.
  See [docs/reference/coordinate-spaces.md](docs/reference/coordinate-spaces.md).
- Lint and check structure before a PR:

```bash
ruff check .
ruff format --check .
lint-imports          # enforces the dependency direction below
```

## Dependency direction

The library (`pipeline.py`, `sam2_utils/`) must not import the drivers (`batch.py`, `gui.py`,
`run_aval.py`) or `eval/`. Drivers and `eval/` import the library, never the reverse. This is
enforced by import-linter. Shared logic between two drivers belongs in the library. See
[ADR 0001](docs/adr/0001-library-plus-thin-drivers.md) and
[ADR 0011](docs/adr/0011-flat-layout-over-src.md).

## Adding a pipeline phase

See [docs/how-to/add-a-pipeline-phase.md](docs/how-to/add-a-pipeline-phase.md). In short: add a small
function in `pipeline.py`, put any tunable on `PipelineConfig` with a behavior-preserving default,
wire it into `run_chain`, keep `run_aval.py` reproducible, and add a torch-free test.

## Recording a decision

When you make a load-bearing decision, write a short ADR in [docs/adr/](docs/adr/README.md). Do not
edit a landed ADR; supersede it with a new one. Keep the running build log in
[docs/CHANGELOG.md](docs/CHANGELOG.md).

## Good first tasks

The active backlog lives in [docs/explanation/design-notes.md](docs/explanation/design-notes.md) and
the forward research plan in [docs/explanation/roadmap.md](docs/explanation/roadmap.md). Low-risk
starting points:

- Add a QC signal and a torch-free test for it (see [docs/reference/qc-signals.md](docs/reference/qc-signals.md)).
- Add `doctest` examples to a pure function in `sam2_utils/alignment.py`.
- Fix one of the GUI usability items noted in the design notes backlog.
