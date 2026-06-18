# segmentation-playground

Tooling for running [SAM2](https://github.com/facebookresearch/sam2) segmentation on
electron-microscopy image stacks from the Zhen Lab *C. elegans* connectome dataset. The job: segment
about 300 neurons (a few thousand maximal-linear-chains) out of a roughly 300-slice `.tif` stack into
per-neuron mask volumes for export to Blender.

It is a semi-automatic, human-in-the-loop pipeline. The machine runs and QC-scores every chain
headless; a human reviews and corrects only the frames QC flags. Everything runs locally on one
Windows and GPU box, on the filesystem, with no server or database.

## Start here

- New to the project? Read [docs/tutorials/getting-started.md](docs/tutorials/getting-started.md).
- Want to know where to change something? See [docs/reference/code-map.md](docs/reference/code-map.md).
- Want the design and the why? See [docs/explanation/architecture.md](docs/explanation/architecture.md)
  and the [ADRs](docs/adr/README.md).

## Setup

```bash
pip install -e .                                          # the library and runtime deps
pip install 'git+https://github.com/facebookresearch/sam2.git'   # SAM2 is not on PyPI
```

Install the CUDA build of torch that matches your machine from pytorch.org first (it is left out of
the dependency list on purpose). Create a `.env` at the repo root with `CATMAID_TOKEN=your_token`
(no quotes, no spaces). Point the paths in `sam2_utils/config.py` at your data: `WORM_PATH` for the
raw stack, and `OUTPUT_ROOT` and `FRAMES_ROOT` for the output and scratch trees.

## Quick start

```bash
py -3 run_aval.py                              # run one chain end to end (start here)
py -3 batch.py --preset original --neurons AVAL   # headless batch over selected chains
py -3 gui.py                                   # review and correct flagged chains
py -3 -m eval.score_batch --preset eval        # score against the cross-worm ground truth
py -3 -m pytest                                # the CPU-only test suite
```

How-to guides: [run a batch](docs/how-to/run-a-batch.md),
[review flagged chains](docs/how-to/review-flagged-chains.md),
[evaluate against GT](docs/how-to/evaluate-against-gt.md),
[add a pipeline phase](docs/how-to/add-a-pipeline-phase.md).

## Repository layout

```
segmentation-playground/
  pipeline.py            # the library: phase functions, PipelineConfig, ChainState, run_chain
  run_aval.py            # driver: run one chain (the regression harness)
  batch.py               # driver: headless batch over all chains, with resume
  gui.py                 # driver: napari review / correction GUI
  pull_worm.py           # utility: pull a worm's annotations from CATMAID
  sam2_utils/            # importable helpers (config, setup, alignment, qc, review, labels, ...)
  eval/                  # ground-truth evaluation (region IoU, VOI, ARAND, ERL, registration)
  finetune/              # scaffold for SAM2 finetuning (not yet implemented)
  tests/                 # unit tests (the suite is CPU-only)
  experiments/           # one-off A/B harnesses, kept for reproducibility
  notebooks/             # reference and exploration notebooks
  archive/               # superseded material, kept for reference
  data/                  # (git-ignored) CATMAID-derived inputs and the GT landing spot
  docs/                  # documentation (see below)
```

## Documentation

The docs follow the four Diátaxis types, so each page has one job:

- `docs/tutorials/` learning-oriented walkthroughs (start here).
- `docs/how-to/` task-oriented guides for a specific job.
- `docs/reference/` neutral lookup: the [code map](docs/reference/code-map.md),
  [configuration](docs/reference/configuration.md), [CLI](docs/reference/cli.md),
  [coordinate spaces](docs/reference/coordinate-spaces.md),
  [state and storage](docs/reference/state-and-storage.md),
  [QC signals](docs/reference/qc-signals.md).
- `docs/explanation/` the why: [architecture](docs/explanation/architecture.md), the
  [roadmap](docs/explanation/roadmap.md), and the detailed
  [design notes and backlog](docs/explanation/design-notes.md).
- `docs/adr/` numbered architecture decision records.
- `docs/CHANGELOG.md` the build history.

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).
