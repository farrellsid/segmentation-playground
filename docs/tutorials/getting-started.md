# Getting started

This page is for a new developer on the project. It gives a reading order that gets you oriented in
under an hour, then walks you through your first run. Follow it top to bottom the first time. You do
not need to make any decisions along the way.

## Read these, in this order

1. [../explanation/architecture.md](../explanation/architecture.md). What the project is, the
   library-plus-drivers shape, and the principles behind it. Read the whole thing.
2. [../reference/code-map.md](../reference/code-map.md). The table that tells you which file owns
   which concern. You will come back to this constantly.
3. `PipelineConfig` at the top of `pipeline.py`. This dataclass is every knob a run can turn. Skim
   the fields and their defaults; you do not need to understand each one yet.
4. `run_chain` in `pipeline.py`. This is the orchestrator. Read it once to see the phase order:
   select anchor, build prompts, image predict, box from mask, prepare frames, propagate,
   postprocess, save, run QC.
5. One phase function, for example `image_predict`. See how a phase takes arrays and config, does one
   job, and returns plain data. The phases are all shaped this way.
6. [../reference/coordinate-spaces.md](../reference/coordinate-spaces.md). The space suffixes
   (`_tif`, `_sam`, `_crop`, `_pcrop`, `_cm`) show up everywhere. Learn them before you touch any
   geometry.

That is the core. The QC signals ([../reference/qc-signals.md](../reference/qc-signals.md)) and the
storage layout ([../reference/state-and-storage.md](../reference/state-and-storage.md)) are worth a
skim once the above makes sense.

## Set up the environment

You need Python 3.9 or newer, a CUDA GPU for any real run, and the SAM2 package, which is not on
PyPI.

```bash
pip install -e .              # the library and its runtime deps
pip install -e ".[test]"      # add the test dependencies (pytest, scipy, scikit-image)
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

Install the CUDA build of torch that matches your machine from pytorch.org first. It is left out of
the dependency list on purpose, because the right build depends on your CUDA version.

Create a `.env` file at the repo root with your CATMAID token, no quotes and no spaces:

```
CATMAID_TOKEN=your_token_here
```

Point the paths in `sam2_utils/config.py` at your data: `WORM_PATH` for the raw stack, and
`OUTPUT_ROOT` and `FRAMES_ROOT` for the output and JPEG-scratch trees.

## Check the install without a GPU

The test suite is CPU-only. It never loads a SAM2 checkpoint, so it runs anywhere. If this passes,
your library install is sound:

```bash
py -3 -m pytest
```

You should see all tests pass.

## Run your first chain

`run_aval.py` runs a single chain end to end. It is the smallest real run and the regression harness
that reproduces the reference notebook output. Open it, check that the paths at the top resolve on
your machine, then:

```bash
py -3 run_aval.py
```

It prints the chain's final status, `done` or `flagged`, and writes the masks, `state.json`, and
`qc.csv` under `OUTPUT_ROOT`.

## Where to go next

- To run many chains, see [../how-to/run-a-batch.md](../how-to/run-a-batch.md).
- To review and correct flagged chains, see
  [../how-to/review-flagged-chains.md](../how-to/review-flagged-chains.md).
- To score against ground truth, see [../how-to/evaluate-against-gt.md](../how-to/evaluate-against-gt.md).
- To add a new pipeline phase, see [../../CONTRIBUTING.md](../../CONTRIBUTING.md).
