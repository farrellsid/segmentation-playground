# How to run a headless batch

This covers running the pipeline over many chains without the GUI. It assumes you have set up the
environment (see [../tutorials/getting-started.md](../tutorials/getting-started.md)) and pointed the
paths in `sam2_utils/config.py` at your data.

## Run with a preset

Run configuration lives in named presets in `sam2_utils/presets.py`. A preset bundles the worm, the
paths, the model, the tier-2 settings, and a default neuron set. Pick one with `--preset` and
override any field with a flag.

```bash
py -3 batch.py                                  # the original preset (target worm)
py -3 batch.py --preset original --neurons AVAL AVAR
py -3 batch.py --preset original --neurons AVAL --clean
```

The batch builds the predictors once, runs every selected chain through `run_chain`, and writes
`output/_manifest.csv`, `_triage.csv`, and `_timing.csv`. See
[../reference/state-and-storage.md](../reference/state-and-storage.md) for the output layout.

## Resume and clean

Re-running resumes where it left off: a chain already `done` or `flagged` is skipped, and an
interrupted chain is retried. `--clean` wipes prior outputs first. It is scope-aware: with no
`--neurons` it does a full reset, otherwise it deletes only the named neurons' outputs and prunes
their rows from the manifest.

If you change a QC or gate threshold between runs, clear or re-score the manifest. It is append-mode,
so otherwise early and late chains silently mix two configs.

## Scope a partial run

```bash
py -3 batch.py --preset original --neurons AVAL AVAR     # explicit neurons
py -3 batch.py --preset original --neuron-limit 3        # first N neurons by sorted name
```

## What comes out

After a run, `output/_manifest.csv` lists each chain and its status. Chains QC flagged are the ones a
human reviews next. To review them, see [review-flagged-chains.md](review-flagged-chains.md).

## Review a finished chain read-only

To eyeball a chain without the full GUI, use the read-only viewer from a notebook or REPL:

```python
from pathlib import Path
from sam2_utils import review, config
chain_dir = Path(config.OUTPUT_ROOT) / "AVAL" / "chain_02"
review.grid_flagged(chain_dir)                  # only the QC-flagged frames
review.to_gif(chain_dir, chain_dir / "aval.gif")
```
