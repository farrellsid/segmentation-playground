# How to add a pipeline phase

A phase is one step in `run_chain`: select the anchor, build prompts, image predict, box from mask,
prepare frames, propagate, postprocess, save, run QC. This guide shows how to add or change one. The
shape to follow is set by [ADR 0001](../adr/0001-library-plus-thin-drivers.md): phases are small
functions in the library that take arrays and config and return plain data.

## Write the phase function

Add the function in `pipeline.py`, next to the other phase functions. Follow the local conventions:

- Take inputs as plain arrays plus the values it needs from `PipelineConfig`. Do not read global
  state or build predictors inside the function.
- Tag every coordinate with its space suffix and route any conversion through `sam2_utils/alignment.py`.
  Do not write `/ scale` inline. See [../reference/coordinate-spaces.md](../reference/coordinate-spaces.md).
- Return plain data (arrays, scalars, dicts). Let `run_chain` decide what to persist.

## Add a knob, do not hardcode

If the phase needs a tunable, add a field to `PipelineConfig` with a default that preserves current
behavior. Knobs live on the config in one place, not scattered as literals. See
[../reference/configuration.md](../reference/configuration.md).

## Wire it into run_chain

Call the phase from `run_chain` in the right position in the order. If it produces something worth
keeping across a resume, add a field to `ChainState` (JSON-serializable) so it survives. See
[../reference/state-and-storage.md](../reference/state-and-storage.md).

## Keep the baseline reproducible

The single-chain driver `run_aval.py` reproduces the reference output. A new phase should default to
off or to behavior-preserving values so that baseline still matches. Turn the new behavior on through
its knob, then measure.

## Test it

Add a unit test under `tests/`. If the phase is pure geometry or pure pandas, keep the test torch-free
so it runs without a GPU (the existing `test_alignment.py` and `test_labels.py` are the models). Run:

```bash
py -3 -m pytest
```

## Measure before you trust it

A new accuracy lever earns its place by shrinking the review queue or improving the eval metrics, not
by seeming reasonable. Run it through the ground-truth eval (see [evaluate-against-gt.md](evaluate-against-gt.md))
before treating it as a win.
