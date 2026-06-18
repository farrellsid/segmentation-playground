# How to evaluate against ground truth

This covers scoring the pipeline's predictions against the cross-worm ground truth (a different worm
with matching EM and confirmed segments). The harness lives in `eval/`. For why these metrics and the
forward plan, see [../explanation/roadmap.md](../explanation/roadmap.md) and
[ADR 0010](../adr/0010-erl-voi-eval-ruler.md).

## Run the pipeline on the GT worm

The same `batch.py` runs on the GT worm through the `eval` preset. It swaps the EM source through a
frame-store seam and bakes the skeleton-to-image transform from the registration; no per-chain code
changes. The full run is guarded, so you must pass a scope.

```bash
py -3 batch.py --preset eval --neurons URYVL          # explicit neuron(s)
py -3 batch.py --preset eval --neuron-limit 3         # first N neurons
py -3 batch.py --preset eval --all                    # everything (opt-in; large)
```

## Score the predictions

```bash
py -3 -m eval.score_batch --preset eval
```

This scores predicted masks against the manually verified GT and writes per-neuron region metrics
(IoU, Dice, precision, recall), the labelmap metrics (VOI split and merge, ARAND), and per-neuron
ERL, plus timing and a `measurement_log.jsonl` provenance record. The CSVs land in the preset's out
directory: `eval_neurons.csv`, `eval_labelmap.csv`, `eval_frames.csv`, `eval_timing.csv`.

## Which metric to trust

For this sparse, per-neuron pipeline the primary measures are per-neuron region IoU, precision and
recall, plus ERL. VOI and ARAND are secondary here: they are built for dense whole-volume
segmentation, so on a scored-neuron subset they are not comparable to dense benchmarks. See
[ADR 0010](../adr/0010-erl-voi-eval-ruler.md) and `eval/README.md`.

## Before trusting any number

The skeleton-to-GT registration both places prompts and samples node labels, so a loose registration
poisons every score. Verify the registration first. The cross-worm GT measures generalization, not
in-distribution accuracy, so spot-check gains on the target worm.
