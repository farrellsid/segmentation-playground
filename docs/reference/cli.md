# Command-line reference

The four entry points and their common flags. All run from the repo root.

## run_aval.py

Runs one chain end to end. The path and chain knobs are set at the top of the file; there is no
argparse. Edit those, then:

```bash
py -3 run_aval.py
```

## batch.py

The headless batch over many chains, with resume.

| Flag | Effect |
|------|--------|
| `--preset <name>` | Use a named preset (`original` or `eval`). Defaults to `original`. |
| `--neurons A B ...` | Run only these neurons. |
| `--neuron-limit N` | Run the first N neurons by sorted name. |
| `--all` | Run everything (opt-in guard for the eval preset's large run). |
| `--clean` | Wipe prior outputs first (scope-aware). |
| `--model-size <size>` | Override the model (tiny, small, base_plus, large). |
| `--output-root <dir>` | Override the output tree. |
| `--frames-root <dir>` | Override the frame cache/view root. |
| `--gif-mode <off\|flagged\|all>` | Override the preset's overlay-gif policy. |
| `--no-tier2` | Disable the tier-2 auto second-pass on flagged chains. |
| `--postprocess` / `--no-postprocess` | Force mask post-processing on/off (overrides the preset). Use the pair to A/B outputs with vs without cleanup; see [configuration.md](configuration.md). |

See [../how-to/run-a-batch.md](../how-to/run-a-batch.md).

## gui.py

The napari review and correction GUI.

| Flag | Effect |
|------|--------|
| `--neuron <name> --chain <i>` | Open straight onto one chain. |
| `--output-root <dir>` | Which output tree to review. |
| `--reviewer <name>` | Stamp labels with a reviewer name. |
| `--point-size <n>` | Prompt point size in grid pixels. |
| `--no-auto-zoom` | Do not auto-zoom to the mask on open. |
| `--hires-em` | Load the full-resolution EM as the background. |

See [../how-to/review-flagged-chains.md](../how-to/review-flagged-chains.md).

## gui_neuron.py

The napari NEURON-level review GUI (second paradigm): opens a whole neuron, its branches
as one multi-color object on a per-neuron crop canvas. See
[../how-to/review-a-neuron.md](../how-to/review-a-neuron.md).

| Flag | Effect |
|------|--------|
| `--neuron <name>` | Open straight onto one neuron. |
| `--output-root <dir>` | Which output tree to review. |
| `--reviewer <name>` | Stamp labels with a reviewer name. |

## eval.score_batch

Scores a batch run against ground truth.

```bash
py -3 -m eval.score_batch --preset eval        # root and out auto-resolved from the preset
```

Common flags: `--preset <name>`, `--no-labelmap` (skip the VOI/ARAND/ERL labelmap metrics, region
metrics only). See [../how-to/evaluate-against-gt.md](../how-to/evaluate-against-gt.md). Other `eval/`
entry points (`eval.run_erl`, `eval.registration`, `eval.diag_registration`,
`eval.scale_registration`) are documented in `eval/README.md`.

## eval.merge_metric

Ground-truth-free severe-bleed / dropout scorer for the target worm (roadmap Phase 0). For each run's
RAW per-chain masks it counts foreign skeleton nodes contained (a merge) and own-node dropout, scored
against the worm's own CATMAID skeletons, so it needs no cross-worm GT and no boundary labels. Writes a
`_merge_metric.csv` into each run tree and prints one summary line per run.

```bash
py -3 -m eval.merge_metric --root <merged_run> [--root <other> ...] [--radius N]
```

Summary columns: `foreign_frame_rate` (fraction of frames whose mask contains another neuron's node),
`dropout_rate` (fraction whose mask lost its own node), `total_foreign`. This is a severe-merge floor
(foreign-node containment at `--radius`, default 3 grid px), so it does not catch mild bleed. Repeat
`--root` to compare runs on a single node-table load.
