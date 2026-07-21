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

## run_perframe.py

Per-frame neuron segmentation, for one or more CATMAID frames: Approach 1 (`prompt`),
image-mode SAM2 once per node, or Approach 2 (`amg`), `SAM2AutomaticMaskGenerator` matched to
nodes with the rest kept as competitors. Both share membrane-aware overlap resolution and F2
scoring. Writes `results/perframe/<run>/{config.json,scores.csv,montages/}` (gitignored,
regenerable) and appends one summary row to
[perframe-experiments.md](../explanation/perframe-experiments.md) (committed).

```bash
py -3 run_perframe.py --approach prompt --frames 1400 1420 --negatives on \
    --selection metric --resolver argmax --scale 8 --model-size tiny \
    --out results/perframe/smoke
```

| Flag | Effect |
|------|--------|
| `--approach prompt\|amg` | Segmentation approach. `prompt` (Approach 1, image-mode SAM2 per node) or `amg` (Approach 2, auto-mask generator + match to nodes). Default `prompt`. |
| `--frames Z [Z ...]` | CATMAID z's to segment. Required. |
| `--negatives on\|off` | Prompt only: pass other cells' nodes as negative points. Default `on`. Ignored with `--sweep` or `--approach amg`. |
| `--selection pred_iou\|generous\|metric` | Prompt only: how to pick among SAM2's 3 multimask candidates. Default `metric`. Ignored with `--sweep` or `--approach amg`. |
| `--match area\|metric` | Amg only: how to match a node to one of the AMG masks. `area` picks the smallest containing mask (no cross-node foreign exclusion, prefer `metric` when the match itself needs to be trusted); `metric` uses the F2 composite. Default `metric`. |
| `--amg-params <json>` | Amg only: JSON object overriding `DEFAULT_AMG_PARAMS` key-by-key, e.g. `'{"points_per_side": 32}'`. |
| `--resolver argmax\|watershed` | Overlap-resolution method (F3). Default `argmax`. Ignored with `--sweep`. |
| `--sweep` | Loop the Approach-1 knob grid (`negatives` x `selection` x `resolver`, 12 combos) over `--frames` instead of one combo. Each combo gets its own auto-named subdirectory of `--out` (e.g. `neg_on-sel_metric-res_argmax`) and its own experiments-log row. Prompt approach only; mutually exclusive with `--tune`. |
| `--tune` | Grid-search AMG params (`pred_iou_thresh` x `stability_score_thresh` x `points_per_side`, or `--tune-grid` override) over `--frames`, maximising `eval.perframe_score.objective`. Writes every trial to `<out>/trials.csv`, the winning params' montages/scores/config to `--out`, and a summary row (with a gameable-objective NOTE) to the experiments log. Amg only; mutually exclusive with `--sweep`. |
| `--tune-grid <json>` | JSON object overriding the default tune grid key-by-key, e.g. `'{"points_per_side": [32]}'` (values are lists). |
| `--scale N` | Downscale factor for the `_sam` grid. Default 8. |
| `--model-size <size>` | SAM2 checkpoint size. Default `tiny`. |
| `--out <dir>` | Results dir, e.g. `results/perframe/<run>`. Required. With `--sweep`, the parent dir under which each combo's subdirectory is written. |
| `--radius N` | Node-containment radius in grid px. Default 3. |
| `--tau T` | Membrane threshold on the normalised `[0, 1]` map. Default 0.5. |
| `--k-max-neg N` | Cap on negative points per node, nearest first. Default 3. |
| `--box-margin N` | Fixed px pad for the first-pass box. Default 10. |

Sweep example, 12 combos over one frame:

```bash
py -3 run_perframe.py --approach prompt --sweep --frames 1400 --scale 8 \
    --model-size tiny --out results/perframe/sweep_smoke
```

Tune example, default 12-combo AMG grid over one frame:

```bash
py -3 run_perframe.py --tune --frames 1400 --scale 8 \
    --model-size tiny --out results/perframe/tune_smoke
```

## eval.merge_metric

Ground-truth-free bleed / dropout scorer for the target worm (roadmap Phase 0 + Phase 2). For each
run's RAW per-chain masks it counts foreign skeleton nodes contained (a merge) and own-node dropout,
scored against the worm's own CATMAID skeletons, so it needs no cross-worm GT and no boundary labels.
By default it also runs a membrane-aware pass (reads the raw EM, see
[ADR 0016](../adr/0016-membrane-map-border-to-border-bleed-detection.md)) that catches mild bleed and
underfill, the two error classes foreign-node containment alone cannot see. Writes a
`_merge_metric.csv` into each run tree and prints one summary line per run.

```bash
py -3 -m eval.merge_metric --root <merged_run> [--root <other> ...] [--radius N] [--no-membrane] [--tau T] [--tol PX]
```

| Flag | Effect |
|------|--------|
| `--radius N` | Foreign/own-node containment radius in grid px (default 3). |
| `--no-membrane` | Skip the Phase-2 membrane pass (Phase-0-only, no EM reads). |
| `--tau T` | Membrane threshold on the normalised `[0, 1]` map (default 0.5). See [configuration.md](configuration.md). |
| `--tol PX` | Tolerance in px for `boundary_on_membrane` (default 2). See [configuration.md](configuration.md). |

Summary line fields: `foreign_frame_rate` (fraction of frames whose mask contains another neuron's
node), `dropout_rate` (fraction whose mask lost its own node), `total_foreign`, this part is the
Phase-0 severe-merge floor (foreign-node containment at `--radius`) and unaffected by `--no-membrane`.
When the membrane pass ran, the line also carries `mild_bleed_rate` (the headline: a spanning membrane
crossing with no foreign node, mild bleed the floor alone misses), `spanning_merge_rate` (spanning
membrane crossings regardless of foreign-node status), `mean_boundary_on_membrane`, and
`mean_underfill_fraction`. These four are absent from the line (and `None` in the returned summary)
when `--no-membrane` is passed or the EM could not be read for any frame.

`_merge_metric.csv` gets four matching per-frame columns alongside the existing `z`, `neuron`,
`chain_idx`, `own_contained`, `n_foreign`, `foreign_ids`, `empty`: `spanning_merge` (bool),
`bled_fraction` (float), `boundary_on_membrane` (float), `underfill_fraction` (float). All four are
blank for a frame the membrane pass could not score (EM unavailable, or `--no-membrane`).

Repeat `--root` to compare runs on a single node-table load.
