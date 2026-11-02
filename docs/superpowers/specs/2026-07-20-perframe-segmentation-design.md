# Per-frame neuron segmentation: design

Status: design, approved-in-principle 2026-07-20 (supervisor-requested direction). Scope: a
per-frame segmentation experiment with two approaches, membrane-aware overlap resolution, a
metric-guided multimask selector, and an AMG parameter tuner, all judged on a per-frame scoring
metric and by eye, with every iteration's results kept for documentation.

## Why and framing

So far the pipeline segments one neuron chain at a time and propagates through z. The supervisor
asked for the complementary view: segment everything present in a single EM frame at once, and use
the overlaps between cells to resolve spills. We build two approaches and compare them. This is
deliberately Phase-2's item 2d (non-overlap arbitration) and the roadmap's R5 dense-path hedge
arriving early because there is a concrete use for them, so the primitives built here feed back into
those. The membrane map from the Phase-2 foundation is what makes the ambitious approach viable.

Honest guardrail, stated once and enforced throughout: the scoring metric rewards node-containment
and low foreign-bleed, not true boundary fidelity, so it can be gamed by masks that simply swallow
their node. Every experiment is co-judged visually, and the write-up says so.

## Shared foundation (both approaches use these)

**F1, per-frame node index.** For a `catmaid_z`, gather every node at that z across all neurons
(real + virtual), each as `(x_tif, y_tif, cell_name, node_id)`. Extends `merge_metric.nodes_by_z`
to one-z-all-cells. Pure, torch-free.

**F2, per-frame scoring metric.** Score a frame's labelled instance map by reusing existing
primitives: own-node coverage (`1 - dropout`), foreign-node bleed (severe merge), the membrane
detectors (`spanning_membrane`, `boundary_on_membrane`, `underfill_fraction` from
`sam2_utils/membrane.py`), and a new pre-resolution overlap scalar (summed pairwise mask
intersection over frame area, i.e. how much cells fight for pixels). Emits per-frame scalars plus a
per-cell breakdown. Pure, torch-free, so it doubles as the tuner's objective and the multimask
selector's objective. The composite objective for tuning/selection: require own-node containment and
zero foreign nodes, then maximise `boundary_on_membrane`, then minimise `spanning_merge` /
overlap.

**F3, membrane-aware overlap resolution (arbitration).** Given candidate masks (labelled targets and
optional unlabelled competitors), assign one instance label per contested pixel. We build **two
methods and compare them** (this comparison is itself a deliverable):
- `argmax`: a membrane-respecting nearest-node rule. For each contested pixel, prefer the label whose
  own node is nearest and whose claim does not cross a membrane ridge to reach the pixel; break ties
  by `boundary_on_membrane` support.
- `watershed`: seeded watershed / mutex-watershed on the membrane map, nodes as attractive seeds,
  inter-cell edges repulsive.
Both are pure array functions over (masks, membrane map, node coords), testable on synthetic input.

## Approach 1, prompt-based per-frame

For each node in the frame, run `image_predict` (positive point + box). Collect all cells' masks
(auto-labelled node to neuron), resolve overlaps (F3), score (F2). Two swept knobs, documented:
- **Negatives {on, off}**, the generous-vs-conservative axis. On: the other cells' nodes in the same
  frame become negatives (a stronger, more principled negative source than the chain-wise neighbours),
  producing tighter masks. Off: generous masks that lean on F3 arbitration to cut overlaps at the true
  membrane. Hypothesis to test: generous + arbitration beats conservative negatives.
- **Multimask selection {pred_iou, generous, metric-guided}**. `pred_iou` = SAM2's own score;
  `generous` = largest gate-passing candidate (existing `multimask_generous`); `metric-guided` = the
  candidate best satisfying F2's composite (contains own node, no foreign node, best
  boundary-on-membrane, least spanning). The metric-guided mode is the "use the new metrics to get
  better masks" idea and generalises the generous-vs-conservative policy into a per-mask choice.

## Approach 2, auto-mask + match + keep

Run `SAM2AutomaticMaskGenerator` on the frame to segment all structures. **Match** each node to its
mask; **keep** the unmatched masks as unlabelled competitors so neighbours push back on bleed.
Resolve overlaps (F3, targets vs competitors), score the labelled subset (F2). Two swept knobs:
- **AMG params** (tuned, see below).
- **Match/rank {area, metric-guided}**: `area` = smallest AMG mask containing the node (avoids
  grabbing a whole-frame blob); `metric-guided` = the containing mask with the best
  `boundary_on_membrane`, tie-break smallest area. The membrane map filters/ranks the AMG soup.

## Metric-guided multimask selector (shared primitive)

A pure function `select_by_metric(candidates, node_xy, foreign_nodes, membrane_map) -> index` that
implements F2's composite as a candidate ranker. Used by Approach 1's `metric-guided` selection and
Approach 2's `metric-guided` matching. This is where the Phase-2 detectors become an active mask
chooser rather than only a post-hoc grader.

## AMG parameter tuner (on Approach 2)

A grid/random search over the AMG knobs (`points_per_side`, `pred_iou_thresh`,
`stability_score_thresh`/`_offset`, `box_nms_thresh`, `min_mask_region_area`, `crop_n_layers`,
`use_m2m`) on the frame sample, scored by F2's composite. Keeps and logs the best set, and dumps the
best set's montages for eyeballing. Grid or random first; Optuna is a later option. The report states
the objective can be gamed and shows the visual check.

## Frame set and iteration

A representative sample of **5 to 10 frames**, chosen to span many-cell regions, thin neurites, and a
soma, configurable and extensible to all node-bearing z. **Downscaled for local iteration** (the
existing `scale` machinery; `_sam` scale 8 locally, full-res on CCDB). Every run keeps its results
(next section), so iterations are comparable across the sweep.

## Documentation and results (a first-class requirement)

Every experiment run writes to a results directory `results/perframe/<run_name>/`:
- `config.json` (approach, all knob settings, frame set, git commit),
- `scores.csv` (F2 per-frame and per-cell),
- `montages/<z>.png` (EM, labelled instance map, membrane overlay).
The bulk (montages) is gitignored; a committed `docs/explanation/perframe-experiments.md` log
summarises each run (config + headline scores + a pointer to its montages), appended per iteration,
and the best montages are promoted into `docs/figures/perframe/` for the report. Useful comparisons
get jotted down there as we go, per the documentation ask.

## File layout (library/driver split)

- Pure primitives, library: F1 (per-frame node index), F3 (both overlap resolvers), the
  metric-guided selector, and the AMG-to-node matcher go in `sam2_utils/` near `membrane.py`; F2
  (per-frame scoring) extends the `eval/` merge-metric family. All torch-free.
- SAM2-touching driver: a new `run_perframe.py` (sibling of `batch.py` / `run_aval.py`) owning the
  prompt-mode runner, the AMG-mode runner, the tuner, and the montage dump. The library never imports
  it (enforced by `tests/test_import_direction.py`).

## Testing

- Pure units (F1 index, F2 scoring, F3 both resolvers, the metric selector, the matcher): CPU,
  torch-free, synthetic arrays.
- SAM2-touching runners: a tiny CPU smoke on a downscaled frame with `--model-size tiny`, plus GPU on
  CCDB for the real sweeps.

## Risks (stated up front)

- AMG is param-sensitive and can fragment on EM neuropil; the tuner + membrane ranking are the
  mitigations, and Approach 1 is the fallback if AMG proves unusable.
- Matching (Approach 2) is fuzzy; the membrane-support tie-break mitigates it, and the metric-guided
  match is compared against the plain area rule.
- The metric is incomplete; visual co-judging is mandatory and the write-up says so.
- Per-frame AMG over many frames is compute-heavy; the tuner runs on the 5-to-10-frame sample first,
  full runs go to CCDB.
- Scope is large (two approaches, two resolvers, three selection modes, a tuner). The plan sequences
  it foundation-first so each piece is independently testable and useful before the next.
