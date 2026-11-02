# Outline: C. elegans nerve-ring segmentation talk (v3, remodel)

Remodel triggered by Mei's feedback (more approachable, define terms as they come up) plus a
restructure request: automask moves to the front of the three methods since that is where
everyone starts, the video demos move into the methods section instead of sitting stranded near
the end, before/after drops its SAM3-favoring framing for an honest tradeoff, "where it breaks"
splits per method instead of one generic list, a new "already tried" section shows real
data/visuals for things that did not pan out (temporal projection, organelle suppression) plus the
two pretrained detectors that generalize but are not wired in yet (MitoNet, NucleoNet), and
"adaptations" is cut, it did not carry its own weight as a slide.

- Audience: the lab, fluent in biology/connectomics/segmentation, no context on this project.
- Goal: a progress update that solicits senior advice.
- Every figure centered at native aspect ratio, no wide stretching, narration carries detail.
- Jargon gets defined the moment it is introduced, not assumed (Mei's note).

## 1. Goal

1. **Title**. Subtitle: "With SAM".
2. **Manual correction is the bottleneck**. Mei's target vs. the prior mEMbrain attempt's cost.

## 2. Pipeline flow

3. **From raw EM to neuron meshes**. Data-flow diagram, inputs to output.

## 3. About SAM

4. **Three ways to use SAM, no training**. Image mode (one slice), video mode (track across
   slices), and automask (whole frame, no prompt at all). This slide introduces automask as a SAM
   capability; the next section covers it as a segmentation *strategy*.

## 4. Methods overview, automask first

5. **Three ways to segment a whole neuron**, automask first since it is where everyone starts.
   (1) Automask: whole frame, no identity. (2) Per-slice: re-prompt every slice from its own node.
   (3) Propagation: seed once, track through the stack. SAM2/SAM3 is a model swap across all three.
6. **Automask detail**.
7. **Per-slice detail**, plus its dense-map video demo moved in from the old current-work section.
8. **Propagation detail**, plus its dense-map video demo (same move).

## 5. Measurement

9. **A ground-truth-free ruler**. The merge metric.
10. **Measurement, part 2**. The membrane ridge filter.

## 6. Results

11. **Which method wins**, refocused specifically on propagation vs. per-slice, SAM2 vs. SAM3 (the
    2x2 that matters). The many-parameter config sweeps move to the backup 4-way table.
12. **Before and after, honestly**. Rewritten from a SAM2-fails/SAM3-succeeds framing to a real
    tradeoff: SAM3 is more conservative, a mild edge either way depending on what you are
    optimizing for, not a clean win.

## 7. Where it still breaks, per method

13. **Common error modes**, split by method instead of one generic list: propagation's merge
    issues, per-slice's jankiness and mitochondria-limited masks, and nucleus-limited masks, which
    hits both.

## 8. Things already tried

14. **Grow-to-membrane fill**. Promoted from backup: real underfill improvement, real leak cost.
15. **Temporal/intensity projection** (negative result). New chart from the real sweep numbers
    (window=0/1/2 vs. foreign-node bleed), the averaging-out-organelles idea did not help.
16. **Organelle suppression: classical, then pretrained** (negative-to-flat result, real detectors
    exist but nothing consumes them yet). Classical shape/intensity detector moved the real bleed
    floor by zero; a real detector (MitoNet + NucleoNet, both spot-checked and generalizing, both
    with a real precision-good/recall-incomplete verdict) moved it by at most one cell. Points at
    the membrane map's own resolution as the real bottleneck, not organelle contamination of it.

## 9. Future directions

17. **Updated, still three umbrella buckets**, refreshed against what actually landed since the
    last version of this deck (propagation second-pass shipped; temporal projection and organelle
    suppression both closed out negative; MitoNet/NucleoNet exist but unwired), fewer bullets per
    bucket, no re-listing what already happened as if it were still open.

## 10. Close

18. **Thanks / questions**.

## Backups (on demand)

- The 4-way SAM2/SAM3 table with exact numbers, now also the home for the many-parameter config
  sweeps pulled out of the main results slide.
- Cross-worm ground-truth registration (a sanity check, not the ruler).
- Dense segmentation and resolving overlaps (ridge filter + automask + dense map on one frame).

## Open items to confirm while executing

- Whether "adaptations" content (branch splitting, dual crop resolution, negative prompts, QC) is
  genuinely dead or needs a one-line mention folded into the pipeline-flow or methods slides so the
  audience is not confused later by an unexplained term. Currently: cut per direct instruction, no
  fold-in planned unless a slide review finds a dangling reference.
