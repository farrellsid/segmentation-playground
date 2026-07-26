# Lab presentation outline: C. elegans EM neuron segmentation pipeline

A design doc for a 10-15 minute lab presentation. This is the agreed outline and the figure plan,
not the slides themselves.

## Audience, goal, message

- Audience: the lab (supervisor plus peers), fluent in biology, connectomics, and segmentation, but
  with no context on this specific project.
- Goal: a progress update that also solicits senior advice. Not a "prove myself" talk.
- Core message, two beats: the approach has potential and is improvable, and this is a big problem
  where we have made real progress and here is where we stand.
- Spine: problem-driven. Neurons are a harder, roughly 20x bigger version of the liver segmentation
  problem, and the talk shows what that forced us to change, how we check whether it worked, and
  where it still breaks.
- Slide style: one big image per slide with a few short bullets. The narration carries the flow; the
  slides support it.

## Authoritative facts (these override older doc phrasings)

- Motivation numbers: a prior attempt using mEMbrain took 2 people about 5 months of manual
  correction. PI Mei Zhen's target is about 30 minutes of manual correction per neuron on average.
  Manual correction is inevitable; the goal is to minimize it. Do not use the unsourced "50x faster"
  claim. Do not use the prior liver pipeline's "~1,120 hr proofreading" figure, it is a different
  dataset (liver) whose scale we cannot state.
- Dataset scale: the full EM stack is 2,354 slices. We work on about 300 slices in the densest part
  of the nerve ring: CATMAID layers 1293 to 1628 (about 336 slices), which are tif files z1300 to
  z1635. Start layer 1293 is explicit in the code (the registration fit slice); the end layer is
  from the SAM3 mask coverage and should be confirmed.
- References: mEMbrain, Meirovitch et al., Frontiers in Neural Circuits 17 (2023), bioRxiv
  2023.04.17.537196 (it integrates with VAST, the tool the ground-truth worm was exported from).
  Liver prior art, Xing et al., bioRxiv 2026.04.22.719970. C. elegans CATMAID/VAST context,
  Frontiers Neural Circuits 2018, fncir.2018.00094.

## Slide-by-slide outline

Each slide: the lead visual, the on-screen bullets, and the spoken beat.

1. Title. Visual: a dense nerve-ring cross-section with a few colored neuron masks (dense-overlay
   output). One line on what the project is.

2. Goal and why. Visual: manual correction is the bottleneck (hand-colour every slice) versus the
   machine-proposes-human-corrects loop. Bullets: manual correction is inevitable and expensive; a
   prior mEMbrain attempt took 2 people about 5 months; Mei's target is about 30 min per neuron; the
   job is to minimize correction time. Beat: you shrink correction time on two levers, better
   segmentation or a smoother correction workflow. That framing sets up two of the closing asks.

3. Dataset. Visual: the scale funnel (PDF page 1) plus an EM slice with Lucinda's CATMAID skeleton
   overlaid. Bullets: sensory-ablated dauer; about 300 slices (layers 1293-1628) through the densest
   nerve ring, from a 2,354-slice stack; inputs are the raw EM and Lucinda's hand-traced skeletons.
   Beat: the skeleton is the route down the middle of each neuron; we need the outline on every
   slice. Mention the separate ground-truth worm (SEM-Dauer 1, from VAST) secured for scoring.

4. What SAM is and how we use it. Visual: PDF page 2 (skeleton node to yellow mask, image mode) and
   PDF page 3 (video mode propagating both ways). Bullets: promptable foundation model, no training;
   the slice stack is a video and a neuron is a drifting object; image mode (point to mask) and video
   mode (track across slices); SAM2, and a newer SAM3. Beat: the skeletons are our prompts, so we
   start with zero training.

5. Pipeline part 1, built on liver but harder. Visual: the liver-vs-worm "why diverged" figure.
   Bullets: base is the published liver EM pipeline (Xing et al.), same skeleton to prompt to
   propagate to Blender; liver cells are compact, convex, sparse, few; neurons are thin, dense, many,
   branched. Beat: same recipe, but neuron morphology breaks it in specific ways, and most of the
   work has been those fixes.

6. Pipeline part 2, what neuron morphology forced (the core slide). Visual: PDF page 4 (skeleton tree
   with per-chain seeds and propagation) plus a two-resolution crop inset. Bullets, four changes:
   split branches into linear chains (SAM tracks one object and follows one arm at a branch);
   two-resolution crops (a neurite is about 3 px wide at the scale that fits in GPU memory); negative
   prompts on touching neighbours (masks bleed into adjacent cells); automated QC and triage (the
   liver path was hand-proofread). Beat: one sentence per change, each as problem then fix.

7. Pipeline part 3, two ways to run it. Visual: side-by-side overlays, video propagation (smooth but
   drifts) versus per-slice re-anchoring (no drift, occasional balloon caught by the blow-up guard).
   Bullets: propagation is temporally smooth but memory drifts onto the wrong cell; per-slice
   re-anchors every slice from its own node, killing drift; the guard catches the rare blow-up. Beat:
   this fork ties directly to the results. (Trim candidate: fold into slide 6 if time is tight.)

8. How we measure (the ruler). Visual: the merge-metric diagram (a mask grown from its own node; a
   foreign neuron's node inside it is a merge; the own node missing is a dropout). Bullets: no ground
   truth on the target worm; a GT-free metric built from the skeletons themselves; foreign node
   inside a mask means bleed, own node missing means dropout; principle, measure, do not trust vibes.
   Beat: this caught errors the QC flags were blind to.

9. Dense segmentation and resolving overlaps (Lucinda's request). Visual: one frame shown several
   ways, raw EM, then the Sato ridge filter (membranes as walls), then a dense segmentation, then
   automask/AMG output, with a small argmax-versus-watershed inset. Bullets: the ridge filter finds
   membranes; automask/dense proposes every cell at once; where masks overlap, resolve by argmax
   (highest score wins the pixel) or watershed (grow to the ridge walls). Beat: the ridge map is what
   lets watershed split neighbours along real membranes instead of an arbitrary line. This is a
   direction we are prototyping (dense overlay with argmax built; automask tried and compute-heavy).

10. Results and where we stand. Visual: the progress-arc chart (bleed 0.45 to 0.09, coverage 0.68 to
    about 1.0) beside the 4-way SAM2/SAM3 table and a dense-overlay image. Bullets: bleed severity
    down about 5x; coverage from about two-thirds to nearly always; per-slice beats propagation; SAM3
    beats SAM2 (about 84% less bleed severity) at about 3-4x compute; whole worm, about 124 neurons.
    Beat: the journey, not a single number, and honest about SAM3's compute cost. Pair the statistics
    with a concrete before/after mask overlay on one real frame: an older mask bleeding into a
    neighbour (whose node is marked) next to the newer mask staying clean, so the audience sees the
    improvement instead of only reading it. The foreign node is exactly what the metric counts, so the
    picture and the number are the same thing.

11. Where it still breaks. Visual: three EM crops, nucleus capture (the mask is the round nucleus), a
    thin faint neurite fading, a branch point. Bullets: nucleus/soma capture; thin faint neurites at
    the working resolution; branches. Beat: the honest limitations, which set up the asks.

12. What I would like your advice on. Visual: four framed questions, minimal text. The four asks:
    keep pushing SAM versus switch to a trained dense method; which hard problem to attack first; is
    the GT-free metric trustworthy; is streamlining human correction as valuable as better raw
    segmentation. Beat: invite discussion, this is the point of the talk.

13. Backup and thanks. Backup slides for Q&A: the per-section registration result (median residual
    19.6 to 4.7 px), the coordinate-space machinery, the review GUI correction loop, the memmap 48x
    speedup, the co-propagation experiment (tried and shelved), "scale-1 is not high resolution" from
    the resolution experiments, and the cross-worm GT eval.

## Figure plan (what exists, what to build)

Reuse as-is:
- Student's PDF (`C:\Users\User\Downloads\pipeline graphics.pdf`), pages 1-4: scale funnel, CATMAID
  skeleton + image mode, video-mode propagation, skeleton-tree with seeds + propagation. Pages 5-6
  are unrelated and ignored.
- Repo figures: `docs/figures/liver-vs-worm/` (pipeline-diff, why-diverged, branch handling, triage
  loop), `docs/figures/membrane-v1/`, `docs/figures/sam3-bakeoff/` (bake-off + node overlays),
  `docs/figures/sam3-bakeoff/dense-overlay/` (dense multi-neuron viewer).

Build:
- Progress-arc chart: foreign-frame-rate and coverage across the config trajectory (data is local in
  `resolution_experiments/*_merged/_merge_metric.csv`). Use the dataviz skill; colorblind-safe.
- Merge-metric diagram: a chain mask with its own node inside and a foreign node inside (a merge).
- Dense-and-overlaps panel (slide 9): one frame as raw EM, Sato ridge map, dense segmentation,
  automask, plus an argmax-vs-watershed inset. Ridge map from `sam2_utils/membrane.py`; dense
  labelmap from `experiments/dense_overlay.py`; argmax/watershed from `sam2_utils/perframe.py`.
- Before/after bleed overlay (slide 10): a real frame where an older config's mask contains a
  foreign neighbour node (a measured merge) and the newer config's mask on the same chain and frame
  does not, both drawn with the own node (green) and the foreign node (red) marked, so the metric is
  visible. Built from the per-frame merge-metric CSVs plus the overlay tooling in
  `experiments/sam3_overlay_disk.py`.
- Optional failure-mode crops for slide 11 (nucleus capture, faint neurite, branch point).

## Open items to confirm

- Confirm the end layer of the worked range (1628).
- Confirm the mEMbrain framing with Ben (2 people, about 5 months) so the number is quoted correctly.
- Recent Narval round (Tests 2/3/5/6): optional additions if the student downloads them. Test 2 adds
  SAM3 underfill/mild-bleed; Test 3 the negatives A/B; Test 6 a compute-vs-accuracy table. None are
  blockers; the headline results already exist.

## Next step

Build the figures that need building (progress arc, merge-metric diagram, dense-and-overlaps panel)
and assemble the deck around the reused graphics.
