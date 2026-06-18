# Future directions, a research-informed roadmap

The forward-looking companion to [`design-notes.md`](design-notes.md) (the lean
current-state reference) and [`CHANGELOG.md`](../CHANGELOG.md) (the build log).
This file is the expansion of PIPELINE_CONTEXT **§9 (research directions)**: it records the
*reasoning and the sources* behind each proposed direction, so the lean doc can stay lean. Nothing
here is committed yet, it's a proposal, written after a deliberate step back.

> **Reading order.** PIPELINE_CONTEXT tells you what's true and what's queued *now*; PIPELINE_HISTORY
> tells you how we got here; this file tells you where we might go next and *why we believe it*, with
> citations. When a proposal lands, fold a one-line decision into PIPELINE_CONTEXT and move the
> detail here or to history, don't let any single file grow append-only again (the §9.4 lesson).

---

## 0. Where this is coming from

So far I've been following my instincts and trying to get the best results I could from that. I
picked a promptable foundation model (SAM2), adapted a prior-art pipeline from a neighbouring
problem, and built outward from it, adding QC, batching, a review GUI, and a stack of
accuracy levers, tuning each by feel against whatever signal I had. That got me a working
semi-automatic pipeline, which is genuinely the point: at one human-approved anchor per chain it's
~50× faster than hand-segmenting slice by slice.

But two things made me stop and take stock. First, I now have **ground-truth segmentation for
another worm**, a different EM stack with a somewhat different look, *with the matching EM images
and explicit markers for which segments are manually confirmed*. For the whole project so far I've
had no real ruler: every A/B was scored against my own flag rule, which I already knew was
unreliable. Ground truth changes that. Second, re-reading the prior art and doing a broad literature
survey made it clear that the parts I'd "established as best" were mostly instinct, and that the
field has worked out a lot of this already.

So this document is the honest version: here's what I built and why, here are its pros, cons, and
the problems I need to fix, and here, after stepping back and reading, are proposed solutions and
ideas, grounded in sources rather than vibes (the §4 *measure-don't-trust, don't-ship-on-vibes*
principle, made concrete).

---

## 1. What I built, and why (the instinct phase)

The pipeline is SAM2 video propagation. For each neuron chain: take a CATMAID skeleton node as an
anchor on its mid-slice, prompt SAM2 *image mode* (point → mask → largest connected component →
bounding box), then seed SAM2 *video mode* with that box and propagate the mask **bidirectionally**
through z into a per-neuron mask volume, for export to Blender.

This was a reasonable instinct, and I'd make the same call again to get started:

- **My professor directed the project to use SAM2**, so SAM2 is fixed as the core.
- There was **direct prior art in the same lab ecosystem**: the Bader Lab human-liver vEM pipeline
  and its `sam2maskpropagator` code (Xing et al., bioRxiv 2026.04.22.719970), which does almost
  exactly this, CATMAID/SWC point prompt → image predict → largest-CC → bbox → bidirectional
  propagate → instance masks → Blender. Same CATMAID, same Blender endpoint.
- A promptable foundation model is **fast to stand up with no training**, and the expensive thing
  I'm automating is the ~300-slice propagation, not the one-time anchor.

Everything I added beyond that prior art, inline QC + flagging, the batch runner with
manifest/resume/triage, the napari review GUI, the anchor crop tiers, multimask/seed knobs, was the
response to a problem that is **harder and ~20× bigger** than the liver case (see §2).

---

## 2. Honest assessment: pros, cons, problems

**What the SAM2-propagation approach gets right (pros).**
It's promptable and needs no training to start; it reuses my CATMAID skeletons directly as prompts;
the video memory propagates one object across ~300 slices cheaply; and the whole thing already
clears the bar that matters, it's much faster than hand-segmentation with a human kept in the loop.

**Where it's structurally limited (cons).**
The literature is blunt about this, and it lines up with what I see. SAM2's encoder **downsamples
away the fine detail** that thin neurites depend on, this is the explicit motivation for FGNet's
fine-grained refinement branch (Li et al., arXiv 2511.13063). Its video memory **assumes smooth
motion and appearance**, which serial sections violate at branch points and across imaging changes,
so multiple SAM2-for-medical-volumes studies find naïve SAM2 *underperforms* finetuned models. And
it carries **natural-image priors**, not EM-adapted ones. None of this is fatal, it's why finetuning
and the fixes below exist, but it means stock SAM2 propagation is a starting point, not a ceiling.

**The concrete problems I need to fix** (each tagged with the PIPELINE_CONTEXT backlog/research anchor
it maps to):

1. **Error detection / QC is the weakest part.** Hand-tuned geometric thresholds (area ratio,
   temporal IoU, skeleton containment, predicted IoU), scored post-hoc. *(§8 theme C; R1)*
2. **My evaluation metric is unreliable.** Flag-rate is what every A/B has leaned on, and it's the
   thing being questioned, I've been optimizing against a noisy yardstick. *(R2)*
3. **Branching/merging: the mask covers only one arm.** SAM2's memory biases it to keep tracking the
   arm it already saw. *(§8 item 19; R4)*
4. **Thin structures at low effective resolution.** A neurite is only a few px wide at scale-8.
   *(§8 item 23 / crop tiers; R7)*
5. **Anchor/prompt quality and propagation drift.** *(§8 items 17, 21; R3)*
6. **Post-processing suspected to hurt thin neurites.** Morphological open/close is a blunt
   instrument here. *(§8 item 18; R7)*

And the bigger strategic question sitting underneath all of them: **is SAM2 video propagation the
right base paradigm at all**, or should the hard regime be handled by a trained, EM-specialized model
(the route the liver paper took with nnU-Net for *its* hard half)? *(R5/R6)*

---

## 3. The step back, what changed

Three inputs reframed the project:

- **I obtained cross-worm ground truth + EM + confirmation markers.** This is the pivotal unlock.
  Almost everything in the M4.5 backlog was "label-gated"; now I have labels. It enables real
  **evaluation** (score the current pipeline against confirmed segments) and **finetuning**
  (a training target). Caveat I have to respect: it's a *different* worm with a different look, so it
  measures **generalization**, not in-distribution accuracy, treat it as a domain-adaptation
  benchmark and spot-check on the target worm.
- **Re-reading the prior art was clarifying.** The liver paper uses SAM2 only for the *easy*,
  compact, sparsely-annotatable structures and reaches for **nnU-Net** (trained, dense) for the hard
  numerous ones. FFN (Januszewski et al., Nature Methods 2018) and MemBrain (Lamm et al., CMPB 2022,
  106990) independently solve their hard regimes with **trained, task-specialized models** and use
  the **model's own confidence/consistency as the error signal**, not hand-tuned geometric QC.
  Three independent votes against exactly the part I flagged as "established as best but might not be."
- **A broad SOTA survey** turned those signals into concrete, SAM2-compatible, single-GPU-feasible
  methods, organized below by the problem each one solves.

---

## 4. Proposed solutions, by problem

Each item: the project problem it addresses, what the literature offers, the proposed action, and
sources. Caveats are kept inline and honest, many quoted numbers are first-party benchmarks on
natural video / organelles / medical volumes, **not** C. elegans neurites, which are harder, so
expect lower absolute numbers.

### 4.1 Fix the ruler first, ERL + split/merge VOI (Problem 2 · R2)

> **Status (June 2026): the ruler is built** (`eval/`). All three connectomics metrics below are
> implemented and wired into the GT scorer (`eval.score_batch` + `eval.score_labelmap`): per-neuron
> **ERL** + split/merge (`eval.erl`), and **VOI_split/VOI_merge + ARAND** via `eval.metrics.voi_arand`,
> which defaults to **scikit-image's reference implementations**, the same `adapted_rand_error` /
> `variation_of_information` (with `ignore_labels=(0,)`, `voi=split+merge`) that the CAD/FGNet papers
> use (pure-numpy fallback when skimage is absent). See the Stage 0.2 results in §5.
>
> **Applicability caveat (June 2026, important).** VOI/ARAND are built for **dense, whole-volume**
> instance segmentation (CAD/FGNet score a fully-agglomerated labelmap). Our pipeline is **sparse,
> per-neuron prompted propagation**, so for the current setup they are **secondary**, not the headline:
> (a) restricting to the scored neurons' GT-foreground makes VOI_merge **blind to bleed into *unscored*
> neighbours** (the costliest real merge, it reads as background/false-positive area, not a merge);
> (b) the composite gives each neuron one id, so VOI_split/merge largely **restate region recall/
> precision**; (c) the numbers are **not comparable** to CAD/FGNet's dense benchmarks (similar values
> are coincidence of the easier sparse setting). **The appropriate primary ruler for the sparse pipeline
> is per-neuron region IoU/precision/recall + ERL** (skeleton-based, 3D, per-neuron, naturally fits a
> subset). VOI/ARAND become genuinely apt only with a **dense** labelmap (the §4.3 refinement / R5 path).

This is the prerequisite for trusting everything else. Adopt the connectomics-standard metrics:

- **Expected Run Length (ERL)**, expected error-free traced length from a random point in a random
  neuron, with any segment containing a *merge* assigned zero length. It's skeleton-based, and I
  already have CATMAID skeletons, so it's essentially free to compute. Source: Januszewski et al.,
  Nature Methods 2018 (s41592-018-0049-4); biological-constraint variants bERL/nERL in II-CATS (Zhai
  et al., PMLR 2024).
- **Variation of Information decomposed into VOI_split + VOI_merge**, lets me weight mergers more
  heavily than splits (mergers are far costlier to fix by hand). Source: Meilă; Nunez-Iglesias et al.
  Also the SNEMI3D adapted-Rand F-score (split/merge components).
- **Set a merge:split cost ratio** (start 5:1 or higher). The hemibrain FFN base segmentation
  illustrates the trade-off concretely: ~163 µm ERL at a 0.25% false-merge rate vs 585 µm ERL at
  27.6% false-merge after aggressive agglomeration.

*Action:* build an eval harness that produces per-neuron ERL and a split/merge breakdown on the
**confirmed** segments of the new worm; deprecate flag-rate as an A/B metric.

### 4.2 Learned error detection, model-confidence, and connectome priors (Problem 1 · R1)

Replace the hand-tuned geometric thresholds with signals the field has shown beat them:

- **Learned split/merge classifier** trained on real errors vs ground truth, exactly what my
  confirmation-marked GT enables. Sources: Guided Proofreading (Haehn et al., arXiv 1704.00848);
  Error Detection & Correction (Zung/Lee et al., arXiv 1708.02599); recent learned proofreading
  (Autoproof, arXiv 2509.26585; point-affinity merge transformers, Troidl et al. 2025).
- **SAM2's own predicted-IoU + occlusion scores** as a confidence signal, calibrated enough to
  drive SAM2Long's memory-tree search, so good enough to flag frames. (I already populate `pred_iou`.)
- **Forward/backward propagation-consistency** (RoboEM-style): propagate +z and −z from the anchor;
  disagreement is a principled merge/drift detector. Cheap and SAM2-compatible. Source: RoboEM,
  Schmidt/Boergens et al., Nature Methods 2024 (s41592-024-02226-5).
- **Quality estimation without ground truth** for the *target* worm where I have no labels:
  In-Context Reverse Classification Accuracy (arXiv 2503.04522); Dice-regression nets (EvanySeg-style).
- **Connectome-prior checks**, my strongest, cheapest lever, and one neither FFN nor MemBrain had:
  C. elegans is a stereotyped ~300-neuron connectome with known identities and neighbours. A mask
  that ends mid-neuropil, bridges two known-distinct cells, or has the wrong branch count violates a
  strong biological prior. FFN's own discussion proposes exactly this (topology-violation as an
  efficient proofreading guide). Far more principled than area ratios.

### 4.3 Branching / merging, consensus, agglomeration, memory-tree, linking (Problem 3 · R4)

The single-arm failure is the field's universal failure mode; there are four complementary attacks:

- **SAM2Long (training-free, drop-in).** Replaces SAM2's greedy memory selection with a constrained
  memory-tree search over multiple pathways scored by cumulative predicted-IoU, which mitigates
  error accumulation and recovers after "occlusion/reappearance", exactly what branch points and
  section artifacts look like. Reported +3.0 J&F average (up to +5.3) on long-video benchmarks.
  **Adopt before any custom work.** Source: arXiv 2410.16268 (ICCV 2025); github.com/Mark12Ding/SAM2Long.
- **Multi-seed / over-segmentation consensus** (the FFN idea): propagate from multiple anchors /
  both directions / multiple resolutions, keep only what's consistent, accept extra splits to kill
  merges. Source: FFN, Nature Methods 2018.
- **Over-segment → watershed → agglomeration as a parallel path.** Affinities (+ Local Shape
  Descriptors) → watershed → mutex-watershed/mean-affinity agglomeration. LSD is "two orders of
  magnitude more efficient" than FFN while competitive (Sheridan et al., Nature Methods 2023,
  s41592-022-01711-z; github.com/funkelab/lsd); Mutex Watershed (Wolf et al., ECCV 2018); GASP
  (arXiv 1906.11713). FGNet shows the SAM2-native version: SAM2 encoder → dual affinity maps →
  watershed+agglomeration (arXiv 2511.13063).
- **Segment-per-slice then link across z** for divisions/branches: Seg2Link overlap-linking,
  purpose-built for EM neuropil (Sci Rep 2023, s41598-023-34232-6; `pip seg2link`), or Trackastra, a
  transformer linker that handles divisions natively at ~1 FPS for 2k objects/frame on one GPU (ECCV
  2024, arXiv 2405.15700; github.com/weigertlab/trackastra). *Caveat:* these model divisions, **not**
  arbitrary merges; genuine merge topology still needs agglomeration or a human.

- **Post-propagation refinement of the composited map (session idea, June 2026; slots here + §M5).**
  Once propagation + the per-neuron composite produce a (sparsely-)dense labelmap, refine it with
  connectomics-style techniques. **Key distinction:** our post-propagation errors are **overlaps +
  bleed**, not an over-segmentation, so the right primitive is **arbitration / region competition**,
  NOT agglomeration *per se* (agglomeration merges an over-segmentation; it does not resolve overlaps, watershed output has none). Three tiers of ambition:
  1. **Overlap arbitration (cheap, ~now).** Replace the composite's `first-writer-wins` with a
     principled owner for each contested pixel, nearest skeleton, or higher `pred_iou`. This *is* the
     open M5 chain-merge-conflict question (§8 item 29: union vs voting), so it pays off at aggregation
     anyway.
  2. **Seeded watershed / region competition (mid).** Skeletons (or masks) as seeds competing for
     contested/gap pixels over the EM membrane gradient, resolves overlaps, snaps edges to membranes,
     fills small gaps. The connectomics-flavoured version of "refine the dense map".
  3. **Hybrid, dense oversegmentation + agglomerate-to-anchors (big; == R5).** Run a dense affinity →
     watershed oversegmentation, then use the propagated masks as **anchors** and agglomerate fragments
     onto whichever anchor they belong to. SAM2 supplies *semantic identity*, the oversegmentation
     supplies *membrane-accurate fragments*, the assignment fixes **bleed and overlap together**. This
     is the R5 dense+agglomeration hedge, expressed as a SAM2-anchored pipeline.

  **Bottleneck (honest):** everything past tier 1 needs a good **membrane/affinity signal**, too coarse
  at scale-8; at full res it likely means a *trained* affinity predictor (LSD/FGNet-style), which
  re-introduces the domain-gap/training cost SAM2-propagation was chosen to avoid. A classical
  EM-gradient watershed may suffice for tiers 1-2; tier 3 realistically implies an affinity model.

### 4.4 Thin structures + resolution, topology loss, skeleton crops, refinement (Problem 4 · R7)

- **clDice (soft centerline-Dice) topology-preserving loss**, proven to preserve connectivity up to
  homotopy for tubular structures; the single most relevant loss for thin neurites, to stop
  fragmentation. Train any mask/affinity head with it. Source: Shit et al., CVPR 2021 (arXiv 2003.07311).
- **Skeleton-oriented / centerline-following crops**, use the CATMAID skeleton to keep the neurite
  centered and at higher effective resolution, the principled version of my tier-2/tier-3 crop idea.
  This is also MemBrain's core trick (normalize the input using known geometry to simplify the task;
  Lamm et al. 2022) and RoboEM's neurite-aligned flight crops (Nature Methods 2024).
- **A fine-grained refinement branch** to recover the detail SAM2's downsampling loses (FGNet's
  Feature-Guided Attention + Fine-Grained Encoder; arXiv 2511.13063).

### 4.5 Anchor/prompt quality + drift (Problem 5 · R3, §8 items 17/21)

- **Center-outward propagation** from the most informative skeleton slice was found most accurate vs
  top-down/bottom-up in a SAM2 volume study (arXiv 2507.23272), matches my current mid-frame anchor
  instinct and worth confirming.
- **Re-seed at CATMAID skeleton nodes** along the chain to bound drift.
- **Prefer box/mask prompts over a single point** where the cross-section is ambiguous (consistent
  with my own seed-ablation finding that `box_pos` beat `pos_only`).

### 4.6 Post-processing, topology-aware, and mesh-space not morphological (Problem 6 · R7)

- Stop morphologically smoothing thin masks (the suspected harm). Move connectivity preservation
  into **training** via clDice (§4.4), and do geometric cleanup in **mesh space** at export
  (quadric decimation + Taubin smoothing, §4.8), not as pixel morphology. Largest-CC is actively
  dangerous near merges. First datum is still the cheap on/off A/B I already planned.

### 4.7 The base-paradigm hedge, finetune SAM2, and build the dense+linking path (R5/R6)

Two threads, now that I have GT:

- **Finetune SAM2 on the confirmed ground truth.** Mask-decoder-only first (frozen prompt encoder),
  add LoRA on the image encoder if needed, PEFT updates <5% of params, fits a single consumer GPU,
  and is the explicit low-data recommendation. Supervise **only on confirmed voxels**; use a
  composite loss (BCE + soft-Dice + soft-clDice). Sources/tooling: micro_sam (Nature Methods 2024,
  s41592-024-02580-4; github.com/computational-cell-analytics/micro-sam, + peft-sam); lightweight
  SAM2 microscopy finetuning in a single Colab (bioRxiv 2025.11.08.687405); SAM2LoRA (arXiv
  2510.10288); FGNet (arXiv 2511.13063). Optional: initialize from CEM500K EM-pretrained features
  (eLife 2021, articles/65894).
- **Build the dense-segmentation + cross-z-linking path in parallel** (§4.3) as the architectural
  hedge: if the cross-worm domain gap turns out large and SAM2 finetuning doesn't beat stock SAM2,
  this path is less sensitive to SAM2's natural-image priors. This is the R5 re-architecture, no
  longer parked, it's the fallback, not a someday-maybe.

### 4.8 Meshing / anisotropy for Blender (R8 / M5)

- Mesh with the anisotropy baked in: `zmesh.Mesher((16,16,50))` (connectomics-native: marching cubes
  + quadric simplification, `max_error` in physical nm; github.com/seung-lab/zmesh) or
  `skimage.measure.marching_cubes(vol, level=0, spacing=(50,16,16))` so vertices come out in nm.
- Decimate with quadric edge collapse (small `max_error`); smooth with **Taubin** (volume-preserving)
  not Laplacian (shrinks thin tubes); export PLY/OBJ to Blender. No morphological pre-smoothing.

---

## 5. Proposed staged plan

Ordered so each stage's output feeds the next, and so the ruler (Stage 0) exists before any tuning.
Thresholds are advance/pivot gates, in the §4 ruler spirit.

- **Stage 0, Instrument & benchmark the current pipeline (now).** ERL + VOI_split/VOI_merge on the
  confirmed cross-worm segments; set the merge:split cost ratio; benchmark the *current* pipeline.
  *Advance when:* I can produce a per-neuron ERL and split/merge breakdown **from the real pipeline,
  through a verified registration**. *(§4.1)*

  The ruler is built (region + VOI + ERL; `eval/`). A first degenerate run, `eval/predict_gt.py`,
  small model, points-only seed, produced the first numbers and, more usefully, exposed *how* Stage 0
  has to finish: (a) the measurement is gated on the **skel→GT coordinate transform**, which both
  places prompts and samples node labels, so a loose transform poisons every number (the dry run:
  ~50% of slices zero-overlap with correctly-sized-but-*displaced* masks; self-consistency ERL = 0
  *with the perfect GT as input* because 47% of nodes sample off their own segment); and (b)
  `predict_gt.py` is a **scaffold, not the benchmark**, a partial reimplementation with v1 shortcuts
  (points-only seed, no postprocess, union-across-chains) that bled badly; the honest number must come
  from the real `batch.py`. So Stage 0 completes in four sub-steps:

  - **0.1, Verify the coordinate transform (keystone, first).** *Model upgrade landed.* The
    `eval/diag_registration.py` structural check showed the residual was **structured**, not noise (a
    per-section affine cut the median centroid residual 19.6 px → 5.1 px), so `registration.py` was
    upgraded from *global linear + per-section translation* to a **per-section affine** (full 2×3 per
    slice, robust fit, z-interpolated/smoothed). Re-fit result: median residual 19.6 → **4.7 px**,
    on-mask **67.9% → 85.7%**. Provenance (right worm = project 280) confirmed four ways, so the earlier
    ~50% miss was an under-powered alignment *model*, not a bad import. *Remaining:* the interactive GUI
    overlay (human gut-check) and confirming self-consistency ERL recovers from the earlier ~0 µm. Done
    at 4× scale, the affine *model* transfers to the full-res re-fit (0.3), only the constants change.
  - **0.2, Eval the real batch pipeline against GT, ✅ BUILT, first numbers in (June 2026).** The
    production `run_chain`/`batch.py` now runs on SEM-Dauer 1 (not the `predict_gt` reimplementation)
    via a worm-agnostic **`pipeline.FrameStore` seam** (default `TifFrameStore` keeps the target worm
    byte-identical; `eval.gt_dataset.GtFrameStore` reads the per-slice PNG EM) plus a skel→image
    transform baked into `annotate_df.x_tif/y_tif` from the **per-section registration**. Driver:
    `batch.py --preset eval` with a configurable subset (`--neurons` / `--neuron-limit N` /
    `--all`, guarded against an accidental 9766-chain run). Scored by `eval.score_batch`
    (`BatchPredictionSource` unions chains + upscales `_sam`→GT grid), which logs live progress,
    `eval_timing.csv`, and a `measurement_log.jsonl` provenance record (what/against-what/when/metrics/
    results/timing). **Labelmap metrics wired** (`eval.score_labelmap`): composite per-slice `_sam`
    labelmaps (neuron→id, first-writer-wins; tier-2 `_pcrop` placed via `crop_window`) → **VOI_split/merge
    + ARAND** (over GT-foreground) and **per-neuron ERL** (registration node-sampling ÷save_downscale to
    `_sam`, with neighborhood sampling). Supersedes `predict_gt.py` as the scored path.

    **Results (3-neuron smoke PVPR/VA4/AS3, a FLOOR, not the final gate):** small single-pass `_sam` →
    micro-IoU 0.022, VOI 0.875, ARAND 0.162; **large + tier-2-default** → micro-IoU 0.024, VOI 0.847,
    ARAND 0.161, ERL ~1%. (1) **Large model alone barely helps** (slightly hurts the `_sam` neurons), the cross-worm **domain gap dominates, not capacity**; (2) **tier-2 helped where it engaged**, VA4
    *kept* tier-2 (`_pcrop`) and ~doubled IoU (0.012→0.022, precision up), but **2/3 chains fell back**
    at the `chain_crop_min_image_score=0.70` floor (crop anchors ≈0.69; a target-worm default,
    mis-calibrated for cross-worm → lower to ~0.6); (3) **merge/bleed-dominated** (VOI_merge ≫ split;
    precision ~2.5%). ⚠️ VOI/ARAND ≈ FGNet's Table-4 range is **coincidence**, ours is over 3 *sparse*
    neurons' GT-foreground (far easier than FGNet's dense volume); region IoU (0.024) + ERL (~1%) are the
    honest read. **Next:** a multi-chain neuron (per-chain fallback + aggregation) + a lowered tier-2 floor.
  - **0.3, Full-res GT export (parallel, manual), ✅ DONE (June 2026).** The VAST masks + EM are
    re-exported at native resolution: `full_scale/` (9728×9216, 851 slices) sits next to
    `one_fourth_scale/` on F:, verified == the metadata's full-res VAST coord grid. `config` now points
    at it (`GT_DOWNSCALE = 1`). This unblocks faithful batch eval (tier-2 skeleton crops need full res)
    and required a full-res registration (A ≈ I, not 0.25·I), **done** by scaling the ¼ fit ×4
    (`py -3 -m eval.scale_registration`; geometrically identical to a from-scratch full-res re-fit but
    instant vs ~1.5 h of HDD decodes, validated mean A ≈ I, on-mask 91.7% spot check). `registration.json`
    is now full-res; the ¼ fit is kept as `registration_quarter_scale.json`.
  - **0.4, ERL merge tolerance (metric robustness).** Stop ERL zeroing a whole neuron for a single
    stray node (a tolerance / majority rule). Without it the skeleton metric stays at 0 even for
    near-perfect segmentations under any registration noise.

  *Loose ends:* the empty-name `--neuron-limit` selection bug and `predict_gt`'s bleed levers are now
  low priority (the scored path is `batch.py`, which has real seeding/postprocess); fix only if
  `predict_gt` is kept as the points-only baseline.
- **Stage 1, Free wins on the existing SAM2 path (1-2 wk).** Drop in SAM2Long; center-outward
  propagation; forward/backward consistency check (replaces several hand-tuned QC thresholds);
  re-seed at skeleton nodes + skeleton-following crops. *Advance when:* ERL up and merge rate down vs
  Stage 0. *(§4.3, §4.5, §4.2)*
- **Stage 2, Finetune SAM2 (2-4 wk).** Decoder-first + optional encoder-LoRA on confirmed voxels,
  composite + soft-clDice loss. *Advance when:* finetuned beats stock SAM2 on held-out confirmed
  segments. *Pivot if not:* inspect the domain gap, lean on Stage 3. *(§4.7, §4.4)*
- **Stage 3, Fix branching structurally (3-6 wk).** Parallel per-slice over-segmentation
  (affinities/LSD or FGNet-style) → watershed → agglomeration → link across z (Seg2Link/Trackastra),
  used where propagation drops an arm. *Advance when:* branch-point recall beats single-arm
  propagation. *(§4.3, §4.7)*
- **Stage 4, Learned QC + triage (ongoing).** Train the split/merge classifier / quality regressor
  on confirmed GT; add connectome-prior checks; route only flagged frames to the human. *Advance
  when:* human review time per neuron drops while ERL holds, the stated success criterion (faster
  than manual, accuracy preserved). *(§4.2)*

---

## 6. Decision points, what would change this plan

- **Large cross-worm domain gap** (Stage 2 finetuning doesn't beat stock SAM2) → shift weight to the
  affinity + agglomeration path (Stage 3), which is less dependent on SAM2's priors.
- **VRAM is binding on the 300-slice stack** → prefer a low-memory VOS memory design (Cutie:
  ~1.35 GB vs XMem ~3.03 GB on long sequences; arXiv 2310.12982) and tile the volume.
- **Mergers stay the dominant error** → raise the merge:split cost ratio and prioritize the
  merge-detector literature (point-affinity transformers; RoboEM consistency).
- **A trained dense model clearly wins** → the SAM2-as-core directive is a constraint to renegotiate
  with my professor, with evidence (ERL numbers) rather than argument. Until then SAM2 stays central
  and the dense path is a hedge/component, not a replacement.

---

## 7. Caveats (read before believing any number above)

- **Compute asymmetry.** FFN's headline accuracy came from GPU clusters; the LSD paper estimates a
  full mouse brain would take "about 226 years" on 1,000 GPUs with FFN. I'm on one box, adopt the
  *ideas* (consensus, ERL, in-loop confidence, topology priors), and use LSD/affinity + SAM2
  finetuning as the compute-appropriate analogues, not FFN-proper.
- **Cross-specimen GT measures generalization, not in-distribution accuracy.** Confirm gains with
  spot-checks on the target worm.
- **First-party benchmark numbers.** FGNet, SAM2Long, Cutie, nnInteractive, micro_sam numbers are on
  natural-video / organelle / medical benchmarks, not thin C. elegans neurites, expect lower
  absolute performance.
- **Licensing.** SAM2 and micro_sam are permissive; nnInteractive checkpoints are non-commercial
  (CC-BY-NC-SA 4.0), verify before any non-research use.
- **Linkers don't model arbitrary merges.** Trackastra/Seg2Link model divisions; ultrack forbids
  merges by design, genuine merge topology still needs agglomeration or a human.
- **Operating ahead of the literature.** As of June 2026 no published SAM2 pipeline targets C. elegans
  neurite instance segmentation specifically; FGNet (mouse/fly EM) is the closest. Budget for
  iteration, the survey is a map, not a recipe.

---

## 8. References

**Project prior art (the starting point)**
- Bader Lab human-liver vEM pipeline + `sam2maskpropagator`, Xing et al., bioRxiv 2026.04.22.719970.
- C. elegans CATMAID/VAST vEM context, Frontiers Neural Circuits 2018, fncir.2018.00094.

**SAM2 / Segment Anything for EM & microscopy**
- FGNet (SAM2 → 3D EM neurons, dual affinity), arXiv 2511.13063 (AAAI 2026).
- micro_sam, Nature Methods 2024, s41592-024-02580-4; github.com/computational-cell-analytics/micro-sam (+ peft-sam).
- **Lightweight SAM2 microscopy finetuning (Colab), bioRxiv 2025.11.08.687405.**
- SAM2LoRA, arXiv 2510.10288. SAM-EM (full SAM2 finetune, particles), arXiv 2501.03153.
- SAM2 3D medical / propagation, arXiv 2408.02635; SegmentWithSAM arXiv 2408.15224; SLM-SAM 2 arXiv 2505.01854; center-outward study arXiv 2507.23272.

**Connectomics reconstruction**
- FFN, Januszewski et al., Nature Methods 2018, s41592-018-0049-4; github.com/google/ffn.
- LSD, Sheridan et al., Nature Methods 2023, s41592-022-01711-z; github.com/funkelab/lsd.
- Mutex Watershed, Wolf et al., ECCV 2018 (978-3-030-01225-0_34); GASP, arXiv 1906.11713.
- RoboEM, Schmidt/Boergens et al., Nature Methods 2024, s41592-024-02226-5.

**Video object segmentation**
- SAM2Long, arXiv 2410.16268 (ICCV 2025); github.com/Mark12Ding/SAM2Long.
- Cutie, arXiv 2310.12982 (CVPR 2024); XMem, arXiv 2207.07115 (ECCV 2022).

**Error detection / proofreading / quality**
- Guided Proofreading, Haehn et al., arXiv 1704.00848. Error Detection & Correction, Zung/Lee et al., arXiv 1708.02599.
- Autoproof, arXiv 2509.26585; ConnectomeBench, arXiv 2511.05542; In-Context RCA, arXiv 2503.04522.

**Metrics**
- ERL / merge-split, FFN (above); II-CATS bERL/nERL, Zhai et al., PMLR v227 2024. VOI, Meilă; Nunez-Iglesias et al. SNEMI3D adapted-Rand.

**Topology losses / geometry-guided**
- clDice, Shit et al., CVPR 2021, arXiv 2003.07311. MemBrain (geometry-normalized crops, regression targets for sparse labels), Lamm et al., CMPB 2022, 106990.

**Domain adaptation / pretraining / linking**
- CEM500K, eLife 2021, articles/65894. MitoEM 2.0, bioRxiv 2025.11.12.687478. VEM transfer, Oxford Bioinformatics Advances 2025, vbaf021.
- Seg2Link, Sci Rep 2023, s41598-023-34232-6; `pip seg2link`. Trackastra, arXiv 2405.15700 (ECCV 2024); github.com/weigertlab/trackastra. ultrack, Nature Methods 2025, s41592-025-02778-0. nnInteractive, arXiv 2503.08373; github.com/MIC-DKFZ/nnInteractive.

**Meshing**
- skimage `marching_cubes` (`spacing` kwarg); zmesh, github.com/seung-lab/zmesh; igneous / cloud-volume, github.com/seung-lab/igneous.
