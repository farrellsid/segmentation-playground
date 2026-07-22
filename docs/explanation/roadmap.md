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

7. **Nested-membrane mis-segmentation (the point-prompt ceiling).** For double-bordered structures
   like somas (a large nucleus inside the cell), a positive point landing inside the nucleus segments
   the *nucleus*, not the neuron. This is inherent to point-prompting on nested membranes, so even with
   perfect prompt placement the current system has an accuracy ceiling on these objects. *(new, 2026-07)*

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

> **Update (July 2026): two deep-research passes, adversarially verified.** Since the June writeup I
> ran two literature surveys with claim-level verification, one on error detection + benchmark design,
> one on improving the segmentation itself. Both confirmed the broad shape below and sharpened it in a
> few load-bearing ways, folded into the subsections and the staged plan:
> - **Forward/backward propagation consistency is the best-attested training-free error detector**
>   (RoboEM cut tracing errors from ~22-24% to 1-4% by keeping only where the two passes agree). We
>   already propagate bidirectionally, so this is the cheapest QC upgrade available (§4.2, Stage 1).
> - **Any SAM2 finetune must be neurite-targeted.** An EM finetune trained on *organelles* measurably
>   *degrades* neurite segmentation (micro_sam, Archit et al. 2024); a neurite specialist improves it.
>   This is now a hard constraint on Stage 2, not a footnote (§4.7).
> - **The dense/affinity hedge is not training-free.** FGNet only beats stock SAM2 after EM finetuning
>   and drops SAM2 propagation entirely (SAM2 as a feature encoder only); Spatial-SAM's learned memory
>   gives modest gains on blobs only. So the dense path is an evidence-gated last resort, not a co-equal
>   path (§4.3, §4.7, Stage 3).
> - **A target-worm annotation benchmark (new §4.2b) is the real gate on trustworthy method comparison.**
>   The cross-worm GT scores generalization; comparing methods in-distribution needs a benchmark built by
>   judging the pipeline's own masks on the target worm.
>
> Overriding caveat, reinforced by both passes: **no verified source evaluates on thin C. elegans
> neurites**, so every quoted number is an optimistic ceiling. The clearest proof is within a single
> method, SAM4EM scores 92.4% Dice on blobby mitochondria but 53.8% on complex synapses.

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

Replace the hand-tuned geometric thresholds with signals the field has shown beat them, ordered here
by how well the evidence transfers to us (verified July 2026):

- **Forward/backward propagation consistency (training-free, do first).** Propagate +z and −z from the
  anchor and flag frames where the two disagree. This is the strongest published evidence for a
  training-free merge/drift detector: RoboEM keeps only locations where forward and backward tracings
  agree, cutting tracing errors from ~22-24% to 1-4% (Schmidt/Boergens et al., Nature Methods 2024,
  s41592-024-02226-5). We already run both passes, so scoring per-frame +z vs −z mask disagreement is
  near-zero cost and hits our dominant merge and abrupt-jump modes directly.
- **Feed the EM image content into QC (the biggest structural gap).** Our QC reads nothing from the raw
  EM today. The durable connectomics detector design feeds the **EM patch + candidate mask over a large
  context region** and predicts a split/merge error map, not a mask scored in isolation, and this recurs
  across two independent primaries: Guided Proofreading (Haehn et al., CVPR 2018, arXiv 1704.00848) and
  the Zung-Lee error-detection net (NeurIPS 2017, arXiv 1708.02599). Even before a learned model,
  boundary-vs-membrane agreement is the feature class to start computing.
- **No-reference quality estimation for the target worm (no labels there).** Reverse Classification
  Accuracy (Valindria et al., IEEE TMI 2017, arXiv 1702.03407) predicts per-object Dice with no GT and
  reliably flags failed segmentations; **In-Context RCA** (arXiv 2503.04522, 2025) does the same using
  SAM2 + DINOv2 retrieval at ~0.4-0.7 s/image, reusing the SAM2 we already run, and we already hold the
  reference database it needs (the cross-worm dense GT). **EvanySeg** (arXiv 2409.14874, 2024) is the
  concrete architecture template for a learned per-frame quality head, a ViT scoring an image+mask crop.
  Two hard caveats: RCA's coverage guarantee relies on calibration/test exchangeability and **degrades
  under domain shift** (a different worm is exactly that), and EvanySeg's correlation fell from ~0.75 to
  0.506 on an out-of-distribution model, so any estimator calibrated on the GT worm must be re-calibrated
  once target-worm labels exist. All numbers are medical, none EM neurites.
- **SAM2's own predicted-IoU + occlusion scores** as a confidence signal (already populated as
  `pred_iou`), calibrated enough to drive SAM2Long's memory-tree search, so usable to flag frames.
- **Connectome-prior checks, my strongest and cheapest lever, and one neither FFN nor MemBrain had.**
  C. elegans is a stereotyped ~300-neuron connectome with known identities and neighbours. A mask that
  ends mid-neuropil, bridges two known-distinct cells, or has the wrong branch count violates a strong
  biological prior. The connectomics-standard cheap version is **topology debugging statistics**: a
  self-loop/autapse is a probable merge, an orphan fragment a probable error, both flaggable with no
  reference labels (Plaza & Funke, Frontiers Neural Circuits 2018). Far more principled than area ratios.
- **Learned split/merge classifier**, trained on real errors vs ground truth, once we have enough
  labels. The recommended low-data route is to **recycle our accumulating human corrections as labels**
  rather than run a fresh campaign (Autoproof, arXiv 2509.26585; ConnectomeBench mines proofreader edit
  histories, arXiv 2511.05542). Honest caveat: both recycle *large* existing corpora, so the "low-data"
  framing is aspirational, the transferable idea is the mechanism (log corrections as labels), not the
  reported 80%-cost / 90%-value figures (a self-reported hypothetical from an un-peer-reviewed preprint).

### 4.2b Building the target-worm benchmark, annotation protocol (R1 headline)

The ruler in §4.1 scores against the *cross-worm* dense GT, which measures generalization. To measure
error detection and method quality *in-distribution* I need a second thing: a benchmark built on the
**target** worm by having a human judge the pipeline's own predicted masks. The GUI's label store
already collects the raw material (per-frame verdict + error_type + anchor verdict); this is about
scaling that into a deliberate benchmark. The verified proofreading literature gives one sharp warning
and a reusable design:

- **Selection bias is the trap.** If a human only inspects frames QC already flagged, I can estimate
  precision but never recall, missed errors are invisible. The benchmark must sample **beyond** the
  flagged set: random/stratified frames plus a deliberately-labelled clean control set, so both
  false-positive and false-negative rates are measurable.
- **Unaided novices make segmentations worse.** In a fixed 30-min task, novice proofreaders *raised*
  edit distance on average; only a tool that guided them to candidate errors helped (Haehn et al.,
  IEEE VIS/TVCG 2014). So surface candidate errors to the annotator, use expert adjudication, and do not
  hand someone raw masks to judge cold. This is also third-party support for the auto-QC value prop.
- **Granularity.** Per-frame correct/wrong verdicts are the most expensive but train a per-frame detector
  directly; per-chain triage (judge the chain, then localize the first/worst wrong frame or the bad
  z-interval) is cheaper and still supports ERL. Start with per-chain-then-localize and reserve dense
  per-frame labelling for a sampled subset.
- **Reusable eval template:** between-subjects, brief training, fixed time window, scored against expert
  GT with VI / Rand / edit distance (Haehn 2014). Adapt the counts to our sparse per-frame setting.
- **Two open problems the literature does not solve for us:** (1) **anchor contamination**, a frame wrong
  only because the seed/box was wrong is not a propagation error and needs its own label (the label
  store's `anchor_passed` verdict is the hook, but the attribution scheme is unproven); (2) the
  quantitative **mapping from a frame-level detector score to neuron-level ERL/VOI**, so a high detector
  score provably predicts reconstruction quality and human-time saved. Both are ours to settle empirically.

### 4.3 Branching / merging, consensus, agglomeration, memory-tree, linking (Problem 3 · R4)

The single-arm failure is the field's universal failure mode; there are four complementary attacks:

- **SAM2Long (training-free, drop-in).** Replaces SAM2's greedy memory selection with a constrained
  memory-tree search over multiple pathways scored by cumulative predicted-IoU, which mitigates
  error accumulation and recovers after "occlusion/reappearance", exactly what branch points and
  section artifacts look like. Reported +3.0 J&F average (up to +5.3) on long-video benchmarks.
  **Adopt before any custom work.** Source: arXiv 2410.16268 (ICCV 2025); github.com/Mark12Ding/SAM2Long.
  Two sibling training-free memory drop-ins surfaced in the July research and are worth trialing the
  same way: **MA-SAM2** (occlusion-resilient, mask-quality-based memory selection; MICCAI 2025,
  arXiv 2507.09577) and **RevSAM2** (built for 3D volumes, replaces SAM2's FIFO memory queue with a
  reverse-propagation query selection that keeps only high-quality masks as prompts across the whole
  stack; arXiv 2409.04298). RevSAM2 is the closest to our existing node-anchored multimask + re-seeding
  idea. Caveat: all three are validated on natural / surgical / 3D-medical video, not EM neurites, so the
  1-5 point gains are an optimistic ceiling.
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

  **Bottleneck (honest, now verified):** everything past tier 1 needs a good **membrane/affinity
  signal**, too coarse at scale-8; at full res it likely means a *trained* affinity predictor
  (LSD/FGNet-style), which re-introduces the domain-gap/training cost SAM2-propagation was chosen to
  avoid. A classical EM-gradient watershed may suffice for tiers 1-2; tier 3 realistically implies an
  affinity model. The July research confirmed the cost is real: **FGNet is not a training-free drop-in**,
  it matches SOTA only with SAM2 weights frozen and beats it (+12.5% VOI on AC3/AC4) only after EM
  finetuning, and its architecture uses SAM2 as a mere feature encoder feeding a *trained* dual-affinity
  decoder + watershed, dropping SAM2 propagation entirely (arXiv 2511.13063). **Spatial-SAM** (CVPR 2026)
  replaces SAM2's memory with a learned SDF memory precomputed by a trained 3D U-Net, gains only ~1-3
  Dice and only on blob-like organelles (mito/nuclei). Both reinforce that this whole tier is a learned,
  evidence-gated last resort, not a drop-in.

### 4.4 Thin structures + resolution, topology loss, skeleton crops, refinement (Problem 4 · R7)

- **clDice (soft centerline-Dice) topology-preserving loss**, proven to preserve connectivity up to
  homotopy for tubular structures; the single most relevant loss for thin neurites, to stop
  fragmentation. Train any mask/affinity head with it. Source: Shit et al., CVPR 2021 (arXiv 2003.07311).
  The July research confirms it as the best-supported lever for the fragmentation/split mode, with two
  honest limits: it is a *training* loss (relevant only once §4.7 finetuning is on the table, not a
  drop-in), and it fixes **connectivity, not cross-section under/over-fill** (it is insensitive to
  boundary shifts within the tube radius and biased toward larger diameters, which motivates cbDice
  variants). Pair it with soft-Dice; do not expect it to fix bleed.
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
  with my own seed-ablation finding that `box_pos` beat `pos_only`). The July research makes this a
  verified, zero-cost lever against the dominant merge/bleed mode: a single positive point is
  under-constrained and lets SAM2 bleed into neighbours, while a box and/or negative points on
  neighbouring objects sharply constrain the decoder's search space (Sanaat et al., SPIE Medical
  Imaging 2025, arXiv 2408.04762). This validates the existing box-seed + nearest-neighbour-negative
  design; the actionable check is to confirm no path ever falls back to a point-only seed.

### 4.6 Post-processing, topology-aware, and mesh-space not morphological (Problem 6 · R7)

- Stop morphologically smoothing thin masks (the suspected harm). Move connectivity preservation
  into **training** via clDice (§4.4), and do geometric cleanup in **mesh space** at export
  (quadric decimation + Taubin smoothing, §4.8), not as pixel morphology. Largest-CC is actively
  dangerous near merges. First datum is still the cheap on/off A/B I already planned.

### 4.7 The base-paradigm hedge, finetune SAM2, and build the dense+linking path (R5/R6)

Two threads, now that I have GT:

- **Finetune SAM2 on the confirmed ground truth, neurite-targeted.** Mask-decoder-only first (frozen
  prompt encoder), add LoRA on the image encoder if needed, PEFT updates <5% of params, fits a single
  consumer GPU. Supervise **only on confirmed voxels**. The July research turned this from a plausible
  plan into a recipe with one hard rule and two simplifications:
  - **Hard rule: the finetune must be neurite-targeted, never organelle-borrowed.** micro_sam's EM
    *generalist* (trained on organelles) reliably improves only roundish structures (mito, nuclei) and
    *degrades* CREMI neurites, "because it was trained to segment organelles rather than membrane
    compartments like neurites"; a task *specialist* finetune does improve neurites across all settings
    (Archit et al., Nature Methods 2024, s41592-024-02580-4). So do not initialize from or borrow an
    organelle model; train on our own neurite labels. This is also the clearest published "when
    finetuning fails to beat stock" signal, and the domain-gap pivot for Stage 2.
  - **Keep the loss simple.** The claim that a BCE+SoftDice+FocalTversky composite is essential was
    *refuted* in the research (0-3). Start with Dice + BCE (the lightweight SAM2 Colab recipe), add
    soft-clDice only for the connectivity/fragmentation mode (§4.4). Likewise, the claim that
    full-model / image-encoder finetuning beats decoder-only was *refuted* (1-2), so **decoder-first is
    defensible**, not a compromise.
  - **Feasibility and data efficiency are settled.** SAM4EM does decoder LoRA in ~4GB VRAM plus a 3D
    memory-attention over serial slices, directly our propagation regime (arXiv 2504.21544, CVPR 2025
    workshop); SAM2LoRA tunes <5% of params (arXiv 2510.10288); microscopy evidence says most of the
    gain arrives with only ~2-5% of training data or as few as ~10 images (micro_sam; lightweight SAM2
    Colab, bioRxiv 2025.11.08.687405). Optional: initialize from CEM500K EM-pretrained features (eLife
    2021, articles/65894). Caveat: all accuracy numbers are organelle/fundus/LM, none thin neurites, so
    treat feasibility as transferable and magnitudes as not.
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

## 5. Staged plan (2026-07-15 redesign: measurement-first, evidence-gated)

> **Redesign note.** This replaces the earlier stage numbering. Its landed detail (the built ruler,
> the per-section registration upgrade, the Stage-0.2 cross-worm GT smoke numbers) now lives in the
> CHANGELOG; the roadmap stays forward-looking. One spine orders everything: *you cannot fairly choose
> a lever until you can measure one*. Two facts from the July research + review round force it. (1) The
> QC flags are blind to bleed. (2) The cross-worm GT is both **eroded** (confirmed in our VAST copy:
> neighbouring masks are inset from the shared membrane by design, an unpublished/incomplete
> segmentation) **and a different animal**, so every boundary metric against it is doubly biased. So the
> plan leads with a ruler we can trust on the *target* worm, then gates each segmentation change on it.
> §4 (solutions by problem) is the reference these phases point into.

A seventh problem joins the §2 list this round: **nested-membrane mis-segmentation** (the point-prompt
ceiling). For double-bordered structures like somas, a positive point inside the nucleus segments the
nucleus, not the neuron; no prompt placement fully removes it. It motivates the Phase 1 prompt fixes and,
ultimately, the Phase 3 model upgrades.

**Phase 0, fix the ruler (now, ~zero cost, no new labelling).**
The cross-worm eval harness already exists (`eval/`: region IoU, VOI, ERL, per-section registration).
What it cannot do is grade quality on the *target* worm. The unlock is a **target-worm skeleton
merge-metric**, scored against our own CATMAID skeletons, GT-free:

- Count **foreign skeleton nodes contained in each mask**, on the **raw** (pre-non-overlap) masks: a mask
  covering another neuron's node is an unambiguous, independently-measured bleed. Add a mask-dropout
  (omission) count. This is a **severe-merge floor**: high precision, low recall, it misses mild bleed
  that stops short of a neighbour's centreline (Phase 2 covers that).
- **Scope it honestly, this is not an ERL benchmark for us.** Because every mask is seeded from its own
  skeleton, the split/ERL side is partly circular (we were handed the topology); the non-circular, useful
  signal is the foreign-node merge rate. Do not report ERL as if from a from-scratch segmenter.
- **Retro-score every run we already have** (fullres, wholeimg_s4, tier2forced, neg, s1neg) on the merge
  metric. This finally grades the negatives + resolution rounds that the blind flags could not.
- Keep the cross-worm eval as a **secondary topology check** (VOI_merge / ERL), demoted from headline;
  fold in the metric-robustness fixes it still needs (ERL merge tolerance, per-section affine edge
  residual).

*Gate:* how bad is severe bleed, and did negatives / full-res actually reduce it. *(§4.1, §4.2)*

**Phase 1, cheap structural fixes (measured by Phase 0).**

- **Re-seed every slice from the skeleton node (per-slice image-mode, no video propagation).** Each
  slice is its own anchor, re-grounded on its own node (nodes are ~one-per-slice, so no locate pass),
  segmented in a node-centred crop, so memory can never carry the wrong cell across slices. Keeps the
  crop resolution win (including crop_scale 1). Directly targets the drift bleed the merge-metric shows
  is dominant. Spec: `docs/superpowers/specs/2026-07-15-phase1-per-slice-reseed-generous-multimask.md`.
  *(§4.5, §4.3)*
- **Generous-capped multimask for the nested-membrane ceiling.** Among SAM2's candidates containing the
  node, prefer the larger one so a soma mask includes the nucleus and reaches the outer membrane, but
  hard-reject any candidate above the max-area cap (SAM2's largest is often the whole frame), with
  resolution-aware leeway. No negative point in the nucleus, that would exclude it. Same spec as above.
  *(§4.5)*
- **Close-out levers, landed and measured.** Two more fixes on top of the two above, both gated off by
  default: a per-slice blow-up guard that caps the gross per-slice tail (median-area-factor cap,
  nearest-accepted-neighbour fallback, guarded frames flagged for review), and a generous-first,
  negatives-in-crop bundle that sizes the tier-2 crop from a generous first pass, then turns negatives on
  once inside the crop. Shipped as three CCDB presets, `original_perslice_only_guard`,
  `original_perslice_guard`, `original_genfirst_negcrop`. Spec:
  `docs/superpowers/specs/2026-07-17-phase1-blowup-guard-and-genfirst-negcrop-design.md`. *(§4.3, §4.5)*

*Gate (resolved 2026-07-21, Phase 1 closed):* the CCDB A/B settled it. The blow-up guard cut per-slice's
gross tail 73% (`total_foreign` 17,481 to 4,776 for `perslice_only`) with dropout still near 0, and
`perslice_only + guard` (no generous) beats the `tier2_s1forced_neg` baseline on foreign-frame-rate
(0.109 vs 0.321), dropout, and mild-bleed. `genfirst_negcrop` tied the baseline at about 2.5x the
compute (rejected), and generous still adds bleed (rejected). Decision: **per-slice re-seeding plus the
blow-up guard, without generous, is the leading candidate.** Its residual underfill (0.616 vs the
baseline's 0.483) is the honest cost of tight masks and points to Phase-2 item 2c (grow-to-membrane) as
the next lever. Full numbers in the CHANGELOG (2026-07-20 entry).

**Phase 2, the per-frame membrane / boundary map (supervisor's near-term request; the lab's own method).**

- **2a + 2b, the foundation, LANDED.** A v1 classical dark-ridge membrane map
  (`sam2_utils/membrane.py`, `membrane_map`) plus three GT-free detector primitives
  (`spanning_membrane`, `boundary_on_membrane`, `underfill_fraction`), wired into
  `eval.merge_metric` as a membrane-aware pass alongside the Phase-0 foreign-node floor. The
  border-to-border spanning criterion catches mild bleed, a mask boundary that crosses a real
  membrane into a neighbour without reaching that neighbour's node, and makes the nested-membrane
  soma case fall out for free (a nucleus is a closed interior loop, not a spanning ridge). Headline
  summary field: `mild_bleed_rate`. See [ADR 0016](../adr/0016-membrane-map-border-to-border-bleed-detection.md)
  and [cli.md](../reference/cli.md). The signature is a swappable interface: a trained model (the
  Mulcahy/Witvliet skeleton-to-membrane expansion, or a small U-Net) can drop in behind
  `membrane_map` later without touching the detectors or the scorer.
- **2c, grow-to-membrane refinement of masks, queued as its own spec.** Reuses the same membrane
  signal and the `underfill_fraction` flood that 2b only measures, this time applying it to grow a
  mask to its bounding membrane. Also the route to de-bias the eroded cross-worm GT into a rough
  boundary ruler.
- **2d, principled non-overlap resolve, delivered early in prototype form (2026-07-21).** With the
  membrane signal in hand, replace the composite's argmax / first-writer-wins with a
  membrane-respecting resolver. The per-frame segmentation experiment
  (`docs/superpowers/specs/2026-07-20-perframe-segmentation-design.md`,
  [perframe-experiments.md](perframe-experiments.md)) built and compared two: a membrane-aware
  nearest-node argmax and a seeded watershed on the membrane map, both pure functions over masks,
  node coordinates, and the membrane map (`sam2_utils/perframe.py`'s
  `resolve_overlaps_argmax`/`resolve_overlaps_watershed`). This is scoped to that per-frame
  experiment for now, not yet wired into the main per-chain composite's aggregation step, so the
  original Phase 1/2 sequencing (needing the boundary signal to beat the current first-writer-wins;
  a raw EM gradient alone would be a weak version of itself) still applies to that wiring. *(§4.3)*
- **An early, lightweight R5 dense-path probe rides along with the above.** The same per-frame
  experiment's Approach 2 segments a whole EM frame at once with `SAM2AutomaticMaskGenerator`
  (`run_perframe.py --approach amg`), then matches nodes to masks and keeps the rest as unlabelled
  competitors that still push back during overlap resolution. That is a first, SAM2-only look at
  what the §4.3 tier-3 dense/hybrid hedge is reaching for (segment the frame densely, then assign
  identity), well short of a trained affinity model, but a cheap first read on whether frame-dense
  segmentation is even usable on this data before committing to the heavier trained path. Early
  finding, not yet a verdict: default AMG parameters are compute-heavy enough that a single
  target-worm frame did not finish in a short local wall-clock budget, so judging this probe
  properly is a CCDB job, tracked alongside 2d above. *(§4.3, §4.7 R5)*

The landed foundation also helps disambiguate outer-vs-inner border for the nested-membrane ceiling.
**Ask the supervisor whether a reusable membrane model or training data survives from the prior
pipeline**; if so, a future trained-model swap collapses to reuse + calibration. Note that the prior
pipeline's ~1,120 person-hours were **dense proofreading**, not building the map (a trained map is a
small U-Net, days to train), so v1's classical filter is the pragmatic starting point either way.
*(§4.2, §4.3 tier 2)*

*Gate:* membrane-aware detect (landed) + refine (2c, queued) cuts mild bleed / underfill.

**Phase 3, boundary benchmark + model upgrades.**

- **Small boundary-accurate target-worm benchmark:** a few hundred cross-sections traced *to the membrane*
  (not the eroded convention), sampled to include thin and junction cases. Calibrates the detector and
  seeds finetuning. Days-to-weeks, not the person-years of a dense volume. *(§4.2b)*
- **Finetune the SAM2 mask decoder** (decoder-first, Dice+BCE, neurite-targeted, never organelle-borrowed)
  and/or an **FGNet-style fine-grained / affinity boundary head** on frozen SAM2 features, targeting the
  domain gap, thin neurites, and the nested-membrane ceiling. *(§4.7, §4.4)*

*Gate:* validated boundary improvement on the benchmark.

**Phase 4, paradigm-decision gate.**
Only if, after 0-3, the merge metric / boundary quality still misses the bar: switch the hard regime to a
dense native-3D method (FFN, or affinity + LSD + mutex watershed), with our skeletons as seeds and
evaluation rather than prompts. This is the §4.7 R5 hedge, now explicitly evidence-gated on the
Phase-0/3 numbers, and the point at which the SAM2-as-core directive is renegotiated with evidence.
Otherwise stay SAM2-augmented. *(§4.3, §4.7, §6)*

---

## 5b. Immediate queue (July 2026)

Mapped to the phases above. DONE / READY / TODO.

1. **Verify the GT erosion** (Phase 0). DONE: confirmed, neighbouring masks are inset from the shared
   membrane by design in our VAST copy.
2. **Resolution + negatives review** (old Stage 1). DONE: cropping is the measurable resolution win,
   whole-image scale is ~irrelevant (1024 resize); negatives and the full-res second pass are
   flag-neutral, pending the Phase-0 metric for a real verdict.
3. **Build the target-worm merge-metric and retro-score all runs** (Phase 0). TODO, the next action:
   foreign-node containment + dropout on raw masks vs CATMAID skeletons.
4. **Re-seed per slice + z-extent-limited propagation** (Phase 1). TODO.
5. **Mutex-watershed / multicut non-overlap resolve** (Phase 1). TODO.
6. **Membrane-probability map + membrane-aware bleed detection** (Phase 2 foundation, 2a + 2b).
   DONE: `sam2_utils/membrane.py` + `eval.merge_metric`'s membrane pass, `mild_bleed_rate`
   headline. Non-overlap arbitration (2d) DONE in prototype form via the per-frame segmentation
   experiment (2026-07-21): two membrane-aware resolvers, argmax and watershed, not yet wired into
   the main per-chain composite. That same experiment's AMG approach is an early R5-lite dense-path
   probe, judged too compute-heavy to finish locally, so its real read is a CCDB step. Still open:
   the supervisor conversation about reusing the prior lab model, grow-to-membrane refinement (2c),
   and wiring 2d into the main composite's aggregation step.
7. **Boundary-accurate benchmark + mask-decoder finetune / FGNet head** (Phase 3). TODO.
8. **SAM3 whole-set cluster comparison** (parallel track, not a numbered phase). The 2-chain
   SAM3-vs-SAM2 bake-off (2026-07-21) found SAM 3 per-slice leads on foreign-node bleed and
   dropout; its own next-step note was a broader run before productionizing. The plumbing for
   that now exists: a `--backend sam3` switch on `batch.py`, `cluster/run_array.sh` wired to
   forward it, and a Narval runbook (`docs/how-to/run-sam3-on-narval.md`); see the CHANGELOG's
   same-day entry for the detail. TODO, still queued: running SAM3 per-slice and propagation
   over the whole target-worm set on Narval and scoring both with `eval.merge_metric` against
   the existing SAM2 baselines, to confirm the 2-chain win holds at scale. Human-executed,
   Duo-MFA blocks a headless submission.

`bigimg` (SAM2 `image_size` 2048) stays retired: it crashes off-distribution and its output would be
unvalidated; the resolution goal is served by cropping / tiling.

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
- **The target-worm merge-metric (Phase 0) shows bleed is mild** -> de-prioritize the architecture
  changes; the negatives / full-res levers may already be close enough, and effort shifts to the
  boundary benchmark and finetune.
- **A reusable membrane model survives from the prior lab pipeline** -> Phase 2 collapses to reuse +
  calibration and can jump ahead of Phase 1.
- **The eroded / different-worm confound proves dominant** (Phase 0) -> treat all past boundary numbers
  (including the ~2-3% precision) as unreliable, and rebuild the ruler on target-worm skeletons + the
  Phase-3 benchmark before trusting any boundary A/B.

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
  absolute performance. Both July research passes were blunt here: **no verified source evaluates on
  thin C. elegans neurites at low resolution**, so every quoted figure is an optimistic ceiling. The
  proof is within a single method, SAM4EM scores 92.4% Dice on blobby mitochondria and 53.8% on complex
  synapses, so morphology alone can halve the number.
- **Organelle transfer degrades neurites.** An EM finetune trained on organelles is *detrimental* for
  neurites (micro_sam), and a quality estimator trained on one segmenter/domain drops sharply
  off-distribution (EvanySeg ~0.75 → 0.506). Anything trained or calibrated on the cross-worm GT or on
  organelle data must be validated, and likely re-calibrated, on target-worm neurites before it is trusted.
- **Licensing.** SAM2 and micro_sam are permissive; nnInteractive checkpoints are non-commercial
  (CC-BY-NC-SA 4.0), verify before any non-research use.
- **Linkers don't model arbitrary merges.** Trackastra/Seg2Link model divisions; ultrack forbids
  merges by design, genuine merge topology still needs agglomeration or a human.
- **Operating ahead of the literature.** As of June 2026 no published SAM2 pipeline targets C. elegans
  neurite instance segmentation specifically; FGNet (mouse/fly EM) is the closest. Budget for
  iteration, the survey is a map, not a recipe.
- **The skeleton merge-metric is a severe-bleed floor, not a full ruler.** It only catches bleed that
  reaches a foreign centreline, and its split/ERL side is partly circular for a skeleton-seeded pipeline,
  so it is not comparable to from-scratch ERL numbers. Mild bleed and boundary quality still need the
  membrane map (Phase 2) and the boundary benchmark (Phase 3).
- **The GT erosion is confirmed for our copy, not the published method.** Our VAST segmentation is
  unpublished/incomplete and its masks are inset from the membrane by design; the published C. elegans
  filling expands *to* the membrane. Boundary metrics against our cross-worm GT are therefore biased, and
  the different-worm mismatch compounds it.
- **The point-prompt paradigm has a hard ceiling on nested membranes** (soma + nucleus): 100% accuracy is
  unreachable by prompt tuning alone, part of the case for the Phase-3 model upgrades.

---

## 8. References

**Project prior art (the starting point)**
- Bader Lab human-liver vEM pipeline + `sam2maskpropagator`, Xing et al., bioRxiv 2026.04.22.719970.
- C. elegans CATMAID/VAST vEM context, Frontiers Neural Circuits 2018, fncir.2018.00094.

**SAM2 / Segment Anything for EM & microscopy**
- FGNet (SAM2 → 3D EM neurons, dual affinity; *not* training-free, beats stock only after EM finetune), arXiv 2511.13063 (AAAI 2026).
- Spatial-SAM (learned SDF memory for 3D EM, trained 3D U-Net; blob organelles only), CVPR 2026.
- micro_sam, Nature Methods 2024, s41592-024-02580-4; github.com/computational-cell-analytics/micro-sam (+ peft-sam). *Key result: organelle generalist degrades neurites; neurite specialist improves them.*
- **Lightweight SAM2 microscopy finetuning (Colab), bioRxiv 2025.11.08.687405.**
- SAM4EM (decoder-LoRA ~4GB VRAM + 3D memory attention), arXiv 2504.21544 (CVPR 2025 workshop); github.com/Uzshah/SAM4EM.
- SAM2LoRA (<5% params), arXiv 2510.10288. SAM-EM (full SAM2 finetune, particles), arXiv 2501.03153.
- SAM2 3D medical / propagation, arXiv 2408.02635; SegmentWithSAM arXiv 2408.15224; SLM-SAM 2 arXiv 2505.01854; center-outward study arXiv 2507.23272.

**Training-free VOS / SAM2 memory strategies**
- SAM2Long (memory-tree search), arXiv 2410.16268 (ICCV 2025); github.com/Mark12Ding/SAM2Long.
- MA-SAM2 (occlusion-resilient memory), arXiv 2507.09577 (MICCAI 2025); github.com/Fawke108/MA-SAM2.
- RevSAM2 (3D-volume reverse-propagation memory), arXiv 2409.04298.
- Prompting discipline (box/negatives constrain vs point bleed), Sanaat et al., SPIE Medical Imaging 2025, arXiv 2408.04762.

**Connectomics reconstruction**
- FFN, Januszewski et al., Nature Methods 2018, s41592-018-0049-4; github.com/google/ffn.
- LSD, Sheridan et al., Nature Methods 2023, s41592-022-01711-z; github.com/funkelab/lsd.
- Mutex Watershed, Wolf et al., ECCV 2018 (978-3-030-01225-0_34); GASP, arXiv 1906.11713.
- RoboEM, Schmidt/Boergens et al., Nature Methods 2024, s41592-024-02226-5.

**Video object segmentation**
- SAM2Long, arXiv 2410.16268 (ICCV 2025); github.com/Mark12Ding/SAM2Long.
- Cutie, arXiv 2310.12982 (CVPR 2024); XMem, arXiv 2207.07115 (ECCV 2022).

**Error detection / proofreading / quality**
- Guided Proofreading, Haehn et al., CVPR 2018, arXiv 1704.00848 (image+mask+context input recipe). Error Detection & Correction, Zung/Lee et al., NeurIPS 2017, arXiv 1708.02599.
- Interactive proofreading tools + eval template (novices worsen segmentations; guided-to-errors helps), Haehn et al., IEEE VIS/TVCG 2014.
- Split/merge cost asymmetry + topology debugging stats (self-loops, orphans), Plaza & Funke, Frontiers Neural Circuits 2018, fncir.2018.00102.
- Reverse Classification Accuracy (GT-free Dice prediction), Valindria et al., IEEE TMI 2017, arXiv 1702.03407. In-Context RCA (SAM2 + DINOv2), arXiv 2503.04522.
- EvanySeg (GT-free per-object quality regressor; architecture template), arXiv 2409.14874.
- Autoproof (recycle corrections as labels), arXiv 2509.26585; ConnectomeBench (mine edit histories), arXiv 2511.05542.

**Metrics**
- ERL / merge-split, FFN (above); II-CATS bERL/nERL, Zhai et al., PMLR v227 2024. VOI, Meilă; Nunez-Iglesias et al. SNEMI3D adapted-Rand.

**Topology losses / geometry-guided**
- clDice, Shit et al., CVPR 2021, arXiv 2003.07311. MemBrain (geometry-normalized crops, regression targets for sparse labels), Lamm et al., CMPB 2022, 106990.

**Domain adaptation / pretraining / linking**
- CEM500K, eLife 2021, articles/65894. MitoEM 2.0, bioRxiv 2025.11.12.687478. VEM transfer, Oxford Bioinformatics Advances 2025, vbaf021.
- Seg2Link, Sci Rep 2023, s41598-023-34232-6; `pip seg2link`. Trackastra, arXiv 2405.15700 (ECCV 2024); github.com/weigertlab/trackastra. ultrack, Nature Methods 2025, s41592-025-02778-0. nnInteractive, arXiv 2503.08373; github.com/MIC-DKFZ/nnInteractive.

**Meshing**
- skimage `marching_cubes` (`spacing` kwarg); zmesh, github.com/seung-lab/zmesh; igneous / cloud-volume, github.com/seung-lab/igneous.
