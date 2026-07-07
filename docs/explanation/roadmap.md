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

## 5. Proposed staged plan

Ordered so each stage's output feeds the next, and so the ruler (Stage 0) exists before any tuning.
Thresholds are advance/pivot gates, in the §4 ruler spirit.

> **Re-evaluation (July 2026): error detection + the target-worm benchmark is the gate, not the tail.**
> The two research passes make the sequencing dependency explicit: I cannot fairly compare any Stage 1-3
> segmentation change until I can *measure* it, which needs both a trustworthy detector and an
> in-distribution benchmark (§4.2, §4.2b). So the cheap training-free pieces of Stage 4 move forward to
> run alongside Stage 1: **forward/backward propagation consistency** and **topology debugging stats**
> (self-loop = merge) cost almost nothing and replace several hand-tuned thresholds immediately, and the
> **target-worm annotation benchmark** should be scoped early because it gates the A/B evidence every
> later stage is judged on. The finetuning stage keeps the new hard constraint: neurite-targeted only.

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
    ~50% miss was an under-powered alignment *model*, not a bad import. The interactive human gut-check
    is now in place too: `eval/registration_overlay.py` is a napari viewer of the full-res VAST EM with
    raw + registered node layers (scrub z, click-to-read coordinates for a CATMAID project-280
    cross-check). Done at 4× scale, the affine *model* transfers to the full-res re-fit (0.3), only the
    constants change.
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
  - **0.5, Edge residual in the per-section affine (HIGH PRIORITY refinement).** The 0.1 human
    gut-check (`eval/registration_overlay.py`) confirmed the affine is near-perfect at frame center but
    drifts by tens of px at the edges (consistent with the fit's p90 ~17 px). An affine fits tangent to
    the true warp at the correspondence centroid and deviates with radius; the likely cause is the
    elastic per-section realignment (`*_realignment_export_*`), which no affine fully matches. It misses
    small/thin cells and is the same off-segment-node fraction capping the ERL ceiling (~11.6 vs 76.3
    µm). Gate the fix on a diagnostic: extend `diag_registration` to measure residual-vs-radius and
    trial-fit a per-section quadratic / thin-plate spline, with edge residual + rim correspondence
    density, to tell a too-simple model from edge data-starvation (interpolating models extrapolate
    badly past the correspondence hull). Then choose the model.

  *Loose ends:* `predict_gt.py` is **discontinued** (the scored path is `batch.py`, which has real
  seeding/postprocess); its empty-name `--neuron-limit` bug and bleed levers retire with it.
- **Stage 1, Free wins on the existing SAM2 path (1-2 wk).** Drop in SAM2Long (and trial the sibling
  training-free memory strategies MA-SAM2 and the 3D-volume RevSAM2, §4.3); center-outward
  propagation; forward/backward consistency check (replaces several hand-tuned QC thresholds, and is
  the cheapest verified error signal, §4.2); confirm box/mask seeds never fall back to point-only
  (§4.5); re-seed at skeleton nodes + skeleton-following crops; **node-anchored multimask selection** (pick
  SAM2's candidate that contains the positive node and, with `multimask_exclude_neg`, excludes the
  neighbour negatives; on for the `eval` preset, adapts the 2025 lightweight-SAM2 paper's
  anchor-containment idea, see [ADR 0012](../adr/0012-node-anchored-multimask-selection.md)). *Advance
  when:* ERL up and merge rate down vs Stage 0. *(§4.3, §4.5, §4.2)*
- **Stage 1b, Parameter optimization (planned, the 2025 lightweight-SAM2 paper's second idea).** Jointly
  tune the pipeline's inference knobs against the eval ruler instead of hand-setting them. Search space
  is *our* prompted-pipeline knobs (the multimask flags, `gate_*` bounds, `k_max_neg`, seed mode,
  `postproc_*`, `chain_crop_min_image_score`, `scale`/`save_downscale`), not the paper's: their notebook
  optimizes the `SAM2AutomaticMaskGenerator` (`points_per_side`, `box_nms_thresh`, `crop_n_layers`...),
  a SAM2 mode this pipeline does not use, so it is a methodology to reuse (Bayesian search like
  `skopt.gp_minimize` on Jaccard/Dice), not a notebook to port. Objective: per-neuron region IoU + ERL
  from `eval.score_batch`. *Cost to plan around:* each config is a full GT run + score (~19 min/full-res
  run today), so a search needs a fast subset or a cheaper proxy. *Advance when:* the tuned config beats
  the hand-set defaults on the held-out eval set.
- **Stage 2, Finetune SAM2, neurite-targeted (2-4 wk; always the plan).** Decoder-first + optional
  encoder-LoRA on confirmed voxels, Dice + BCE loss (add soft-clDice only for connectivity; the
  composite-is-essential and full-model-beats-decoder claims were both refuted, §4.7). **Hard
  constraint from the July research: train on our own neurite labels, never initialize from or borrow
  an organelle-trained EM model, which measurably degrades neurites.** The lightweight-SAM2 Colab is
  the concrete low-cost recipe (freeze image + prompt encoders, tune the mask decoder, AdamW, cosine
  annealing); SAM4EM shows decoder-LoRA + 3D memory-attention in ~4GB VRAM. Data efficiency is
  favorable (most gain by ~2-5% of data / ~10 images), but confirm the *minimum* label volume on our
  own morphology, the literature only documents large/organelle regimes. *Advance when:* finetuned
  beats stock SAM2 on held-out confirmed segments. *Pivot if not:* inspect the domain gap, lean on
  Stage 3. *(§4.7, §4.4)*
- **Stage 3, Fix branching structurally (3-6 wk).** Parallel per-slice over-segmentation
  (affinities/LSD or FGNet-style) → watershed → agglomeration → link across z (Seg2Link/Trackastra),
  used where propagation drops an arm. *Advance when:* branch-point recall beats single-arm
  propagation. *(§4.3, §4.7)*
- **Stage 4, Error detection + triage (start the cheap parts NOW, in parallel with Stage 1).** This
  moved up: it gates trustworthy A/B for every other stage (see the §5 re-evaluation note). Land in
  order of evidence strength: (a) **forward/backward propagation consistency** and **topology debugging
  stats** (self-loop = merge, orphan = error), both training-free and cheap; (b) **EM-content features**
  into QC (image + mask + context), the biggest structural gap; (c) **no-reference quality estimation**
  (In-Context RCA reusing our SAM2, with the cross-worm GT as its reference set) for the target worm,
  re-calibrated once target labels exist; (d) a **learned split/merge classifier / quality regressor**
  (EvanySeg-style head) trained by recycling human corrections as labels. Build the **target-worm
  annotation benchmark** (§4.2b) alongside, so the detector has both a training set and an honest
  precision/recall+recall eval that is not blinded by selection bias. *Advance when:* human review time
  per neuron drops while ERL holds, the stated success criterion (faster than manual, accuracy
  preserved). *(§4.2, §4.2b)*

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
