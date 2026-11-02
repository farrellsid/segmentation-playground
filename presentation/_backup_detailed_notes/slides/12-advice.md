# Future directions

<div class="mt-1 mx-auto max-w-[62rem] text-left text-[1.12rem] leading-relaxed">

**Segmentation**
- Tune SAM3's config: A/B sweeps on negative prompts
- Autofill underfilled masks by growing to the membrane
- Fix the failure modes: nucleus capture, thin neurites, branches
- Finetune SAM on our own EM, or explore alternative methods and models

**Measurement and QC**
- Build a hand-corrected ground-truth set
- Stronger error detection and prediction

**Correction**
- Hybrid re-anchoring: re-segment a flagged frame from its best neighbour
- Streamline the human review step

</div>

<!--
Opening: "Where this goes next, in three buckets."

Beats: on segmentation, tune SAM3's config (it is more conservative, so sweep the negative prompts), keep chipping at the failure modes, and finetune on our own EM, with a trained dense method held in reserve if SAM plateaus. On measurement, build a properly hand-corrected ground-truth set and a stronger error detector so QC is not the weakest link. On correction, hybrid re-anchoring (actually re-segment a flagged frame, not just flag it) and generally make the human review faster. Open the floor here.

Timing: 2 to 3 minutes
-->
