# Future directions

<div class="mt-1 mx-auto max-w-[62rem] text-left text-[1.05rem] leading-snug">

**Segmentation**
- The membrane map's own resolution is the real lever: two fixes aimed at its organelle noise
  both closed negative, both point at the map itself. Higher-res, learned, or finetune SAM directly
- Keep chipping at the per-method failure modes

**Measurement and QC**
- Build a hand-corrected ground-truth set
- MitoNet and NucleoNet both work, no consumer yet, wire one into a real QC signal

**Correction**
- Propagation's re-anchoring is validated at full scope (bleed -36%, dropout -81%): close the
  accept-gate gap, it only checks the frame's own node, not the specific flag that triggered it
- Streamline the human review step

</div>

<!--
Opening: where this goes next, in three buckets, updated against what actually landed since the
last version of this deck

- segmentation: the negative results from the last three slides all point the same direction, the
  membrane map's own resolution, not what is confusing it, higher-res or a learned map, or
  finetune SAM directly; keep chipping at the per-method failure modes from two slides back
- measurement: still need a real hand-corrected ground-truth set; the two detectors that already
  work (MitoNet, NucleoNet) are sitting there unused, wiring either one into a real QC signal is
  cheaper than building a new detector from scratch
- correction: the re-anchoring pass from propagation second-pass is now validated at full scope,
  not just a one-chain signal, real numbers on the video slide a few back; still has a known gap
  (checks "contains its own node", not "the flagged foreign node is gone"), so the true
  improvement is probably even larger than what is already shown; human review itself is still
  the bottleneck underneath all of this
- open the floor here

Timing: 2 to 3 minutes
-->