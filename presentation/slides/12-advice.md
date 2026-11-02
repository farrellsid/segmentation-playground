# Future directions

<div class="mt-1 mx-auto max-w-[62rem] text-left text-[1.05rem] leading-snug">

**Segmentation**
- Keep chipping at the per-method failure modes

**Measurement and QC**
- Build a hand-corrected ground-truth set
- MitoNet and NucleoNet both work, no consumer yet, wire one into a real QC signal

**Correction**
- Propagation's re-anchoring shows a real aggregate improvement at full scope, still checking the
  actual renders for new visual artifacts before calling it a clean win; also close the accept-gate
  gap, it only checks the frame's own node, not the specific flag that triggered it
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