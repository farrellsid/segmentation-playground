# Already tried: detect the organelles, not their shape

<div class="fig-2 flex justify-center gap-6 mt-1">
  <div class="text-center">
    <img src="/images/mitonet-spotcheck.png" style="max-height:100px !important; max-width:100%; object-fit:contain;" />
    <div class="opacity-70 text-[0.8rem] mt-1">MitoNet: mitochondria</div>
  </div>
  <div class="text-center">
    <img src="/images/nucleonet-spotcheck.png" style="max-height:100px !important; max-width:100%; object-fit:contain;" />
    <div class="opacity-70 text-[0.8rem] mt-1">NucleoNet: nuclei</div>
  </div>
</div>

<div class="mt-2 mx-auto max-w-[56rem] text-[0.88rem] leading-tight">

Both real, pretrained EM detectors (empanada, BSD-3-Clause), generalize on sight: tight
boundaries, no cross-organelle false positives. Real gate, same bleed measurement as the
membrane-fill slide (z=1456, 117 neurons, `uf_min=0.6`), baseline `new_bleed=7`:

| Organelle mask source | New bleed cells | Moves the floor? |
|---|---|---|
| Classical shape/intensity filter | 7 | No, flat |
| **MitoNet + NucleoNet (real detections)** | **6** | Barely, one cell |

</div>

<!--
Opening: third thing tried, same target as the last two slides, the organelle-corrupted membrane
map, but detecting the real thing instead of a shape/intensity guess

- the classical shape filter calibrated to about 96 percent single-pixel Otsu noise, and moved the
  real bleed floor by exactly zero
- so this round swapped in a REAL detector: MitoNet and NucleoNet, both from the same pretrained
  EM organelle package, both spot-checked here and generalizing, tight masks, no cross-organelle
  false positives
- even a real, correct organelle mask barely dents the floor, one cell out of 16 to 18 filled, at
  best
- that is useful evidence, not just a shrug: it rules out "the detector was just noisy" as the
  explanation, and points at the membrane map's own resolution as the real bottleneck instead
- both detectors are real and working, just not wired into anything yet: no consumer built, that
  is on the future-directions slide

Timing: 2 minutes
-->