# Per-slice

<div class="flex flex-col items-center gap-3 mt-1">
  <img src="/images/per-slice-tree.png" class="ps-tree" />
  <img src="/images/perslice-jitter.png" />
</div>

<div class="text-center mt-3 mx-auto" style="max-width:64rem">
  <span class="opacity-80">Image mode is re-anchored on every slice, from that slice's own node. No video memory to drift.</span><br>
  <span style="color:#009E73"><b>Pro:</b> grounded and accurate on each frame.</span>&nbsp;&nbsp;&nbsp;
  <span style="color:#D55E00"><b>Con:</b> little continuity, the boundary jumps slice to slice, janky even when every frame is metrically fine.</span>
</div>

<!--
Opening: same idea, run per slice.

- image mode re-run on every slice from its own node, so no memory to drift onto the wrong cell
- each frame is grounded and accurate, stays in its own cell
- cost is continuity: slices are independent, so the boundary jumps (the strip is one neuron on five consecutive slices)
- looks janky as a stack even when each frame scores clean

Timing: 90 seconds
-->
