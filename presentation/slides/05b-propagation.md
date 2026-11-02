# Propagation

<div class="two-img flex justify-center items-center gap-12 mt-3">
  <img src="/images/skeleton-tree-seeds.png" />
  <img src="/images/prop-bleed.png" />
</div>

<div class="text-center mt-5 mx-auto" style="max-width:62rem">
  <span class="opacity-80">Seed once, then track down the tree. The neuron is split into linear chains, since SAM follows a single arm at a branch.</span><br>
  <span style="color:#009E73"><b>Pro:</b> failures are coherent, easy to spot and re-anchor.</span>&nbsp;&nbsp;&nbsp;
  <span style="color:#D55E00"><b>Con:</b> the video memory drifts, so it bleeds more often and can lose track of the original cell.</span>
</div>

<!--
Opening: this is the original method.

- seed one mask, SAM's video memory carries it down the stack
- needs the branch tree split into linear chains (SAM tracks one arm at a branch)
- distinctive failure is drift: bleeds more often than per-slice, can lose the cell (this frame already spread into a neighbour)
- upside: drift is coherent, so errors are easy to see and re-anchor

Timing: 90 seconds
-->
