# For comparison: the same volume, propagation instead of per-slice

<div class="flex justify-center mt-2">
  <video src="/videos/dense-scroll-propagation-full.mp4" autoplay loop muted playsinline controls
         style="max-height: 330px; max-width: 92%; object-fit: contain;"></video>
</div>

<div class="mt-3 text-center opacity-80">Propagation's dense map over the same band (z 1293 to 1628), same colours, full 133-neuron scope.</div>

<!--
Opening: the propagation version of the last slide, same volume, same colours, different method

- full scope now (129 neurons), not the smaller subset an earlier version of this video used, so
  the density here is directly comparable to the second-pass video right after it
- watch a cell's shape hold steady frame to frame here, that consistency is propagation's whole
  appeal, and it's what per-slice structurally cannot give you
- the cost is on an earlier slide: drift over a long chain, occasional dropout; a chain missing its
  own anchor here just does not paint that chain's colour, not a visible gap at the frame level
  since something else usually has a mask by then
- controls are on here too, pause on any slice

Timing: 30 seconds
-->
