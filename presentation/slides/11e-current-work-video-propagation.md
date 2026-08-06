# For comparison: the same volume, propagation instead of per-slice

<div class="flex justify-center mt-2">
  <video src="/videos/dense-scroll-videomode.mp4" autoplay loop muted playsinline controls
         style="max-height: 330px; max-width: 92%; object-fit: contain;"></video>
</div>

<div class="mt-3 text-center opacity-80">Propagation's dense map over the same band (z 1293 to 1628), same colours. 334 of the 336 slices have a mask; the two missing at the start are the only gap, propagation needs its anchor established before it produces one.</div>

<!--
Opening: the propagation version of the last slide, same volume, same colours, different method

- same z band as before, 334 of 336 slices have a mask (per-slice has all 336)
- watch a cell's shape hold steady frame to frame here, that consistency is propagation's whole
  appeal, and it's what per-slice structurally cannot give you
- the cost is on an earlier slide: drift over a long chain, occasional dropout
- controls are on here too, pause on any slice

Timing: 30 seconds
-->
