# And with the correction pass: bleed down 36%, dropout down 81%

<div class="flex justify-center mt-2">
  <video src="/videos/dense-scroll-secondpass.mp4" autoplay loop muted playsinline controls
         style="max-height: 330px; max-width: 92%; object-fit: contain;"></video>
</div>

<div class="mt-3 text-center opacity-80">Same volume, same colours, propagation's flagged frames re-segmented from their nearest clean neighbour. Full scope, 4,041 chains: bleed rate 27.2% to 17.4%, dropout 14.0% to 2.7%, z-to-z consistency unchanged (0.681 to 0.668).</div>

<!--
Opening: the propagation video again, now with the second-pass correction applied

- same 336-slice band, same colours, so any visible difference from the last slide is the pass
  actually doing something, not a different render
- real numbers, full 133-neuron scope, not a curated subset: bleed rate falls 36% relative,
  dropout falls 81% relative, and z-to-z consistency barely moves, so this is not trading away
  propagation's whole appeal to get here
- caveat still open: the accept gate only checks the re-segmented frame contains its own node, not
  that the SPECIFIC foreign node that triggered the flag is gone, so the true improvement is
  probably even larger than these numbers, not smaller
- one-chain spot check when this landed called it "a first signal, not a verdict"; this is the
  verdict

Timing: 45 seconds
-->