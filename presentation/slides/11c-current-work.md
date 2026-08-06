# Current work: a dense multi-neuron map

<div class="flex justify-center mt-2">
  <img src="/images/current-work.png" />
</div>

<div class="mt-3 text-center opacity-80">Every per-slice mask merged into one dense labelmap. 124 neurons across the stack, about 117 on this frame.</div>

<div class="mt-1 text-center opacity-60 text-[0.95rem]">Overlaps are resolved along the membrane walls, not an arbitrary line.</div>

<!--
Opening: what we are building toward now, the whole cross-section at once

- every neuron's per-slice mask, all 124, merged into one dense labelmap on a single frame
- this is the input to the 3D reconstruction
- where the overlap-resolution and arbitration work go next
- if I make a scroll-through video, this is the slide it plays on

Timing: 60 seconds
-->
