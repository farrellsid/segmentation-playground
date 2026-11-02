# Dense segmentation, and resolving overlaps

<div class="flex justify-center mt-2">
  <img src="/images/ways-of-seeing.png" class="max-h-[64vh] max-w-[80%] object-contain" />
</div>

<!--
Opening: "One frame, four ways of looking at it."

Beats: the Sato ridge filter lights up the membranes, the walls. The dense segmentation is all 25 prompted neurons at once. Automask, SAM in its class-agnostic mode, proposes every profile but with no neuron identity. Where masks overlap, we split them along the ridge walls rather than an arbitrary line. This is a direction we are prototyping, not a finished result.

Timing: 90 seconds
-->
