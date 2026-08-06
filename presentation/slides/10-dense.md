# Backup: dense segmentation and resolving overlaps

<div class="flex justify-center mt-2">
  <img src="/images/ways-of-seeing.png" class="max-h-[64vh] max-w-[80%] object-contain" />
</div>

<!--
Opening: one frame, four ways of looking at it

- Sato ridge filter lights up the membranes, the walls
- dense segmentation: all 25 prompted neurons at once
- automask (SAM class-agnostic): proposes every profile, no neuron identity
- overlaps split along the ridge walls, not an arbitrary line
- a direction we are prototyping, not a finished result

Timing: on demand
-->
