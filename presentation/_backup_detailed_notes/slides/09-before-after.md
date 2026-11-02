# Before and after, one frame

<div class="flex justify-center mt-3">
  <img src="/images/before-after.png" class="max-h-[62vh] max-w-[92%] object-contain" />
</div>

<div class="mt-3 text-center opacity-80">URAVL, one slice. The old mask swallows a neighbour's node (red). The new one stays in its own cell.</div>

<!--
Opening: "Here is that improvement on a single real frame, so the number is not abstract."

Beats: on the left, the old SAM2 mask fuses two adjacent profiles and swallows the neighbour's skeleton node, the red marker. On the right, the SAM3 per-slice mask sits inside the correct cell with the dividing membrane between it and the red node. The foreign node is exactly what the metric counts, so the picture and the number, 1 to 0, are the same fact.

Timing: 90 seconds
-->
