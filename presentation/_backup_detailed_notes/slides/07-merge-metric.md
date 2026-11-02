# Measuring without ground truth

<div class="flex justify-center mt-3">
  <img src="/images/merge-metric/diagram-excalidraw.svg" class="max-h-[56vh] max-w-[86%] object-contain" />
</div>

<div class="mt-4 text-center opacity-80">Each mask is grown from its own skeleton node, so the skeletons grade the masks.</div>

<!--
Opening: "The hard question on the target worm is, how do we know a mask is good when we have no ground truth?"

Beats: I built a metric that needs none. Every mask is grown from one neuron's own node, so if a different neuron's node lands inside the mask, that is a merge, a bleed. If a mask loses its own node, that is a dropout. This caught errors the QC flags were blind to. The theme of the whole project: invest in a fix only after measuring that it helps.

Timing: 2 minutes
-->
