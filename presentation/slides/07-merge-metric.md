# Measuring without ground truth

<div class="flex justify-center mt-3">
  <img src="/images/merge-metric/diagram-excalidraw.svg" class="max-h-[56vh] max-w-[86%] object-contain" />
</div>

<div class="mt-4 text-center opacity-80">Each mask is grown from its own skeleton node, so the skeletons grade the masks.</div>

<!--
Opening: on the target worm, how do we know a mask is good when we have no ground truth?

- built a metric that needs none
- each mask is grown from one neuron's own node
- a different neuron's node inside the mask: a merge, a bleed
- a mask that loses its own node: a dropout
- caught errors the QC flags were blind to
- theme of the whole project: invest in a fix only after measuring that it helps

Timing: 2 minutes
-->
