# From raw EM to neuron meshes

<div class="flex justify-center mt-3">
  <img src="/images/data-flow/diagram-excalidraw.svg" class="max-h-[56vh] max-w-[94%] object-contain" />
</div>

<div class="mt-4 text-center opacity-80">The skeleton is the route down each neuron. We need the outline on every slice.</div>

<div class="mt-1 text-center opacity-60 text-[0.95rem]">EM dataset: sensory_ablated_dauer</div>

<!--
Opening: here is what goes in and what comes out

- input one: 2,354-slice EM stack, we work on about 300 slices (layers 1293 to 1628, densest part of the nerve ring)
- input two: Lucinda's hand-traced CATMAID skeletons, a line down the middle, the route not the outline
- output: a 3D mask per neuron for Blender
- second worm, SEM-Dauer 1, secured with real ground truth for scoring

Timing: 90 seconds
-->
