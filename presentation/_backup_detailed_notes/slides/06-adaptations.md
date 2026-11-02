# Adaptations for neuron morphology

<div class="flex justify-center mt-4">
  <img src="/images/adaptations/diagram-excalidraw.svg" class="max-h-[52vh] max-w-[96%] object-contain" />
</div>

<div class="mt-4 text-center opacity-80">Modeled on Cheng's liver pipeline. Neuron morphology (thin, dense, branched) forced four changes.</div>

<!--
Opening: "The pipeline is modeled on Cheng's liver-EM pipeline, same idea, skeleton prompts to masks. Liver cells are fat, round, and far apart. Neurons are thin, packed, and branched, so four things had to change."

Beats: split each neuron at its branch points into linear chains, because SAM tracks one object and follows one arm at a branch. Crop to two resolutions, because a neurite is only about 3 pixels wide at the scale that fits in memory. Add negative prompts on the neighbours, because touching neurons bleed into each other. And add automated QC, because at this scale a human cannot proofread every frame blind.

Timing: 90 seconds
-->
