# Three ways to segment a neuron (that I tried with SAM)

<div class="fig-three flex justify-center items-start gap-10 w-full mt-2">

  <div class="text-center" style="flex:1">
    <div class="font-bold text-[1.15rem] mb-1">Propagation</div>
    <img src="/images/skeleton-tree-seeds.png" />
    <div class="opacity-70 text-[0.95rem] mt-1">seed a chain, follow the tree</div>
  </div>

  <div class="text-center" style="flex:1">
    <div class="font-bold text-[1.15rem] mb-1">Per-slice</div>
    <img src="/images/per-slice-tree.png" />
    <div class="opacity-70 text-[0.95rem] mt-1">image mode, a seed on every slice</div>
  </div>

  <div class="text-center" style="flex:1">
    <div class="font-bold text-[1.15rem] mb-1">Automask</div>
    <img src="/images/automask.png" />
    <div class="opacity-70 text-[0.95rem] mt-1">whole frame at once, messy</div>
  </div>

</div>

<div class="text-center mt-5 text-[1.05rem]" style="color:#009E73">SAM2 and SAM3 are a model swap used with all three</div>

<!--
Opening: "This is the core of what I tried. Three ways to turn the prompts into a whole-neuron mask."

Beats: propagation, seed a chain once and let SAM follow it down the branch tree, smooth but can drift. Per-slice, run image mode again on every slice from its own node, no drift, at the cost of the odd blown-up slice. Automask, segment the whole frame at once, which finds everything but with no neuron identity and a lot of mess to clean up. SAM2 and SAM3 are just a model swap used with all three, not a fourth method.

Timing: 2 minutes
-->
