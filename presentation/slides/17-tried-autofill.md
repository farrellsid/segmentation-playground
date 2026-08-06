# Already tried: grow masks to the membrane, fixes a lot and breaks a lot

<div class="flex justify-center mt-2">
  <img src="/images/dense-fill.png" class="max-h-[58vh] max-w-[94%] object-contain" />
</div>

<div class="mt-3 text-center opacity-80">z=1456, 117 neurons grown out to their ridge-filter walls. Underfill 0.49 to 0.31, but area +86% and 27 of 117 hit the runaway cap. The green panel is what the fill added: a lot of it leaks along membranes into neighbours.</div>

<!--
Opening: first of three things tried against the failure modes on the last slide, this one targets underfill

- grow each neuron's mask out to its Sato ridge-filter walls (the membrane map from the metrics
  section), one frame, every neuron at once
- underfill really does drop (0.49 to 0.31) and many masks become believable full cells
- but area jumps +86%, 27 of 117 masks blow past the 5x runaway cap, green panel shows growth
  leaking along membranes into neighbours, trading underfill for bleed
- caveat: watershed is run PER NEURON independently, so neurons overlap and the overlaps are
  resolved by dumb higher-id stacking, not a real partition
- it also does not fix nucleus capture, growth stays trapped by the nuclear envelope
- so: promising but not free, needs a tighter guard or a cleaner (higher-res or learned) membrane
  map before it is usable on its own

Timing: 90 seconds
-->
