# Backup: the cross-worm ground-truth check

<div class="mt-8 mx-auto max-w-[54rem] text-left leading-relaxed opacity-90">

The second worm, **SEM-Dauer 1**, has real labelmaps from a VAST export. It is a sanity check, not the ruler, because it is a different animal and its masks are inset from the membrane by design.

Aligning the skeletons to it needs a per-section affine (the VAST stack was realigned slice by slice). That upgrade cut the median node residual from **19.6 px to 4.7 px**, and raised the on-mask rate from **67.9 to 85.7 percent** (91.7 at full resolution).

The target worm, by contrast, needs only a single global affine, verified stable across the stack.

</div>

<!--
Beats: this is why the target-worm merge metric is the primary ruler and the cross-worm GT is a corroborating check.
Timing: on demand
-->
