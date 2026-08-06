# Already tried: average organelles out across z, made it worse

<div class="flex justify-center mt-2">
  <img src="/images/temporal-sweep.png" class="max-h-[56vh] max-w-[92%] object-contain" />
</div>

<div class="mt-2 text-center opacity-80">z=1456, 117 neurons, uf_min=0.6. Widening the z-window monotonically increases bleed, it does not fall at any setting.</div>

<!--
Opening: second thing tried, aimed at the same organelle-corrupted membrane map as the last slide

- idea: average adjacent z-slices before running the ridge filter, so transient organelles
  (mitochondria, vesicles) wash out while the stable cell membrane survives
- real result: the opposite, bleed rises monotonically with window size, at every combine
  statistic tried (median, mean, max)
- caveat worth stating: this chart holds the fixed uf_min=0.6 gate, which pulls MORE cells into
  the grow step as the window widens (17/117 at window=0 up to 41/117 at window=2), so part of
  the rise is more cells being touched, not only worse-per-cell growth. A population-matched
  re-check (same cell count regardless of window) still found a real regression, smaller than
  this chart alone suggests but the same direction, window=1 close to flat, window=2 moderately
  worse
- root cause found alongside this: real registration misalignment, 14 to 30 percent of crop pairs
  needed their estimated shift clamped (up to 173 to 269px raw), so blur is not the whole story
- verdict: this route is closed, not a tuning problem

Timing: 90 seconds
-->