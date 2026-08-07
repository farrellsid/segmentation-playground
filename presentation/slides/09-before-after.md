# SAM3 trades bleed for tighter masks, not a clean win

<div class="flex justify-center mt-2">
  <img src="/images/before-after.png" class="max-h-[52vh] max-w-[92%] object-contain" />
</div>

<div class="mt-2 text-center opacity-80">URAVR, one slice, a real fix: the SAM2 mask fills most of the neighbouring cell (red node); the SAM3 mask stays inside its own compartment.</div>

<div class="mt-2 text-center" style="color:#D55E00">The real cost: SAM3 is more conservative (underfill 0.62 to 1.09), so it also misses more of the true cell, plus about 2.2 to 2.5x the compute.</div>

<!--
Opening: one real fix, but the honest picture is a tradeoff, not a win across the board

- left: SAM2 mask spreads across a real membrane wall and fills most of the neighbouring cell,
  clearly past the boundary, swallowing its skeleton node (red marker)
- right: SAM3 mask sits inside the correct compartment only, that bleed is gone
- but SAM3's masks are tighter everywhere, not just here: underfill nearly doubles (0.62 to 1.09),
  meaning it also leaves more of the true cell uncovered on average, that cost does not show up in
  a single "look, it's fixed" picture
- and it costs 2.2 to 2.5x the compute per cell
- so: fewer merges, more underfill, slower. Which one matters more depends on what you are
  optimizing for, not an unqualified upgrade

Timing: 90 seconds
-->