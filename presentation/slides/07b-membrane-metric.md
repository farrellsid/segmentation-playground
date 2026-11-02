# Measurement, part 2: the membrane ridge filter

<div class="flex justify-center mt-2">
  <img src="/images/membrane-metric.png" />
</div>

<div class="mt-3 text-center opacity-80">A Sato ridge map of the membranes. It helps us catch milder bleeds and underfills the node metric misses.</div>

<!--
Opening: a second, complementary measurement, also with no ground truth

- Sato ridge filter turns raw EM into a membrane map (bright walls, middle panel)
- overlay on a mask (right panel, mask in cyan, membrane in red)
- ridge cutting across the mask border to border: a subtle bleed the node metric can't see
- honest caveat: this is a v1
- ridge also lights up organelles; threshold tau trades sensitivity for specificity
- a promising second signal, not a finished ruler

Timing: 90 seconds
-->
