# Measurement, part 2: the membrane ridge filter

<div class="flex justify-center mt-2">
  <img src="/images/membrane-metric.png" />
</div>

<div class="mt-3 text-center opacity-80">A Sato ridge map of the membranes. It helps us catch milder bleeds and underfills the node metric misses.</div>

<!--
Opening: "A second, complementary measurement, also with no ground truth."

Beats: a Sato ridge filter turns the raw EM into a membrane map, the bright walls in the middle panel. Overlay it on a mask (right panel, mask in cyan, membrane in red): if a membrane ridge cuts across the mask border to border, that is a subtle bleed the node metric cannot see. Honest caveat: this is a v1. The ridge also lights up organelles, and a threshold, tau, trades sensitivity for specificity, so I treat it as a promising second signal, not a finished ruler.

Timing: 90 seconds
-->
