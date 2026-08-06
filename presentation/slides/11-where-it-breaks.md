# Common error modes, by method

<div class="mt-1 mx-auto max-w-[58rem] text-[0.92rem] leading-tight">

**Propagation: merge issues**, drift bleeds into a neighbour &nbsp;·&nbsp;
**Per-slice: jankiness + mitochondria-limited masks** &nbsp;·&nbsp;
**Both: nucleus-limited masks**

</div>

<div class="flex justify-center mt-1">
  <img src="/images/failure-modes.png" class="max-h-[34vh] max-w-[80%] object-contain" />
</div>

<div class="mt-1 text-center opacity-70 text-[0.85rem]">All three read as CLEAN to the merge metric: a bleed-and-dropout floor, not the whole truth.</div>

<!--
Opening: the honest failures, organized by which method actually produces them

- propagation's failure mode IS merging, that's the bleed-rate gap already on the results slide,
  not a new number, just naming what drift actually looks like
- per-slice trades that away for two of its own: fragments (jankiness, no temporal consistency to
  lean on) and mitochondria-limited masks (locks onto a small dark round organelle next to the node)
- nucleus capture hits both, since a nucleus is round and membrane-bound just like a cell, shape
  alone cannot disambiguate them
- all three in the picture read as clean to the metric: mask contains its own node, no foreign
  node, so the ruler is a bleed-and-dropout floor, not a correctness guarantee
- other known hard cases: thin faint neurites, branch points

Timing: 2 minutes
-->