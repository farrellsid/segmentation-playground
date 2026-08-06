# Automask

<div class="flex items-center justify-center gap-12 mt-4 w-full">

  <img src="/images/automask.png" />

  <div class="text-left" style="max-width:40%">
    <div class="opacity-80 mb-4">The whole frame is segmented at once, with no prompts at all.</div>
    <div class="mb-2" style="color:#009E73"><b>Pro:</b> finds everything, no skeleton needed.</div>
    <div style="color:#D55E00"><b>Con:</b> no neuron identity, and messy. A work in progress.</div>
  </div>

</div>

<!--
Opening: automask runs SAM's automatic mask generator on the whole frame, no prompt at all

- finds essentially every profile, which is appealing
- but assigns no neuron identity, and the output is messy (spurious, overlapping masks to clean up)
- least developed of the three, a work in progress
- feeds the dense-map direction later

Timing: 60 seconds
-->
