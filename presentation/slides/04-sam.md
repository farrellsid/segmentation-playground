# Three ways to use SAM, none of them training

<div class="fig-three-stack flex flex-col items-center gap-2 w-full">

  <div class="text-center">
    <div class="opacity-70 mb-1 text-[1rem]">Image mode: one point (a prompt), one mask</div>
    <img src="/images/skeleton-image-mode.png" />
  </div>

  <div class="text-center">
    <div class="opacity-70 mb-1 text-[1rem]">Video mode: carry the mask across slices</div>
    <img src="/images/video-propagation.png" />
  </div>

  <div class="text-center">
    <div class="opacity-70 mb-1 text-[1rem]">Automask: SAM samples its own grid of points, no human prompt</div>
    <img src="/images/automask.png" />
  </div>

</div>

<!--
Opening: SAM is a segmentation model you steer at inference time, not train

- a "prompt" here just means a point you click, telling SAM where to look; CATMAID skeleton
  nodes already give us that point for free, so we start with zero training
- image mode (top): one point on a slice, one mask
- video mode (middle): seed a mask, it carries across slices
- automask (bottom): SAM samples its own grid of points across the frame as prompts, not one
  human-placed point, and proposes whatever object each sampled point lands on
- next slide: how these three combine into whole-neuron segmentation strategies, automask
  first, since that is where most people start

Timing: 90 seconds
-->
