# Manual skeleton annotations as prompts

<div class="fig-two flex flex-col items-center gap-3 w-full">

  <div class="text-center">
    <div class="opacity-70 mb-1 text-[1rem]">Image mode: one point, one mask</div>
    <img src="/images/skeleton-image-mode.png" />
  </div>

  <div class="text-center">
    <div class="opacity-70 mb-1 text-[1rem]">Video mode: carry the mask across slices</div>
    <img src="/images/video-propagation.png" />
  </div>

</div>

<!--
Opening: "SAM is a segmentation model you steer with a prompt, not train."

Beats: the CATMAID skeleton nodes are already those prompts, so we start with zero training. Two ways to use SAM2 or SAM3. Image mode, top: one point on a slice gives one mask. Video mode, bottom: seed a mask and it carries across slices. The next slide is how we combine these into whole-neuron methods.

Timing: 90 seconds
-->
