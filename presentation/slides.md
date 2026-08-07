---
theme: default
background: /images/dense-overlay-title.png
class: text-center
highlighter: shiki
lineNumbers: false
transition: slide-left
title: Segmenting the C. elegans nerve ring from EM
colorSchema: light
mdc: true
# No webfont fetch: this deck has to render identically offline, and the theme's
# remote font silently falls back to something decorative when it cannot load.
fonts:
  provider: none
  sans: Segoe UI
  serif: Georgia
  mono: Consolas
---

<div class="bg-black/60 rounded-xl px-8 py-7 backdrop-blur-sm inline-block">

# Segmenting the *C. elegans* nerve ring

<div class="text-[1.5rem] mt-3 opacity-95">With SAM</div>

<div class="text-[1rem] mt-6 opacity-75">
Zhen Lab progress update · July 2026
</div>

</div>

<!-- TODO: put your name in the byline above -->

<style>
:root {
  --deck-sans: 'Segoe UI', 'Helvetica Neue', Arial, Roboto, sans-serif;
}
.slidev-layout,
.slidev-layout h1, .slidev-layout h2, .slidev-layout h3,
.slidev-layout p, .slidev-layout li, .slidev-layout div {
  font-family: var(--deck-sans) !important;
}
h1 { font-size: 2.5rem; line-height: 1.12; }
h2 { font-size: 1.9rem; }
p, li { font-size: 1.2rem; }
:root { --primary: #0072B2; --secondary: #E69F00; --accent: #009E73; --text: #1f2937; }
.slidev-layout h1 { margin-bottom: 0.3rem; }
/* Figures: centred, native aspect, never stretched */
.fig { display: flex; justify-content: center; align-items: center; margin-top: 0.6rem; }
.fig img { max-height: 66vh; max-width: 92%; object-fit: contain; }
.fig-2 { display: flex; justify-content: center; gap: 2rem; margin-top: 0.6rem; }
.fig-2 img { max-height: 60vh; max-width: 46%; object-fit: contain; }
.cap { text-align: center; color: #6b7280; font-size: 0.95rem; margin-top: 0.3rem; }
</style>

<!--
Opening: every colour here is one neuron, segmented automatically, on one slice of a worm nobody has hand-segmented

- 10 to 15 minutes, a progress update
- last slide is four questions I would genuinely like answered

Timing: 45 seconds
-->

---
src: ./slides/02-goal.md
---

---
src: ./slides/03-data-flow.md
---

---
src: ./slides/04-sam.md
---

---
src: ./slides/05-three-methods.md
---

---
src: ./slides/05d-automask.md
---

---
src: ./slides/05c-perslice.md
---

---
src: ./slides/11c-current-work.md
---

---
src: ./slides/11d-current-work-video.md
---

---
src: ./slides/05b-propagation.md
---

---
src: ./slides/11e-current-work-video-propagation.md
---

---
src: ./slides/11f-current-work-video-secondpass.md
---

---
src: ./slides/07-merge-metric.md
---

---
src: ./slides/07b-membrane-metric.md
---

---
src: ./slides/08-results-arc.md
---

---
src: ./slides/09-before-after.md
---

---
src: ./slides/11-where-it-breaks.md
---

---
src: ./slides/17-tried-autofill.md
---

<!-- 18-tried-temporal.md pulled from the live deck for now, not verified yet. File kept on disk. -->

---
src: ./slides/19-tried-organelle-detectors.md
---

---
src: ./slides/12-advice.md
---

---
src: ./slides/13-thanks.md
---

---
src: ./slides/14-backup-4way-table.md
---

---
src: ./slides/15-backup-registration.md
---

---
src: ./slides/10-dense.md
---
