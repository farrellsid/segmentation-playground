# How the pipeline works, for a general audience

This is the plain-language tour of what the project does and how. It assumes no machine-learning or
imaging background. If you want the technical version, read
[architecture.md](architecture.md); for what is planned next, read [roadmap.md](roadmap.md). This
page is the one to hand to a labmate, a collaborator, or a new student who just wants to understand
the idea.

The second half of the page is a set of graphic ideas: pictures that would make this easier to
follow. Two already exist (the CATMAID skeleton diagram and the four-frame propagation strip); the
rest are sketches to build.

---

## 1. The goal, in one breath

A *C. elegans* worm has a nervous system of about 300 neurons. We have a tall stack of electron
microscope images: the worm sliced into roughly 300 thin sections, each one photographed at very
high magnification. Inside those gray, grainy images are the cross-sections of every neuron.

We want a 3D model of each neuron, the kind you can spin around in Blender. To get there, every
neuron has to be traced through every slice it appears in and its outline filled in. By hand that is
brutal: hundreds of slices, and a few thousand separate neuron branches to follow. The pipeline does
that tracing semi-automatically.

The jump in scale is part of why this is hard. We start with a whole animal about a millimeter long,
end up looking at features a few billionths of a meter across, and have to keep track of which blob
in one slice is the same neuron as a blob in the next.

## 2. What we start with

Two inputs go in.

The first is the raw images: the stack of EM slices, saved as a `.tif` file. Think of this as the
road. It is all there, but it is gray and crowded, with every neuron packed in together.

The second is a CATMAID skeleton. Earlier in the project, a person (Lucinda) traced each neuron by
hand as a stick figure: a set of connected points running down the middle of the neuron, slice to
slice. This is the route, like a line drawn on a map. It tells us where a neuron goes, but not its
actual shape or boundary on any given slice.

So we have the route and we have the road, and they do not yet line up. The pipeline's whole job is
to turn the route into a filled-in shape on every slice: to color in the neuron, not just trace its
centerline.

## 3. The key idea: treat the slice stack like a video

Two ideas make this tractable.

First, the stack of slices behaves a lot like a video. Play the slices in order and a single neuron
is an object that sits in roughly the same place from one frame to the next, shifting and changing
shape a little each time. Tracking an object across video frames is a problem people have already
built good tools for.

Second, the tool we use, SAM2, is one of those. SAM2 is an AI segmentation model with two handy
abilities. You can point at something in an image and it draws the outline of just that thing. And
in its video mode, once you have outlined an object on one frame, it will follow that object across
the following frames on its own. Point once, track everywhere.

Put those together and the recipe follows: outline a neuron carefully on one slice,
then let SAM2 carry that outline up and down the whole stack.

## 4. The recipe, step by step

**Break the branching neuron into simple chains.** A real neuron branches like a tree, and SAM2's
tracking works best on one simple object at a time, not a forking one. So we first cut the tree into
straight, non-branching paths, called chains. Each branch point becomes the end of one chain and the
start of another. We also drop in extra in-between points (virtual nodes) so the guidance is dense
enough along each chain.

**Point at the target on one slice.** We pick a slice near the middle of a chain and tell SAM2 what
we want. A positive prompt (think of it as a green dot) goes on the target neuron, sitting right on
the skeleton point. Negative prompts (red dots) go on the neighboring neurons that crowd around it,
meaning "not these." SAM2 returns a clean outline of just the target on that slice.

**Let it propagate through the stack.** That single outline becomes the seed for SAM2's video mode,
which then tracks the neuron outward in both directions, slice by slice, up the stack and down,
until it has an outline on every slice. Stacking those outlines gives a 3D shape: the neuron's mask
volume.

**Find it cheaply, then zoom in.** The full images are huge (about 9000 by 9000 pixels), and feeding
that to SAM2 at full size can exhaust a normal computer's memory. So the first pass runs on a
shrunken version of the image just to find roughly where the neuron is. Then we crop a window
tightly around it and rerun at high resolution only inside that small window. We get full detail
where it matters without paying for it everywhere.

## 5. The human stays in the loop

The pipeline is semi-automatic on purpose. The machine runs every chain on its own, usually
overnight, and as it goes it scores its own work with automatic quality checks. Those checks flag
the slices that look wrong: a mask that suddenly balloons, jumps, or shrinks to nothing.

The next day, a person opens only the flagged results in a review tool and fixes them, chain by
chain. The human is the scarce, expensive resource, so we spend their attention only where the
machine is unsure, not on the slices it clearly got right.

This is the core payoff. Even at the cost of one human-approved starting point per chain, the
pipeline is far faster than painting every neuron on every slice by hand. The expensive part, the
long tracing through hundreds of slices, is what the machine takes off the person's plate.

## 6. How good is it, honestly

It works and it is much faster than hand-tracing, but it is not perfect, and being honest about the
weak spots matters.

Two failure modes show up most. At a branch point, where a neuron splits in two, the tracker tends
to follow only one arm and lose the other, because it keeps chasing the shape it has already been
following. And very thin neurites can fade out: when the image is shrunk for the cheap first pass, a
neurite only a few pixels wide can be too faint to hold onto.

Until recently we had no fair way to measure any of this. We tuned the pipeline by feel. That
changed when we got ground truth from a second worm: a separate EM stack where the neuron shapes
were confirmed by hand. We can now lay our automatic result next to a known-correct answer and
actually score it. One useful score asks, in effect, how far you can trace along a neuron before
hitting a mistake. The longer that distance, the better. The catch worth stating plainly is that
this second worm looks somewhat different from our main one, so it tests how well the method
generalizes rather than giving a perfect in-house grade.

## 7. Where it is headed

With a real way to measure quality in place, the plan is a short ladder, each rung building on the
one below:

- **Build the ruler** (done): set up the ground-truth scoring so every later change can be judged on
  evidence instead of feel.
- **Free wins**: drop in known improvements to the tracking that need no retraining.
- **Teach SAM2 our images**: SAM2 learned on everyday photos and video, which look nothing like EM.
  Giving it focused practice on confirmed neuron examples should help it adapt.
- **Fix branching properly**: handle the split-in-two case with a sturdier method than single-object
  tracking.
- **Smarter quality checks**: learn from the confirmed examples which results are actually wrong, so
  the machine sends the person a shorter, better-chosen list to review.

---

## Graphics that would make this clearer

A ranked set of pictures for a general audience. For each: what it shows, why it helps, and a rough
layout to build from. Two already exist and are noted as such; the rest are proposals.

The ranking is by how much each one helps someone understand the pipeline for the first time, not by
how easy it is to draw.

### 1. Scale-funnel hero (goes with section 1)

*What it shows.* A left-to-right funnel: a whole worm, then a zoom box into its body, then one gray
EM slice, then that slice as one card in a tall stack, then a clean colored 3D neuron in Blender.

*Why it helps.* It answers "what are we even looking at" in one glance and sells the scale jump from
a whole animal down to nanometer features. It is the single best opener.

*Layout.* Five panels in a row joined by arrows, each panel labeled with its real size (about 1 mm,
then a few microns, then nanometer detail). Keep the EM panels gray and the final neuron in a bright
color so the payoff pops.

### 2. Route vs road input pair (goes with section 2)

*What it shows.* The same EM slice twice, side by side. On the left, only the CATMAID skeleton drawn
as a thin line or a few dots (the route). On the right, the neuron fully filled in as a colored mask
(the road, the thing we want).

*Why it helps.* It makes concrete the gap the pipeline closes: we are handed a centerline and have
to produce a shape. The before/after framing is immediately readable.

*Layout.* Two identical EM crops. Left labeled "what we are given: a traced centerline," right
labeled "what we need: the filled outline." A short arrow or "pipeline" label between them.

### 3. Stack-as-film-strip tracking metaphor (goes with section 3)

*What it shows.* The EM slices drawn as a strip of film frames, with one neuron highlighted in the
same color across consecutive frames, drifting and reshaping a little each frame.

*Why it helps.* It plants the central metaphor (slices behave like video; SAM2 tracks an object
across frames) using something everyone already understands, a moving subject in a video.

*Layout.* Four or five film-strip frames left to right, sprocket holes optional for the metaphor.
The target neuron tinted the same color in each, its outline shifting slightly. A caption: "to the
tracker, the stack is a video and the neuron is the moving subject."

### 4. Tree into colored chains (goes with section 4, improves the existing skeleton diagram)

*What it shows.* A branching neuron tree on the left; on the right, the same tree cut at its branch
points into several straight paths, each path a different color.

*Why it helps.* "Break the branching neuron into simple chains" is the least intuitive step. Showing
one forked shape becoming a handful of simple colored strands makes the cut obvious.

*Layout.* Left: a black tree with its branch points circled. An arrow. Right: the same geometry,
recolored so each non-branching run is its own color, with the branch points marked as the seams.
This can extend the existing CATMAID skeleton graphic rather than replace it.

### 5. Point-at-it prompts (goes with section 4)

*What it shows.* One EM slice crowded with several neurons. A green dot sits on the target; red dots
sit on the neighbors. An arrow leads to the same slice with only the target outlined.

*Why it helps.* It turns "positive and negative prompts" into something physical: you point at the
one you want and point away the ones you do not. No jargon needed.

*Layout.* Left panel, the busy slice with one green dot and two or three red dots, a small legend
("green: this one / red: not these"). Arrow. Right panel, the same slice with a single clean colored
outline on the target.

### 6. Triage funnel (goes with section 5)

*What it shows.* A funnel: thousands of chains enter at the top, the machine runs them overnight in
the middle, and only a thin trickle of flagged chains comes out the bottom to a single human
reviewer.

*Why it helps.* It is the clearest way to show the human-in-the-loop design and why it scales: the
person only ever sees the small flagged fraction.

*Layout.* Wide top labeled with a large chain count, a "machine, overnight, auto-QC" band in the
middle, a narrow spout labeled "only flagged slices," and a single person icon at the bottom labeled
"reviews and fixes only these."

### 7. Speedup comparison (goes with section 5)

*What it shows.* Two bars or two clocks side by side: hand-tracing every slice versus the pipeline
with one approved start per chain.

*Why it helps.* It states the payoff in the terms a busy reader cares about, time, and backs up the
"much faster" claim with a picture.

*Layout.* Two horizontal bars, hand-tracing long, pipeline short, each labeled with its rough effort
per neuron. Keep the multiple qualitative ("many times faster") unless a measured figure is on hand,
so the graphic stays honest.

### 8. The branching failure, drawn honestly (goes with section 6)

*What it shows.* A Y-shaped neuron where the tracked mask correctly covers the trunk and one arm but
misses the second arm.

*Why it helps.* Showing a known weakness builds trust and explains why "fix branching" is on the
roadmap. It also makes the limitation concrete rather than abstract.

*Layout.* A single Y-shaped neuron, the covered trunk and one arm tinted solid, the missed arm left
gray with a small "missed" tag. Caption: "at a split, the tracker tends to follow only one arm."

### 9. Ground-truth ruler (goes with section 6)

*What it shows.* Two versions of one neuron overlaid: the pipeline's result and the hand-confirmed
truth, with agreement and disagreement shown in different colors. A small inset illustrates "how far
can you trace before a mistake" as a length along the neuron.

*Why it helps.* It explains, without math, both how we now check quality and what the main score
means.

*Layout.* One neuron with overlap regions in green and disagreement in red or orange. An inset
strip showing a trace running along the neuron until it hits an error, with the error-free length
called out. Caption: "we score against a hand-confirmed worm; longer error-free traces are better."

### 10. Roadmap staircase (goes with section 7)

*What it shows.* Five rising steps labeled with the plain-language stages from section 7, with the
first step ("build the ruler") marked done.

*Why it helps.* It leaves the reader with a clear sense of direction and shows the work is sequenced,
not a wish list.

*Layout.* A simple rising staircase, one short label per step, a check mark on the first. Optional: a
small icon per step (a ruler, a wrench, a graduation cap, a fork, a checklist).

### Lower-priority extras

These are worth building only if there is room; the ten above carry the story.

- **Image mode to video mode handoff.** Two stacked boxes: "outline once, carefully" feeding
  "track everywhere, automatically." Clarifies that the careful single-slice step and the
  many-slice tracking step are different jobs.
- **What teaching SAM2 our images means.** A small before/after: SAM2 shown a stack of everyday
  video frames (what it learned on) next to an alien-looking EM slice, with a "needs practice on
  these" note. Makes the finetuning rung in the roadmap land.
- **The find-it-cheap-then-zoom-in trick as its own panel.** A shrunken full image with a rough box
  found on it, then that box reopened at full resolution. Explains the two-pass memory trick for
  readers who ask why it is not all done at full size.
