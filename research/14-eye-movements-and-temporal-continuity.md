# 14 — Eye movements and temporal continuity

## What the brain does
We move our eyes about three times a second, so the visual system sees each object as a sequence of slightly different glimpses. Objects persist over time, so things seen one after another usually belong to the same object. Neurons can use that: if a neuron's learning uses a short memory trace of its own recent activity, it comes to respond to all the views that followed each other. That gives invariance to position, size and pose, learned without labels.

## Evidence
- The trace rule: Földiák 1991, *Neural Computation* [known]; Wallis & Rolls 1997 [known].
- Slow feature analysis: Wiskott & Sejnowski 2002 [known].
- In monkeys, swapping an object's identity during a saccade retrains IT position tolerance: Li & DiCarlo 2008, *Science* [known].

## How it helps our work
Data scaling (docs/iteration-log.md R2-scale): SPARC barely improves with 5× more still images. A child learns from continuous experience, not still images. Shift invariance is what our 5×5 V1 plus fixed pooling only gives roughly, and it matters most for natural images (CIFAR-10) and for few-shot (one example covers more views).

## How it works in SPARC (R2-3, flag `glimpses`)
- Each training image is seen as `glimpses − 1` randomly shifted views (±2 px), then the fixated view.
- Later glimpses do not recruit new IT units: continuity suppresses novelty.
- **Trace rule**: the IT winners of the previous glimpse are pulled toward the current view, with strength `trace_w`/(1 + usage).
- The label (readout, hippocampus, correction) binds once, to the fixated view.
- Rule P2 (each example seen once): the glimpses are one viewing of one object, as with real eyes, not extra epochs. Stated as a declared interpretation.
- Cost: × `glimpses` time per image.

## Test
Quick screen, 3 datasets: glimpses 3 with trace 0.5, against glimpses 3 with trace 0 (shifted views only: control for the trace rule).

## Status
Implemented (flag off; MNIST predictions unchanged with it off). Screen queued after the data-scaling runs.
