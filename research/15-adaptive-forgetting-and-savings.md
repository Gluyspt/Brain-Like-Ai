# 15 — Adaptive forgetting, reminders and savings

## What the brain does
- **People forget most details**: the Ebbinghaus forgetting curve (1885) shows steep loss of new material within days unless it is repeated [known].
- **But forgetting is selective and partly active.** Categories, meanings and skills persist, especially when used. The brain actively forgets to stay flexible and to generalise (Richards & Frankland 2017, *Neuron* [known]). Mechanisms: dopamine-driven forgetting in flies (Berry et al. 2012), neurogenesis-induced forgetting in the hippocampus (Akers et al. 2014), synaptic downscaling in sleep (Tononi & Cirelli) [known].
- **Much forgetting is lost access, not lost storage.** Relearning is faster than first learning (the *savings* effect, Ebbinghaus). Engram studies: "forgotten" memories in amnesic mice came back when their stored cells were activated directly (Ryan, Roy, Pignatelli, Arons & Tonegawa 2015, *Science*) [known].
- **Reminders keep memories alive**: occasional re-exposure (spaced repetition, retrieval practice) protects what is still useful [known].

## What this means for SPARC (user direction, 2026-11-26)
- **Categories should never be truly erased**: even when not recallable, a few reminders should bring them back fast. **Details** (single hippocampal episodes) may fade.
- The standard class-incremental benchmark shows each class once and never again, which is not how life works. Two new tests in `bench.py --only life` (docs/benchmark.md):
  1. **REMIND**: about 2 % of the stream are labelled reminders of earlier classes, from a reserved 10 % of their images. Metrics: REMIND_ACC, REMIND_FGT.
  2. **SAVINGS**: after the whole stream, relearn the most-forgotten old task from fresh reserved images. Metrics: examples to get back to 95 % of its old peak vs examples in first learning (SAV_ratio > 1 = savings), how far it recovers, and the drop on other classes.
- **External recall** (a notebook outside the model): possible, as people use notes, but storing raw images is a replay buffer and breaks P7; storing codes is what the hippocampus already does. Kept as a separate, clearly labelled future experiment, never in headline numbers.

## How SPARC already forgets
- Prototype merging (detail → gist), hippocampal eviction, sleep cleanup of contradicted memories, lability windows. Old IT prototypes are never deleted, only merged, which is an engram-like store that reminders can reactivate.

## Possible mechanisms (if SAVINGS is weak)
- **Silent engrams**: when a class's readout weakens, keep its IT prototypes and hippocampal index at low gain instead of letting new classes absorb them; a reminder restores the gain (reactivation → reconsolidation).
- **Retrieval-triggered lability**: a labelled reminder of an old class opens its lability window (the reconsolidation path already exists), so its links relearn quickly.
- **Use-dependent decay of hippocampal episodes** (not categories): traces not replayed or reactivated fade, freeing memory for new episodes.

## Status
Benchmark parts implemented (`--only life`, tested on B4). SPARC runs queued.
