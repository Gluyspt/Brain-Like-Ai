# 03 — Hippocampus, sleep and consolidation

## What the brain does
**Complementary learning systems (CLS):** the hippocampus learns fast, from single episodes, using sparse separated codes. The neocortex learns slowly, from many interleaved experiences. During sleep, the hippocampus replays recent memories mixed with old ones, so the cortex can absorb new knowledge without overwriting old knowledge. New memories depend on the hippocampus until consolidated.

Sleep also **downscales** synapses overall (synaptic homeostasis) and prunes weak ones.

**Reconsolidation:** recalling a memory makes it changeable again for a short while.

**Extinction:** an old association is often outcompeted by a new one rather than erased.

## Evidence
- McClelland, McNaughton & O'Reilly 1995, CLS theory, *Psych Review* [known].
- Tadros et al. 2022, sleep-like unsupervised replay reduces catastrophic forgetting [abstract] https://doi.org/10.1038/s41467-022-34938-7.
- HiCL: CA3 autoassociative memory plus prioritised replay [abstract] https://arxiv.org/abs/2508.16651.
- Synaptic homeostasis hypothesis. Tononi & Cirelli 2003/2014 [known].
- Reconsolidation. Nader, Schafe & LeDoux 2000, *Nature* [known].
- Arbitration between brain systems by their relative uncertainty. Daw, Niv & Dayan 2005, *Nat Neurosci* [known].

## How it helps our work
- **Sleep:** +1.3 ACC and −1.7 FGT on MNIST val (It 4). Removing it costs 1.2 pts Fashion ACC.
- **Hippocampus size:** 31 000 numbers, against 392 500 for a 500-image replay buffer.
- **Reconsolidation plus lability:** label reversal is adopted in 30 examples per class with no damage to other classes (S5). Without the lability window, Fashion ACC drops 8.7 pts.
- **CLS deferral** (It 7): while a class has fewer than 50 hippocampal traces, the fast system's vote dominates. Removing it costs 7.7 pts MNIST few-shot.

## How it might work next
- **Replay through IT as well as the readout.** Sleep could reactivate stored codes and let IT prototypes and readouts re-align, as in Tadros et al.'s noisy replay. This targets drift-induced staleness (Fashion FGT).
- **Prioritised replay:** replay traces whose class is currently confused, using the per-class error traces SPARC already keeps.
- **Pattern completion:** store partial codes and complete them with a CA3-like attractor, which saves memory.

## Test
Fashion FGT; FEW-5 old-class drop; the ablation without sleep.

## Status
Used: hippocampal store, surprise-gated writes, class-balanced sleep replay, slow readout, sleep cleanup, reconsolidation, lability, CLS deferral. Planned: IT-level replay, prioritised replay.
