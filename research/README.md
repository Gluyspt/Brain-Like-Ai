# Research notes — brain mechanisms for SPARC

This folder holds the background research behind SPARC. `goal.md` keeps only the plan, rules and iteration log, and points here.

Each file covers one brain system or theme, with the same sections:
1. **What the brain does**: the mechanism, in plain words.
2. **Evidence**: papers, with links where we actually opened the source.
3. **How it helps our work**: which measured SPARC gap it could close (numbers from `results/`).
4. **How it might work in SPARC**: a concrete, local learning rule or circuit.
5. **Test**: which benchmark metric should move, and the ablation that proves it.
6. **Status**: used / tried / planned / rejected, with the `goal.md` §10 iteration.

## Index

| File | Brain system | Status in SPARC |
|---|---|---|
| [01-plasticity-and-neuromodulators.md](01-plasticity-and-neuromodulators.md) | three-factor rules, ART, competitive Hebbian, intrinsic plasticity, critical periods | core: used |
| [02-sparse-codes-and-the-fly.md](02-sparse-codes-and-the-fly.md) | mushroom body, sparse expansion, pattern separation | used (fast readout); KC pathway rejected (It 8) |
| [03-hippocampus-sleep-consolidation.md](03-hippocampus-sleep-consolidation.md) | complementary learning systems, replay, sleep, reconsolidation | used (It 4, 5b, 7) |
| [04-hemispheres-and-visual-streams.md](04-hemispheres-and-visual-streams.md) | left/right specialisation, global vs local, ventral/dorsal streams | It 10: rejected (concatenation) |
| [05-top-down-attention-pfc.md](05-top-down-attention-pfc.md) | prefrontal bias signals, predictive feedback | used: rethink loop (It 12) |
| [06-thalamus-pulvinar-gating.md](06-thalamus-pulvinar-gating.md) | thalamic routing, synchrony, arbitration | partly (readout arbitration) |
| [07-basal-ganglia-dopamine.md](07-basal-ganglia-dopamine.md) | action selection, gating of working memory, reward prediction error | planned (roadmap §11.3) |
| [08-cerebellum-and-learning-systems.md](08-cerebellum-and-learning-systems.md) | supervised forward models; division of labour between learning systems | partly (delta-rule readouts); roadmap §11.1 |
| [09-related-ai-systems.md](09-related-ai-systems.md) | SoftHebb, HiCL, Nested Learning/Hope, BDH, Fly Model | context and baselines |
| [10-experiment-backlog.md](10-experiment-backlog.md) | prioritised list of the next experiments, each tied to a gap | living list |
| [11-glossary.md](11-glossary.md) | terms used across the project | reference |

## Rules for these notes
- **Evidence levels**: write next to each claim: **[read]** = we opened the source (link given); **[abstract]** = only the search snippet or abstract; **[known]** = a standard textbook result cited from memory, with no link. Upgrade a claim before it drives a design decision.
- **Brain-first rule** (CLAUDE.md): a mechanism enters SPARC only with its brain analogue named here. Record its cost too.
- **No myths**: popular claims such as "left-brained people" are checked and marked as such (see file 04).
- **Numbers** about SPARC come from `results/*.json` (see `results/report_tables.md`), never from memory.

## Where SPARC stands (test split, 3 seeds, It 9 config)
MNIST ACC 94.7 / FGT 0.9 / FEW-5 43.7; Fashion ACC 80.8 / FGT 10.9 / FEW-5 81.8. Open gaps: MNIST few-shot (target 70), few-shot old-class drop (2.2 / 3.2, target ≤ 2), Fashion FGT (target ≤ 10). See `ARCHITECTURE.md` §4.
