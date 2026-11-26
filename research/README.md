# Research notes — brain mechanisms for SPARC

This folder holds the background research behind SPARC (findings, evidence, ideas). The goal is in `goal.md`; rules in `docs/rules.md`; results in `docs/iteration-log.md`. Full reference list: [references.md](references.md).

Each file covers one brain system or theme, with the same sections:
1. **What the brain does**: the mechanism, in plain words.
2. **Evidence**: papers, with links where we actually opened the source.
3. **How it helps our work**: which measured SPARC gap it could close (numbers from `results/`).
4. **How it might work in SPARC**: a concrete, local learning rule or circuit.
5. **Test**: which benchmark metric should move, and the ablation that proves it.
6. **Status**: used / tried / planned / rejected, with the `goal.md` docs/iteration-log.md iteration.

## Index

| File | Brain system | Status in SPARC |
|---|---|---|
| [01-plasticity-and-neuromodulators.md](01-plasticity-and-neuromodulators.md) | three-factor rules, ART, competitive Hebbian, intrinsic plasticity, critical periods | core: used |
| [02-sparse-codes-and-the-fly.md](02-sparse-codes-and-the-fly.md) | mushroom body, sparse expansion, pattern separation | used (fast readout); KC pathway rejected (It 8) |
| [03-hippocampus-sleep-consolidation.md](03-hippocampus-sleep-consolidation.md) | complementary learning systems, replay, sleep, reconsolidation | used (It 4, 5b, 7) |
| [04-hemispheres-and-visual-streams.md](04-hemispheres-and-visual-streams.md) | left/right specialisation, global vs local, ventral/dorsal streams | It 10: rejected (concatenation) |
| [05-top-down-attention-pfc.md](05-top-down-attention-pfc.md) | prefrontal bias signals, predictive feedback | used: rethink loop (It 12) |
| [06-thalamus-pulvinar-gating.md](06-thalamus-pulvinar-gating.md) | thalamic routing, synchrony, arbitration | partly (readout arbitration) |
| [07-basal-ganglia-dopamine.md](07-basal-ganglia-dopamine.md) | action selection, gating of working memory, reward prediction error | planned (roadmap docs/roadmap.md #3) |
| [08-cerebellum-and-learning-systems.md](08-cerebellum-and-learning-systems.md) | supervised forward models; division of labour between learning systems | partly (delta-rule readouts); roadmap docs/roadmap.md #1 |
| [09-related-ai-systems.md](09-related-ai-systems.md) | SoftHebb, HiCL, Nested Learning/Hope, BDH, Fly Model | context and baselines |
| [10-experiment-backlog.md](10-experiment-backlog.md) | prioritised list of the next experiments, each tied to a gap | living list |
| [11-glossary.md](11-glossary.md) | terms used across the project | reference |
| [13-credit-assignment.md](13-credit-assignment.md) | learning from mistakes deeper in the network without backprop | R2-2 in test |
| [14-eye-movements-and-temporal-continuity.md](14-eye-movements-and-temporal-continuity.md) | saccades, the trace rule, invariance from temporal continuity | R2-3 in test |
| [15-adaptive-forgetting-and-savings.md](15-adaptive-forgetting-and-savings.md) | what to forget, reminders, savings (fast relearning), engrams | REMIND/SAVINGS tests added |
| [16-episodic-recall.md](16-episodic-recall.md) | hippocampal recall at answer time (vs RAG) | R2-5 in test |
| [17-understanding-thinking-reasoning.md](17-understanding-thinking-reasoning.md) | brain basis of stages 2–7: prediction, cognitive maps, working memory, simulation, grounding, reasoning | ultimate goal |
| [12-prior-attempts-and-warnings.md](12-prior-attempts-and-warnings.md) | who tried this before (ART, Leabra, Spaun, HTM, Human Brain Project, bio-plausible backprop) and the warnings | read before claiming results |

## Rules for these notes
- **Evidence levels**: write next to each claim: **[read]** = we opened the source (link given); **[abstract]** = only the search snippet or abstract; **[known]** = a standard textbook result cited from memory, with no link. Upgrade a claim before it drives a design decision.
- **Brain-first rule** (CLAUDE.md): a mechanism enters SPARC only with its brain analogue named here. Record its cost too.
- **No myths**: popular claims such as "left-brained people" are checked and marked as such (see file 04).
- **Numbers** about SPARC come from `results/*.json` (see `results/report_tables.md`), never from memory.

## Where SPARC stands (test split, 3 seeds, It 9 config)
MNIST ACC 94.7 / FGT 0.9 / FEW-5 43.7; Fashion ACC 80.8 / FGT 10.9 / FEW-5 81.8. Open gaps: MNIST few-shot (target 70), few-shot old-class drop (2.2 / 3.2, target ≤ 2), Fashion FGT (target ≤ 10). See `ARCHITECTURE.md` §4.

## Takeaways that drove the starting design
1. Forgetting is mostly **interference**. Sparse, high-dimensional codes plus updating only active synapses removes most of it (fly).
2. **Recruit new units for novel input; refine old units only on a good match** (ART). Stability without freezing everything.
3. **Use-dependent plasticity** (learning rate ∝ 1/usage) makes heavily used units stable, like a critical period, while fresh units stay plastic.
4. Local, feedback-free layers can learn good features fast (SoftHebb), so backprop is not required for usable representations.
5. **Sleep with replay** can repair what waking learning damaged (SRC).
6. Gap: the systems above each cover a few properties, and most continual-learning systems still train with backprop over many epochs. In this scan we did not find one model tested on **all** of P1–P9 at once under strict rules. That combined, measured system is the contribution; do not claim the individual parts are new.
