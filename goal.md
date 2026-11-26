# goal.md — Build an architecture that learns like a real brain

> Status (2026-11-26): loop run It 0–12 (stop rule reached). Final SPARC (It 12, test, 3 seeds): MNIST 94.4 ACC / FGT 0.9, Fashion 80.5 / FGT 10.4; all SHOULDs pass; MUST gaps: MNIST few-shot (42.7 vs 70), Fashion FGT (10.4 vs 10), Fashion few-shot old-class drop (2.8 vs 2). Details: ARCHITECTURE.md, docs/iteration-log.md.

## 1. Why
Today's LLMs and deep nets learn in ways no brain does.

| Brain | Backprop nets / LLMs |
|---|---|
| **Local plasticity**: a synapse changes using only its own pre/post activity plus a broadcast neuromodulator | **Global backprop**: exact error sent backwards through symmetric weights (weight transport); forward pass frozen while it happens |
| Learns **online** from one continuous stream; each moment seen once | Shuffled i.i.d. data, many epochs, then frozen |
| Keeps old skills while learning new ones | **Catastrophic forgetting** when trained sequentially |
| **One-shot** learning of a new thing (hippocampus) | Thousands of gradient steps per new concept (outside the context window) |
| Mostly **unsupervised**; labels and rewards are rare | Needs a dense training signal for every example |
| **Sparse** activity (a few % of neurons active) | Dense activity, high energy |
| **Knows when something is new** (mismatch / surprise signals) | Confidently wrong on novel inputs |
| **Sleeps**: offline replay consolidates memories | No offline consolidation phase |

## 2. Goal
Design a new architecture whose **learning** works like a brain's, prove it on a benchmark that measures brain-like learning, and iterate until it passes.
Scope: the learning machinery, tested at small scale (image streams). Language scale is on the roadmap (docs/roadmap.md), not in this loop.

SPARC is a **new (integrative) architecture**: it replaces how a model learns. It is not a harness around an existing model. Its parts come from prior work; the combination under strict rules, measured on P1–P10, is the contribution. A harness around a pretrained LLM (memory, sleep, reversal bolted on) would be a different project and breaks P1.

## 3. Testable definition of "learns like a brain"
Every change must also follow the **brain-first rule** (docs/rules.md).

- **P1 Local:** every weight update uses only pre-synaptic activity, post-synaptic activity and broadcast scalar modulators (three-factor rule). No backprop, no autograd, no weight transport. A single-layer delta rule is allowed (the error lives at the output neuron itself, like a teaching input).
- **P2 Online:** batch size 1, each example seen once, stream order (no shuffling, no epochs), learning and inference interleaved.
- **P3 Continual:** class-incremental stream, no task labels at test time.
- **P4 Fast:** learns a new class from 1–5 examples without disturbing old classes.
- **P5 Mostly unsupervised:** representations learned without labels; labels only bind representations to names.
- **P6 Sparse:** ≤ 5 % of units active per input in every learned layer.
- **P7 Fixed resources:** neuron budget fixed up front; no raw-input storage (no image replay buffer). Compressed internal codes are allowed (hippocampal index) with a declared, reported budget.
- **P8 Novelty-aware:** an internal surprise signal that is high for never-seen classes.
- **P9 Sleep:** an offline replay phase that measurably helps.
- **P10 Self-correcting:** a wrong association can be unlearned when contrary evidence arrives (reversal learning), without damaging unrelated classes. Brain basis: confident-error prediction-error signals (dopamine/noradrenaline) boost plasticity where the mistake happened; recalled memories become labile again (reconsolidation); old links are often outcompeted rather than erased (extinction).

## 4. Success criteria
**MUST** (mean of 3 seeds, both datasets unless stated)
- **M1** P1, P2, P6, P7 hold by construction (code audit: no autograd, no shuffling, no raw buffer).
- **M2** ACC ≥ 90 % MNIST, ≥ 75 % Fashion.
- **M3** FGT ≤ 5 pts MNIST, ≤ 10 pts Fashion.
- **M4** beats B1 and B2 on ACC; within 10 pts of B3 on MNIST.
- **M5** FEW-5 novel-class accuracy ≥ 70 % MNIST, ≥ 50 % Fashion, with old-class drop ≤ 2 pts.
- **M6** LBL5 ACC within 5 pts of full-label ACC.
- **M7** ≤ 5 % active units in every learned layer.

**SHOULD**
- **S1** NOVELTY AUROC ≥ 0.80 (MNIST).
- **S2** sleep improves ACC or FGT by ≥ 1 pt.
- **S3** every mechanism earns its place: ablating it costs ≥ 1 pt on some metric.
- **S4** beats B4 by ≥ 8 pts ACC.
- **S5** REVERSAL: new mapping adopted within ~100 examples per class, other classes drop ≤ 2 pts; NOISE: affected-class accuracy within 3 pts of the clean run.

Stop rule: round 1 (12 iterations) is done. Round 2 (docs/loop.md) has up to 8 iterations, with added CIFAR-10 targets; it stops early when all targets pass on the full test bench.

## 5. Where everything else is
| What | File |
|---|---|
| Working rules (brain-first rule, evaluation, model boundaries) | `docs/rules.md` |
| Benchmark protocol and baselines | `docs/benchmark.md` |
| Loop protocol (how an iteration runs) | `docs/loop.md` |
| Iteration log (all results, failures included) | `docs/iteration-log.md` |
| Current design, equations, results, gaps | `ARCHITECTURE.md` |
| Starting design (historical) | `docs/design-v1.md` |
| Roadmap after the loop | `docs/roadmap.md` |
| Agent team and briefs | `docs/agents.md` (+ `.claude/agents/`) |
| Environment and data sources | `docs/environment.md` |
| Audit of rules and numbers | `docs/audit.md` |
| Research: brain systems, prior work, warnings, backlog | `research/` (start at `research/README.md`) |
| Code | `sparc.py`, `v1.py` (model); `bench.py`, `data.py`, `baselines.py` (world); `tools/` (reports) |
| Run lists and results | `runs/`, `results/` |
