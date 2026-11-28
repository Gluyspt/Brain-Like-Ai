# goal.md — Toward a brain-like model that understands, thinks and reasons

> Status (2026-11-26): **Stage 1** rounds 1–3 done (test: MNIST 94.7 / FGT 0.7, Fashion 81.5 / FGT 7.6; open MUSTs: MNIST few-shot, few-shot old-class drop; relearning unsolved). Handoff for the next session: docs/handoff.md. **Stage 2** passes its own criteria on test (continuous stream, 3 seeds: online 0.924, retention 0.863, violation 0.951, completion5 0.860); formally done once stage 1's open MUSTs pass. **Stage 3** started: number sense, adds unseen pairs (val 0.78) where memorising baselines are at chance. Details: ARCHITECTURE.md, docs/iteration-log.md.

## 0. Ultimate goal (user, 2026-11-26)
A model that actually **understands**, can **think**, and does real **reasoning**, built from brain mechanisms (brain-first rule, docs/rules.md) and learning the way a brain does.

These words are vague, so the project uses testable meanings:

| Ability | Testable meaning | Brain basis |
|---|---|---|
| **Understand** | predicts what comes next and what an action will cause, not just labels the present | cortex as a prediction machine (predictive coding); world models |
| | grasps structure: applies a relation learned on some things to new things (A > B, B > C ⇒ A > C) | hippocampal–entorhinal cognitive maps of abstract knowledge |
| **Think** | holds information in mind and updates it step by step | prefrontal working memory, gated by the basal ganglia |
| | imagines before acting: simulates options internally | hippocampal replay and preplay; model-based planning |
| **Reason** | combines known pieces into new answers (composition, analogy) | prefrontal relational reasoning, abstract rule learning |
| | knows what it doesn't know and checks itself | metacognition (ACC/PFC conflict monitoring) |

Nobody has reached this in a brain-like system, so the project climbs a **ladder of stages**. Each stage has its own benchmark and must pass before the next builds on it (docs/roadmap.md; background in research/17):

1. **Perception + memory**: continual, few-shot, self-correcting learning without forgetting. **Current stage** (§2–§4).
2. **Prediction over time**: a world model of sequences.
3. **Structure and relations**: cognitive-map codes that transfer to new items.
4. **Working memory + gating**: multi-step tasks.
5. **Thinking as simulation**: planning by imagined replay.
6. **Grounded symbols and language**: words bound to sensory codes.
7. **Abstract reasoning**: novel puzzles that need composition.

The sections below define **Stage 1**.


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

## 2. Stage-1 goal
Design a new architecture whose **learning** works like a brain's, prove it on a benchmark that measures brain-like learning, and iterate until it passes.
Scope: the learning machinery, tested at small scale (image streams). Later stages (prediction, structure, working memory, planning, language, reasoning) are in docs/roadmap.md.

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

## 4. Stage-1 success criteria
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
| Roadmap: stages 1–7 toward understanding and reasoning | `docs/roadmap.md` |
| Agent team and briefs | `docs/agents.md` (+ `.claude/agents/`) |
| Environment and data sources | `docs/environment.md` |
| Engineering to-do: speed, Rust plan, running on your own PC | `docs/todo.md` |
| Audit of rules and numbers | `docs/audit.md` |
| Research: brain systems, prior work, warnings, backlog | `research/` (start at `research/README.md`) |
| Code | `sparc.py`, `v1.py` (model); `bench.py`, `data.py`, `baselines.py` (world); `tools/` (reports) |
| Run lists and results | `runs/`, `results/` |
