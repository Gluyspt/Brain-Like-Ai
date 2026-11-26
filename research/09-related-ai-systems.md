# 09 — Related AI systems (context, not brain mechanisms)

These are machine-learning systems that borrow from the brain. We use them as ideas and comparison points. None of them meets all of SPARC's rules (local learning, online, no raw replay), so their numbers are **not** directly comparable.

| System | What it shows | Uses backprop? | What we took |
|---|---|---|---|
| SoftHebb (Journé et al., ICLR 2023) [abstract] https://arxiv.org/abs/2209.11883 | Multilayer Hebbian soft winner-take-all with no feedback signal; 99.4 % MNIST, 80.3 % CIFAR-10 with a linear classifier on top | no (features); offline epochs | proof that local features can be good; **depth** is the open lesson for SPARC (few-shot gap) |
| HiCL (AAAI 2026) [abstract] https://arxiv.org/abs/2508.16651 | Hippocampus-inspired continual learner: DG top-k sparsity, CA3 memory, prototype-gated experts, prioritised replay | yes (EWC + replay) | DG-style top-k, prioritised replay (file 03) |
| Nested Learning / Hope (Google, Nov 2025) [abstract] https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/ | A model as nested optimisation at several update frequencies; memory writes prioritised by surprise | yes | multi-timescale plasticity; surprise-gated writes |
| Dragon Hatchling, BDH (Pathway, Sep 2025) [abstract] https://arxiv.org/abs/2509.26507 | Language model of locally interacting neurons with Hebbian working memory; GPT-2-level results reported at 10M–1B parameters | partly | evidence that Hebbian fast weights scale (roadmap §11.4) |
| Fly Model (Zou et al. 2025) [abstract] https://arxiv.org/abs/2502.01427 | Mushroom-body plug-in against forgetting | with a backprop backbone | fast readout (file 02) |

## Baselines in our own benchmark (test, 3 seeds; `results/report_tables.md`)

| Model | MNIST ACC | Fashion ACC |
|---|---|---|
| B1, MLP trained sequentially | 19.6 | 19.9 |
| B2, B1 + 500-image replay | 86.9 | 75.6 |
| B3, joint training (upper bound) | 97.9 | 87.9 |
| B4, class means on raw pixels | 82.1 | 67.7 |
| SPARC (It 9) | 94.7 | 80.8 |

## Still to look up (scout questions)
- Published online class-incremental Split-MNIST / Split-Fashion numbers for ER, DER and GDumb with small buffers, to place B2 in context.
- Any other system evaluated on reversal learning plus class-incremental learning together.
