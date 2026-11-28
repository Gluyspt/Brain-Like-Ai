# Human signatures: does SPARC behave like people, and is that good for the work?

User idea (2026-11-27): after each experiment, compare the result with what humans do. A result that matches humans can be "right" even when a score is lower. But being human-like and being good for the work are **two separate questions**:

1. **Human-like?** Does SPARC show the known human effect (yes / no / opposite)?
2. **Good for the work?** Judged by the rule *copy the function, not the limitation* (docs/rules.md): keep a human-like effect if it does a job for understanding, memory or reasoning; a human limitation that does no job is not a goal.

| Combination | Meaning | Action |
|---|---|---|
| human-like **and** useful | the brain mechanism works as intended | keep / adopt |
| human-like, **costs** | a limitation copied with its mechanism | find the job it does; if none, look for the brain's partner mechanism that removes the cost |
| **not** human-like, useful | SPARC beats humans here | fine, if no rule is broken (e.g. no raw storage) |
| **not** human-like, harmful | a missing brain mechanism | research what the brain does, test it |

References marked [known] are cited from memory, not re-read.

| Human effect | Reference | SPARC shows it? | Good for the work? | Source in this repo |
|---|---|---|---|---|
| **Savings**: relearning forgotten material is faster and better than first learning | Ebbinghaus 1885 [known] | default: **opposite** (relearning lowers accuracy, Fashion 88.5 → 79.5); **SHY sleep: yes** (73.5 → 91.0, 2 of 3 seeds) | yes, wanted (P7, research/22) | R4-diag, R4-2, R4-4 |
| **Spacing effect**: spaced practice beats massed after a delay | Cepeda et al. 2006 [known] | **no** (massed ≈ spaced, MNIST 67.9 / 68.6) | neutral: humans gain because they forget between sessions; SPARC does not, so nothing to repair | R4-3 |
| **Sleep protects memory**: less forgetting after sleep than after equal wake; sleeping first protects against the next day's interference | Jenkins & Dallenbach 1924; Diekelmann & Born 2010 | **opposite** for a just-relearned memory: Fashion sleep 69.8 vs wake 80.0 (defaults), 83.0 vs 87.0 (SHY); sleep does help the other classes under SHY (+3.0) | partly fixed: the loss came from merging fresh IT units at sleep (R4-diag3); protecting fresh memories removes it (sleep − immediate −4.0 → +1.0 on Fashion s0), but a day awake still beats sleep | R4-5 – R4-7 |
| **Selective sleep**: sleep keeps what matters later, actively forgets the unimportant (REM, MCH neurons) | Wilhelm et al. 2011 [known]; Izawa et al. 2019 | **no**: sleep replays everything the hippocampus holds | probably yes (less interference): candidate | research/24 (REM parked → stage 5) |
| **Little catastrophic forgetting** across tasks | everyday experience; McCloskey & Cohen 1989 (networks forget, people do not) [known] | MNIST yes (FGT 0.7); Fashion partly (7.6) | yes | F6 test |
| **Few-shot learning**: a new category from 1–5 examples | Lake et al. 2015 [known] | weak on MNIST (FEW5 43.6 test; SHY 69.6 val), good on Fashion | yes, a stage-1 MUST | F6, R4-4 |
| **Novelty detection**: new things feel surprising | Ranganath & Rainer 2003 [known] | yes (AUROC 0.85–0.93) | yes | F6 |
| **Numerical distance effect**: far numbers are compared faster/better than close ones | Moyer & Landauer 1967 [known] | yes (ORDER 0.92–0.96 with distance effect) | yes: sign of a map-like number code | S3 |
| **Symbolic distance effect in transitive inference** | Bryant & Trabasso 1971 [known] | yes (novel pairs 0.97–1.00, far > near) | yes | S3-3 |
| **List linking from one bridge fact** | Treichler & Van Tilburg 1996 [known] | yes (cross pairs 1.00; backprop 0.56–0.60) | yes | S3-4 |
| **Schema-based one-trial learning** | Tse et al. 2007 | yes (new item: novel pairs 1.00 after one fact) | yes | S3-5 |
| **Statistical learning from a continuous stream** | Saffran et al. 1996 | yes (learns the worlds without boundaries; test online 0.924) | yes | S2-F2 |
| **Event segmentation at surprise peaks** | Zacks et al. 2007 [known] | **no**: SPARC's boundaries over-segment (995 vs ~400 real) | not needed for stage 2 scores; may matter for episodic memory later | S2-8, S2-9 |

Update this table whenever an experiment touches a known human effect.
