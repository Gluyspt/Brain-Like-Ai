# 07 — Basal ganglia and dopamine: selection and gating

## What the brain does
The basal ganglia pick one action, or one item of working memory, among competing options, and suppress the rest. Dopamine neurons signal **reward-prediction error**: better than expected → burst; worse → dip. That signal trains the basal ganglia to choose well. In working-memory models, the basal ganglia also act as a gate: what enters the prefrontal "memory" (input gating) and what gets used (output gating).

## Evidence
- PBWM model (Hazy, Frank & O'Reilly 2006): basal ganglia gate prefrontal working memory, with gating learned by dopamine [abstract] https://pubmed.ncbi.nlm.nih.gov/16343792/; Frank, Loughry & O'Reilly 2001 [abstract] https://pubmed.ncbi.nlm.nih.gov/12467110/; review [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC2440774.
- Dopamine as reward-prediction error. Schultz, Dayan & Montague 1997, *Science* [known].

## How it helps our work
- **Hippocampal writes** today use surprise only, a fixed rule. A learned gate could decide what is worth storing from whether stored traces later helped: did replaying them reduce errors?
- **Roadmap §11.3, reward:** to learn from rewards rather than labels, the modulator M in the three-factor rule becomes a reward-prediction error. That would show that SPARC's rules work with sparse, delayed feedback.
- **Selection:** the k-WTA competition in IT is basal-ganglia-like winner selection, already in place.

## How it might work in SPARC
- **Learned write gate**: each class (or each IT unit) keeps a value trace v. After sleep, traces whose replay reduced the next day's errors raise v. The write probability becomes σ(z + v).
- **Reward mode**: replace labels with a reward r ∈ {0, 1} for a guessed class. Then δ = r − p(guess), and the fast-readout update is ΔA_guess ∝ δ · h.

## Test
LBL5-style sparse-feedback runs: accuracy when only a reward for a guess is given, not the label. Few-shot old-class drop for the learned write gate.

## Status
Planned (roadmap §11.3).
