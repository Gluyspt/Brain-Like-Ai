# 08 — Cerebellum and the division of labour between learning systems

## What the brain does
The cerebellum learns **predictions** (forward models): given the current state and command, what comes next. It uses an error signal from the climbing fibres, a supervised, error-driven rule at a single synaptic layer.

A classic view divides learning among regions:
- **cortex**: unsupervised, building representations;
- **basal ganglia**: reinforcement, choosing actions;
- **cerebellum**: supervised, predicting.

Later work says the three overlap and act as one system.

## Evidence
- Doya 1999, *Neural Networks*: cerebellum = supervised, basal ganglia = reinforcement, cortex = unsupervised [abstract] https://web.math.princeton.edu/~sswang/developmental-diaschisis-references/doya00_neural-networks.pdf.
- Consensus paper 2016: cerebellum, basal ganglia and cortex as one integrated system [abstract] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5243918/.
- Forward models in motor control. Wolpert & Kawato 1998 [known].

## How it helps our work
SPARC already follows this division:

| Learning style | Region analogue | In SPARC |
|---|---|---|
| unsupervised | cortex | V1 and IT (no labels) |
| supervised, single-layer error | cerebellum-like | the fast readout and the sleep delta rule |
| reinforcement | basal ganglia | not yet (file 07) |

The big missing piece is **prediction over time**, roadmap §11.1, the step towards sequences and language.

## How it might work in SPARC
- **Temporal predictor**: a single-layer, cerebellum-like map P from the current IT code h_t to the next one, ĥ_{t+1} = P h_t, learned with a delta rule ΔP = η (h_{t+1} − ĥ_{t+1}) h_tᵀ. The prediction error becomes a better surprise signal than 1 − max match, because it measures "unexpected given context".
- Test on sequence streams (e.g. digit sequences with structure), then on character-level text.

## Test
A sequence benchmark (to be defined in goal.md §6 when roadmap §11.1 starts); NOVELTY AUROC using prediction error vs match-based surprise.

## Status
Partly (supervised single-layer readouts). Temporal prediction planned (roadmap §11.1).
