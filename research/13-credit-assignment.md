# 13 — Learning from mistakes deeper in the network (credit assignment)

## What the brain does
Backprop sends an exact error backwards through the same weights (weight transport) in a separate backward phase. No known brain circuit does that (Crick 1989; Lillicrap, Santoro, Marris, Akerman & Hinton 2020, "Backpropagation and the brain", *Nat Rev Neurosci* [known]). The brain still learns from mistakes, in more local ways:
- **Category learning reshapes IT tuning** toward the features that separate the trained categories (Sigala & Logothetis 2002, *Nature*; Freedman et al. 2001 for PFC [known]), through dopamine-gated LTP/LTD: a teaching signal times local activity.
- Proposed brain-plausible ways to get error signals deeper: **feedback alignment** (fixed random backward weights), **target propagation**, **predictive coding**, **dendritic error** in apical dendrites, and **burst-dependent plasticity** [known].

## Warning
Bartunov et al. 2018: feedback alignment and target propagation did not scale to ImageNet [abstract] https://arxiv.org/abs/1807.04587 (research/12, W1). Try them, but don't expect them to fix scale on their own.

## How it helps our work
In SPARC, mistakes change only the top (readouts, recruitment, correction). V1 and V2 learn without any error, so their features are never tuned toward "what tells a coat from a pullover". This likely limits CIFAR-10 (36.1 % vs 47.7 % joint) and MNIST few-shot.

## How it might work in SPARC
1. **R2-2a, error-driven IT tuning** (`it_lvq`): on a wrong fast prediction, the active IT winners that vote for a wrong class move *away* from the input (Δw = −η·M·a·(x − w), with M = the confidence of the wrong prediction); winners of the true class move toward it as usual. This is a three-factor rule, like LVQ (Kohonen), and is one layer deeper than the readout.
2. **R2-2b, feedback alignment into V2** (if 2a helps): a fixed random matrix B carries the class error e = onehot − p back to V2 units, and the dictionary update of active V2 features is signed by (B e)_j and gated by M, during V2's critical period only.

## Test
CIFAR-10 ACC and NOVELTY, MNIST FEW-5, Fashion FGT; quick screen on 3 datasets, then val seeds 0–2.

## Results
- First screen invalid: the rule never fired (match tracking recruited first). Fixed.
- Fixed rule, quick screen: small gains (Fashion 82.1 vs 80.6, CIFAR 38.6 vs 37.6).
- **Full stream, 3 seeds: worse on Fashion** (80.8 / FGT 11.0 vs 81.3 / 9.5). Pushing old prototypes away from new look-alike inputs distorts the old classes over a long stream.

## Status
Rejected (R2-2a). Feedback alignment into V2 (R2-2b) not tried: with the IT-level version failing, error-driven change of shared features risks the same distortion; it would need protection of old classes (lability gating) first.
