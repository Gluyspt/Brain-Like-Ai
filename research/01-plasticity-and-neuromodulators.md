# 01 — Plasticity rules and neuromodulators

## What the brain does
A synapse changes using what is available locally: the activity of the neuron before it (pre), the activity of the neuron after it (post), and a few chemicals broadcast widely. Dopamine, acetylcholine (ACh) and noradrenaline (NE) are the main ones. These chemicals say "this moment matters", "this was surprising" or "this was better or worse than expected". That gives the **three-factor rule**: Δw = η · M · f(pre) · g(post).

Other local controls sit on top of this:
- **Metaplasticity**: a synapse's own history sets how easily it changes.
- **Intrinsic plasticity**: a neuron adjusts its own excitability to keep its firing rate near a target.
- **Critical periods**: early sensory areas are very plastic during development, then plasticity closes and tuning stabilises.

## Evidence
- Adaptive Resonance Theory: refine a category only when the input matches it well (resonance); otherwise recruit a new one. Grossberg, Scholarpedia [abstract] https://doi.org/10.4249/scholarpedia.1569; survey [abstract] https://arxiv.org/abs/1905.11437; self-adjusting vigilance [abstract] https://arxiv.org/abs/2511.17983.
- ARTMAP match tracking: a wrong prediction raises vigilance and forces a new category. Carpenter & Grossberg, early 1990s [known].
- Hebbian learning with a global direction signal: Oja's rule plus competition, steered by a neuromodulator-like scalar [abstract] https://arxiv.org/abs/2601.21367.
- ACh/NE as expected vs unexpected uncertainty. Yu & Dayan 2005, *Neuron* [known].
- Homeostatic plasticity and intrinsic excitability. Turrigiano, many reviews 1998–2011 [known]. Adult-born dentate granule cells are hyperexcitable while young [known].
- Critical-period closure through PV-interneuron maturation and perineuronal nets. Hensch 2005, *Nat Rev Neurosci* [known].

## How it helps our work
These rules are SPARC's core. Every weight update is one of them (see `ARCHITECTURE.md` §2). Measured value, from the ablations (val, seed 0):
- Removing **refinement** costs 8.3 pts Fashion ACC.
- Removing **adaptive vigilance** costs 2.5 pts Fashion ACC and slows MNIST reversal from 30 to 80 examples per class.
- Removing **intrinsic excitability** raises FGT by 2.2 (MNIST) and 2.6 (Fashion).
- **Closing the V1 critical period** (It 9) cut Fashion FGT from 19.3 to 10.8 and raised ACC from 77.2 to 81.1. It was the biggest single gain of the project.

## How it might work next
- **Separate ACh and NE signals.** Today one surprise z-score does both jobs. ACh (expected uncertainty) could set the learning rate inside a familiar context; NE (unexpected uncertainty) could trigger resets and recruitment. SPARC's lability and uncertainty traces are a first step.
- **Graded critical period.** Today V1 closes at a hard step after 12 000 images. Instead, plasticity could decay with experience, and novel input statistics could reopen it slightly. That is like the adult plasticity that strong novelty can reopen.
- **BCM-style sliding threshold.** Replace the k-WTA cut-off in V1 with a threshold that slides with each unit's recent activity. That is a more local form of competition.

## Test
FGT and ACC on Fashion (the dataset most sensitive to drift), plus REVERSAL speed. Ablate each new rule separately.

## Status
Used: three-factor form, ART vigilance, match tracking, 1/usage metaplasticity, intrinsic excitability (It 7), critical-period closure (It 9). Planned: ACh/NE split, graded critical period.
