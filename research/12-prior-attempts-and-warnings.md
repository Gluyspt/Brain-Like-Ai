# 12 — Prior attempts and warnings

People have tried to build brain-like learners for 50 years. This file lists the main attempts, what went wrong for them, and what that means for SPARC. Read it before claiming a result.

## Prior attempts

| Project | What it was | Outcome | Evidence |
|---|---|---|---|
| **ART** (Grossberg, 1976 →) | stable category learning: resonance vs recruitment | influential theory; rarely competitive on standard benchmarks. SPARC uses its vigilance and match-tracking rules | [abstract] https://doi.org/10.4249/scholarpedia.1569 |
| **Leabra** (O'Reilly) | local Hebbian plus error-driven learning; models of PFC and basal ganglia | explains brain and behaviour data well; not used for AI performance | [known]; PBWM [abstract] https://pubmed.ncbi.nlm.nih.gov/16343792/ |
| **Spaun** (Eliasmith et al. 2012, *Science*) | 2.5M spiking neurons; sees digits, answers with an arm; 8 tasks | a demo of many brain parts working together; far smaller than a brain, and its authors note it falls short in power and scale | [abstract] https://compneuro.uwaterloo.ca/files/publications/eliasmith.2012.pdf; [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC4124604 |
| **Numenta HTM / Thousand Brains** | cortex-inspired sequence learning | few independent tests; an independent study found that the only published evaluation of its predictions was Numenta's own; a forum post by one of its authors called 91 % on MNIST "extremely bad" | [abstract] https://pdf.arxiv.org/pdf/1512.05463; [abstract] https://discourse.numenta.org/t/neuromorphic-architecture-for-the-hierarchical-temporal-memory/4381 |
| **Human Brain Project** (EU, 2013–2023) | detailed simulation of the human brain | over 750 neuroscientists signed a 2014 open letter calling it premature and badly run; governance was reformed | [abstract] https://www.nature.com/articles/513027a; https://www.pbs.org/newshour/science/neuroscientists-criticize-worlds-largest-neuroscience-project |
| **Biologically plausible alternatives to backprop** (feedback alignment, target propagation) | learn without exact backprop | Bartunov, Santoro, Richards, Marris, Hinton & Lillicrap 2018: none scaled to ImageNet, though they looked fine on MNIST and CIFAR-10 | [abstract] https://arxiv.org/abs/1807.04587 |
| **SoftHebb, BDH, HiCL, fly models** | recent brain-inspired learners | promising; each covers only some brain properties | file 09 |

## Warnings, and what each means for SPARC

### W1. Toy benchmarks do not predict scale
Brain-like rules that work on MNIST often fail on ImageNet (Bartunov et al. 2018).
**For SPARC:** our 94.7 % on MNIST proves only that the mechanisms work at toy scale. **Action:** an early CIFAR-10 check (B-list below).

### W2. Continual-learning benchmarks are easy to fool
GDumb (Prabhu, Torr & Dokania, ECCV 2020) greedily stores samples and retrains from scratch on them at test time. It beat most published continual-learning methods, often by large margins, even on the settings they were designed for [abstract] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf.
**For SPARC:** a method can look good next to weak baselines. **Action:** baseline **B5 = GDumb** with a 500-image memory (it violates P7, like B2).

### W3. The Bitter Lesson
Over 70 years, general methods that scale with compute have beaten hand-built knowledge (Sutton 2019) [abstract] https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf.
**For SPARC:** we have about 15 hand-designed mechanisms and many settings, and MNIST few-shot is still stuck. Counterpoint: we study *how learning works*, not task knowledge, and critics note that deep-learning architectures are themselves carefully hand-designed. **Action:** prefer mechanisms that remove settings (self-tuning) over ones that add them; count settings in ARCHITECTURE.md.

### W4. Realism usually costs performance
Leabra, Spaun and HTM explain the brain better than they perform. SPARC is 3.2 pts behind joint backprop on MNIST and 7.1 on Fashion.
**For SPARC:** claim only where brain-like learning *wins*: forgetting (0.9 vs 15.0), few labels, novelty detection, reversal (30 vs 238 examples per class), and memory (31k vs 392.5k numbers). Do not claim "a better image model".

### W5. Self-evaluation and hype
HTM and the Human Brain Project were criticised for evaluating themselves and for overclaiming.
**For SPARC:** keep the safeguards: an independent auditor agent, validation-only tuning, ablations, the honest-gaps section in ARCHITECTURE.md, and numbers only from JSON files. External replication would be the next step.

### W6. Many settings and garden paths
Many settings tuned on one validation split can overfit that split.
**For SPARC:** settings were chosen on val with seed 0 only. **Action:** confirm the final settings on val seeds 1–2 before the test bench, and report the number of tuned settings.

## Actions taken (2026-11-26)
- **B5 GDumb** baseline added (`baselines.py`).
- **CIFAR-10** loader (restricted unpickler, md5-checked archive) and colour V1 (5×5×3 patches) added.
- Results: see `goal.md` §10 (rows B5 and C1).
