# 11 — Glossary

| Term | Meaning in this project |
|---|---|
| **ACC** | Final 10-way test accuracy after the whole class-incremental stream |
| **ACh / NE** | Acetylcholine / noradrenaline: broadcast chemicals linked to expected and unexpected uncertainty; in SPARC, the surprise and uncertainty signals |
| **Arbitration** | Choosing how much to trust each subsystem (fast vs slow readout) from its confidence and recent errors |
| **ART / vigilance** | Adaptive Resonance Theory: refine a category if the input matches it above the vigilance ρ, otherwise recruit a new one |
| **Callosal integration** | Joining the two hemispheres' signals (through the corpus callosum); in SPARC, IT reads both V1 codes |
| **Class-incremental** | Classes arrive in groups over time, and the model must tell all classes apart at test with no task label |
| **CLS** | Complementary learning systems: fast hippocampus plus slow neocortex, linked by replay |
| **Consolidation (structural)** | Merging redundant IT prototypes when the neuron budget fills (sleep pressure) |
| **Critical period** | A developmental window of high plasticity that then closes; in SPARC, V1 learns for the first 12 000 images |
| **Delta rule** | Single-layer error-driven update Δw = η (target − output) · input; allowed because the error lives at the output neuron |
| **FEW-k** | Accuracy on classes 8/9 after only k labelled examples of each, plus the drop on old classes |
| **FGT** | Forgetting: mean over tasks of (best accuracy ever − final accuracy) on that task's classes |
| **Fly readout** | Only the synapses from active units onto the labelled class change |
| **IT** | Inferotemporal cortex: object-level area; in SPARC, the prototype layer |
| **k-WTA** | k-winners-take-all: only the k most active units fire (lateral inhibition) |
| **Lability** | A memory is changeable only shortly after it was reactivated (reconsolidation window) |
| **LBL5** | Accuracy when only 5 % of the stream carries labels |
| **Match tracking** | ARTMAP: a wrong prediction raises vigilance and forces a new category |
| **Metaplasticity** | A synapse's history sets its learning rate; in SPARC, η = 1/(1 + usage) |
| **NOVELTY** | AUROC of the surprise signal for never-seen classes vs seen ones |
| **P1–P10** | The brain-like learning properties in goal.md §3 (local, online, continual, fast, unsupervised, sparse, fixed resources, novelty-aware, sleep, self-correcting) |
| **Pattern separation** | Making similar inputs use different neurons (dentate gyrus, fly Kenyon cells) |
| **Reconsolidation** | A recalled memory becomes editable again for a while |
| **REVERSAL / NOISE** | Benchmarks for self-correction: swapped labels, and 20 % wrong labels followed by correct ones |
| **Spatial frequency** | Coarse (low: global shape) vs fine (high: local detail) image structure |
| **Three-factor rule** | Δw = η · modulator · pre · post |
| **V1** | Primary visual cortex: local edge and stroke features; in SPARC, a 5×5 patch dictionary |
| **val / test split** | Tuning only on the 6 000-image validation hold-out; test used only for the logged final runs |
