# 17 — Understanding, thinking and reasoning in the brain

Background for the ultimate goal (goal.md §0) and the stage ladder (docs/roadmap.md). All entries are **[known]** (cited from memory, not yet re-read); upgrade them before they drive a design decision.

## Understanding as prediction (stage 2)
- **Predictive coding**: cortex predicts its input, and lower areas pass up mainly prediction errors. Rao & Ballard 1999; Friston, free-energy principle 2005–2010.
- **Forward models**: the cerebellum predicts the sensory results of actions. Wolpert & Kawato 1998.
- **Sequence memory in cortex**: neurons predict the next element using dendritic context. Hawkins & Ahmad 2016 (HTM; see warnings in research/12).
- **For SPARC**: a temporal predictor on IT codes; prediction error becomes the surprise signal.

## Understanding as structure: cognitive maps (stage 3)
- Tolman 1948: rats learn maps, not just responses.
- Place and grid cells: O'Keefe & Nadel 1978; Hafting, Fyhn, Molden, Moser & Moser 2005.
- The same machinery maps **abstract** spaces (social hierarchies, concepts): Constantinescu, O'Reilly & Behrens 2016; Behrens et al. 2018, "What is a cognitive map?", *Neuron*.
- **Tolman–Eichenbaum Machine**: Whittington et al. 2020 separates structure (entorhinal) from content (hippocampus), so a learned structure transfers to new items.
- **For SPARC**: transitive inference and structure-transfer tasks; a structure code separate from the item code.

## Thinking as working memory (stage 4)
- PFC persistent activity holds goals and items: Goldman-Rakic 1995; Miller & Cohen 2001 (research/05).
- Basal-ganglia gating decides what enters and leaves working memory: PBWM, O'Reilly & Frank 2006 (research/07).
- Activity-silent working memory through short-term synaptic plasticity: Mongillo, Barak & Tsodyks 2008.
- **For SPARC**: a gated PFC-like buffer with fast Hebbian synapses; n-back and 1-2-AX tasks.

## Thinking as simulation (stage 5)
- Hippocampal **replay** of past paths during rest, and **preplay** of future and never-taken paths: Foster & Wilson 2006; Pfeiffer & Foster 2013, *Nature*.
- Model-based vs model-free control: Daw, Niv & Dayan 2005; dopamine reward-prediction error, Schultz, Dayan & Montague 1997.
- **For SPARC**: use the stage-2 predictor to roll out imagined sequences before acting; maze shortcuts.

## Grounded symbols and language (stage 6)
- Multimodal convergence zones: Damasio 1989.
- Grounded cognition: Barsalou 2008.
- **For SPARC**: bind spoken-word codes to visual IT codes in the hippocampus; cross-modal recall.

## Abstract reasoning (stage 7)
- Relational reasoning in rostrolateral PFC: Christoff et al. 2001; Bunge et al. 2005.
- Raven's Progressive Matrices as a fluid-intelligence test; ARC (Chollet 2019, "On the Measure of Intelligence") as few-example abstraction.
- Metacognition: Fleming & Dolan 2012.
- **For SPARC**: composition of stage-3 relations with stage-4 working memory; the rethink loop is a first metacognitive step.

## Caution
The higher stages are where brain-inspired systems have most often overclaimed (research/12, W5). Each stage needs its own benchmark, strong non-brain baselines, and no claim of "understanding" beyond what the tests measure.
