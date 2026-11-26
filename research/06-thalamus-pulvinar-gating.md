# 06 — Thalamus and pulvinar: routing between areas

## What the brain does
The thalamus is not only a relay. Higher-order nuclei such as the **pulvinar** form loops with the cortex and control **which cortical areas communicate**, by synchronising their rhythms (alpha band) depending on what is attended. The **thalamic reticular nucleus** inhibits channels that should be ignored. In effect, the thalamus is a switchboard that boosts one route and quiets others.

## Evidence
- Saalmann, Pinsk, Wang, Li & Kastner 2012, *Science*: attention raises pulvinar-to-cortex alpha coherence with V4 and TEO; the pulvinar regulates cortical synchrony [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC3714098.
- Review: the pulvinar is the largest primate thalamic nucleus and forms loops almost only with cortex [abstract] https://www.nature.com/articles/nrn3347.

## How it helps our work
SPARC has several pathways that must be combined: the fast readout, the slow readout, and possibly two hemispheres (file 04). Today the combination is the **arbitration** rule: confidence × (1 − recent uncertainty) × consolidation level. That is a thalamus-like gate between systems, and the ablation shows it matters: without it, MNIST reversal needs 1720 examples per class instead of 30.

## How it might work next
- **Per-input routing between hemispheres**: a gate that weights the global and local V1 streams by how much each agreed with the final decision on recent samples. The rule is local, a Hebbian trace on the gate.
- **Attention as gain on routes**: when the "PFC" state of file 05 holds a conflict, the gate boosts the stream that best separates the conflicting classes.

## Test
Only useful once there is more than one sensory stream (It 10). Compare a learned gate with a fixed `hemi_w`.

## Status
Partly used (readout arbitration). Planned as a stream gate if It 10 is adopted.
