# 02 — Sparse codes and the fruit-fly mushroom body

## What the brain does
In the fly's olfactory system, about 50 input channels project randomly onto about 2000 Kenyon cells (KCs). Inhibition lets only about 5 % fire for any odour, so different odours use different cells (**pattern separation**). Learning happens only at synapses from the active KCs onto output neurons, gated by dopamine, so memories of other odours are not touched. The mammalian dentate gyrus does a similar sparse expansion before the hippocampus stores a memory.

## Evidence
- Shen, Dasgupta & Navlakha 2021: sparse expansion plus updating only the active synapses reduces catastrophic forgetting [abstract] https://arxiv.org/abs/2107.07617.
- Zou, Zang & Ji 2025: a plug-in "Fly Model" built on these structural features [abstract] https://arxiv.org/abs/2502.01427.
- HiCL uses dentate-gyrus-style top-k sparsity for pattern separation [abstract] https://arxiv.org/abs/2508.16651.

## How it helps our work
SPARC's **fast readout** is the fly rule: a labelled sample changes only row c, toward the code of the active units. Together with 0.5 % active IT units, this is why MNIST forgetting is 0.9 points against 15.0 for ER-500 replay.

## What we tried
**It 8: fixed random V1 → 4000-KC expansion, top 200 firing, fly readout.** It did not improve MNIST few-shot (47.1 % vs 49.2 %). Used alone, the KC readout had almost no old-class drop (0.35 pts) but the same new-class accuracy. Rejected; the flag stays off. Lesson: a random expansion of a weak feature code cannot add information the code doesn't have.

## How it might work next
- Use the KC pathway as a **novelty-insensitive memory for few-shot classes only**. Its near-zero old-class drop is valuable; it could be combined with CLS deferral (file 03).
- Make expansion sparsity adaptive per input (a global inhibitory neuron, like the fly's APL neuron) instead of a fixed top-k.

## Test
FEW-5 old-class drop (target ≤ 2) and FEW-5 accuracy, both datasets.

## Status
Used: sparse IT code and fly readout. Rejected: KC pathway as a third vote (It 8). Possible: KC readout for few-shot classes.
