# 19 — Context (latent-cause) inference

## What the brain does
When the same cue predicts different outcomes in different situations, the brain does not average them into one blurred association. It infers **which situation (latent cause) it is in**, and learns and recalls within that context.
- **Latent-cause inference**: a new cause is inferred when prediction errors persist; old memories stay attached to their own cause (Gershman, Blei & Niv 2010, *Psychological Review*; extinction and renewal effects) [known].
- **COIN model**: motor memories are learned and expressed under inferred contexts; this explains savings, spontaneous recovery and why old skills come back (Heald, Lengyel & Wolpert 2021, *Nature*) [known].
- **Hippocampal remapping**: place cells form a new map in a new environment and return to the old map in the old one [known].

## Why SPARC needed it (stage 2)
Sequence worlds share digit pairs with conflicting next items. One predictor either overwrites old worlds (fixed rate: retention 0.27) or learns new ones slowly (metaplastic: online 0.86, retention 0.58). See docs/iteration-log.md S2-1 … S2-3.

## How it works in SPARC (`sparc_seq.CtxSeqSPARC`)
- Up to K = 8 context slots, each a predictor of the next category from the sparse conjunction of current and previous category activity.
- Every step, each slot's prediction is scored: an exponential average (λ = 0.2) of the log-probability it gave the observed category.
- The best-scoring slot is the active context, and only it learns (delta rule, metaplastic rate).
- If the best slot's score stays below a vigilance level (−1.6) after a grace period, a new slot is recruited: vigilance at the level of contexts, as ART does for units.
- At test, the first few items of a sequence select the context (warm-up).

## Result (smoke, 1000 steps per world)
Online 0.902, retention **0.630** (counting n-gram 0.614), violation AUROC 0.839, completion 0.150.

## Links to other stages
- Stage 1's relearning problem (research/15) has the same shape. COIN explains savings by reactivating the old context, not by relearning the old association, which suggests trying context inference for stage-1 classes too.
- Stage 3 (structure) needs the same machinery to keep structure and items apart (cognitive maps).
