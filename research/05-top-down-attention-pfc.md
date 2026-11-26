# 05 — Top-down attention and the prefrontal cortex

## What the brain does
The prefrontal cortex (PFC) keeps the current goal active. From there it sends **bias signals** back to sensory areas, so neurons coding task-relevant features win the competition against stronger but irrelevant ones. One area "boosts" another this way.

More generally, cortex is full of **feedback**: higher areas send predictions down, and lower areas pass up mainly what was not predicted (predictive coding).

## Evidence
- Miller & Cohen 2001, integrative theory of PFC function: actively maintained goal patterns bias processing elsewhere [abstract] https://pubmed.ncbi.nlm.nih.gov/11283309/.
- Gazzaley et al. 2007: PFC adjusts the coupling with visual association cortex according to stimulus relevance [abstract] https://sites.lifesci.ucla.edu/psych-rissmanlab/wp-content/uploads/sites/70/2021/02/Gazzaley_Rissman_CerebCtx_2007.pdf.
- Caveat: reviews note the PFC's role in visual attention is not fully settled, and most work studies spatial attention [abstract] https://www.montana.edu/attentionlab/documents/5_2013_ARN.pdf.
- Predictive coding in visual cortex. Rao & Ballard 1999, *Nat Neurosci* [known].
- ART's top-down expectation and match check is a form of feedback. Grossberg [abstract] https://doi.org/10.4249/scholarpedia.1569.

## How it helps our work
SPARC is **feed-forward**: nothing above IT changes which IT units win. Few-shot and look-alike errors happen because the features that separate a new class from an old look-alike are weak compared with the shared ones. A top-down bias that strengthens **diagnostic features** for the classes currently in conflict is the brain's way to fix exactly that.

## How it might work in SPARC (It 11 candidate)
- A small **"PFC" state** holding the classes involved in recent confident errors. SPARC already tracks this in the per-class uncertainty trace U.
- **Top-down bias on IT competition**: for each class pair in conflict, units whose readout weights differ most between the two classes get a gain boost when choosing the k winners. The gain is the bias signal: a broadcast per class, multiplied by the unit's own readout synapses, so it stays local.
- **Feedback at test time, too**: after a first pass, take the top two candidate classes, bias IT toward their diagnostic units, and decide on a second pass. This is a two-step recognition, like a "second look".
- Optional **predictive coding**: IT sends its prototype down to V1, and the error drives V1 learning during its critical period.

## Test
MNIST FEW-5 and the old-class drop; Fashion ACC; wall-clock, since a second pass costs about 2×.

## Status
Planned.
