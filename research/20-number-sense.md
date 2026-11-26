# 20 — Number sense: from recognising "3" to understanding addition

User question (2026-11-26): children first look at a 3 and know it's 3, then they can add 3 + anything. Is that what SPARC does?

## What the brain does [known]
- **Approximate number system**: infants and many animals sense approximate quantities before language. Discrimination follows Weber's law (ratio-dependent). Feigenson, Dehaene & Spelke 2004; Nieder & Dehaene 2009.
- **Parietal number line**: the intraparietal sulcus (IPS) holds magnitude codes; neurons tuned to a preferred numerosity, overlapping for nearby numbers (Nieder & Miller 2003). The distance effect (comparing 2 vs 8 is faster than 5 vs 6) reflects this overlap (Moyer & Landauer 1967).
- **Symbol grounding**: children map number words and digits onto this magnitude system gradually, around ages 3–5. Counting principles: one-to-one, stable order, cardinality (Gelman & Gallistel 1978). "Cardinal-principle knowers" understand that the last counted word gives the set size (Wynn 1990).
- **Addition develops by strategy**: counting all, then counting on from the larger addend (the min strategy), then fact retrieval (Siegler & Shrager 1984). Mental arithmetic as movement along the number line, e.g. "operational momentum": addition overshoots (McCrink, Dehaene & Dehaene-Lambertz 2007).
- **Triple-code model**: visual digit form (ventral occipitotemporal cortex), verbal number facts (left perisylvian), magnitude (bilateral IPS) (Dehaene 1992).

## Where SPARC is
- Stage 1 = the visual digit form: recognises "3" (95 % on MNIST), but the digit has no quantity meaning.
- Stage 2 = sequences: the "+3" world is learned as memorised transitions (like reciting "3, 6, 9"), not as an operation on quantities.
- Missing: magnitude (number line), grounding of digits in quantity, and operations that generalise to unseen pairs.

## Plan
Stage-3 number-sense benchmark (docs/roadmap.md): dot arrays ↔ digits (grounding); a number-line code learned from counting order (magnitude comparison on unseen pairs, distance effect); addition with held-out pairs as the test of understanding vs memorising.
