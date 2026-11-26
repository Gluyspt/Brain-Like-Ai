# 04 — Hemispheres and visual streams

## What the brain does
The two halves of the brain are not "two personalities". Some functions are lateralised **locally**: language is mostly left, and parts of attention control are mostly right.

In vision, there is a consistent bias:
- the **right hemisphere** is better at **global shape**, carried by low spatial frequencies (the coarse layout);
- the **left hemisphere** is better at **local detail**, carried by high spatial frequencies (fine strokes).

The two sides are joined by the corpus callosum, and object areas (IT) receive from both. Separately, visual cortex splits into a **ventral stream** ("what": object identity, towards IT) and a **dorsal stream** ("where/how": location, motion, action).

## Evidence
- **Myth check.** Nielsen et al. 2013, 1,011 resting-state scans: lateralisation is a property of local networks; people are not "left-brained" or "right-brained" [abstract] https://pubmed.ncbi.nlm.nih.gov/23967180/; press release https://healthcare.utah.edu/press-releases/2013/08/researchers-debunk-myth-of-right-brain-and-left-brainpersonality-traits.
- **Global/local.** Robertson & Ivry 2000: hemispheric asymmetry for global (right) vs local (left), with lesion evidence [abstract] https://ivrylab.berkeley.edu/files/organized_pubs_pdfs/2000_robertson_ivrycurrent_dir.pdf; review [abstract] https://ane.pl/index.php/ane/article/download/1151/1151. Mechanism: the spatial-frequency hypothesis, Sergent 1982 [known]. The effect depends on stimulus type and is weaker in left-handers [abstract].
- **Two visual streams.** Ungerleider & Mishkin 1982; Goodale & Milner 1992 [known].

## How it helps our work
SPARC's worst gap is **MNIST few-shot (43.7 %)**. The new classes 8 and 9 look like 3, 5, 4 and 7. They differ mainly in **global shape**: closed loops, overall layout. SPARC's only V1 sees fine 5×5 patches pooled on a 4×4 grid, a "left-hemisphere-only" visual system. On Fashion, the look-alike classes (pullover, coat, shirt) also differ mostly in coarse silhouette.

## How it might work in SPARC (It 10)
- A second V1, the **"right hemisphere"**: the same competitive-Hebbian rule, run on a 2×2-averaged 14×14 image (low spatial frequency), pooled on a coarse 2×2 grid (global layout). 256 extra features.
- **Callosal integration**: IT receives both codes, each normalised, the global one weighted by `hemi_w`.
- Both V1s share the critical period (they close after 12 000 images).
- Code: `v1.py` (`lowsf=True`), `sparc.py` (`hemi`, `hemi_grid`, `hemi_w`).

Later ideas:
- Give each hemisphere its **own IT population** and readout, and let them vote (two partly independent experts, joined at the decision).
- A small **dorsal-like stream** (location and size invariance) for real images.

## Test
FEW-5 on MNIST (target 70), Fashion FGT and ACC, and an ablation with the hemisphere turned off.

## Status
It 10 grid running (val, seed 0): hemi_w 1.0 / 0.5, global grid 2 / 3. Results go in `goal.md` §10.
