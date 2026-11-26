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

## Result (It 10, val, seed 0) — rejected
| config | MNIST ACC / FGT / FEW5 | Fashion ACC / FGT / FEW5 |
|---|---|---|
| one V1 (It 9) | 94.9 / 1.0 / 49.2 | 81.1 / 10.8 / 71.2 |
| + global V1, w 0.5 | 94.5 / 0.9 / 46.4 | 80.5 / 12.8 / 72.4 |
| + global V1, w 1.0 | 93.9 / 0.8 / 43.4 | 78.4 / 14.1 / 73.6 |
| + global V1, 3×3 grid | 94.8 / 1.0 / 48.0 | 77.1 / 16.3 / 76.0 |

Concatenating a coarse "right hemisphere" into IT does not help MNIST few-shot. It raises Fashion forgetting: pullover, coat and shirt share a silhouette and differ in local detail, so global features make them more alike. The hypothesis "the look-alikes differ in global shape" was wrong for Fashion and gave no gain for MNIST.

What is still open: the brain does not simply concatenate hemispheres. Options are separate IT populations per hemisphere that vote, with a thalamus-like gate that learns which stream separates which classes (file 06), and top-down selection of the useful stream (file 05).

## Status
Rejected as concatenation (It 10). Separate streams with a learned gate remain an option (backlog).
