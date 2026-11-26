# 18 — How blind people learn, and what it says about AI "seeing"

User question (2026-11-26): blind people can't see, so maybe they are like an AI that can't actually see. Do they "see"?

## What the research shows

1. **"Visual" cortex is not only for vision.** In people born blind, the occipital ("visual") cortex works for touch (Braille), sound and language. Even sighted adults blindfolded for five days start using visual cortex for Braille, and disrupting it with TMS then impairs their Braille reading [abstract] https://pubmed.ncbi.nlm.nih.gov/18728773/. Expert blind echolocators process click-echoes in calcarine ("visual") cortex, not auditory cortex (Thaler, Arnott & Goodale 2011; only 2 experts) [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC3102086. The idea: brain areas are defined by the **computation** they do (shape, space), not by which sense feeds them (Pascual-Leone & Hamilton 2001, the "metamodal" brain [known; term not verified]).

2. **Object categories form without vision.** The ventral stream (towards IT) shows the same living vs non-living organisation in adults blind since birth (Mahon et al. 2009, *Neuron*) [abstract] https://www.cmu.edu/dietrich/psychology/caoslab/pdf/Mahon%20et%20al._Neuron_2009.pdf. Touching objects activates category patterns in inferior temporal cortex that correlate with seeing them (Pietrini et al. 2004, *PNAS*) [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC397466. The category layout in blind people closely matches sighted people's (van den Hurk et al. 2017) [abstract] https://pubmed.ncbi.nlm.nih.gov/28507127/. Caution: a shared layout does not mean identical content.

3. **Blind people know what things look like, mostly by inference, not by being told.** Congenitally blind and sighted adults order and group animals by size, shape and texture in very similar ways; the biggest gap is colour. The authors argue blind people infer appearance from what kind of thing it is and where it lives, rather than memorising sighted people's descriptions (Kim, Elli & Bedny 2019, *PNAS*) [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC6561279. In a reply, blind and sighted people shared more with each other than either did with a text-statistics (distributional) model. Critics questioned the inference account (Ostarek et al. 2019 letter).

4. **Knowing a shape by touch does not instantly give you its look.** Molyneux's question (1688): if a blind person who knows a cube by touch gains sight, will they recognise it by eye? Children given sight by cataract surgery (Project Prakash) could **not** match felt shapes to seen ones right away, but learned to within days (Held et al. 2011, *Nature Neuroscience*) [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC3238323; https://news.mit.edu/2011/vision-problem-0411.

5. **"Seeing" through another sense can be learned, slowly.** Sensory-substitution devices turn camera images into sound (vOICe) or tongue stimulation (BrainPort). Trained blind users perceive light, simple objects, motion and short words, and visual cortex activates. But they need extensive training, and short training is not enough [abstract] https://pmc.ncbi.nlm.nih.gov/articles/PMC7381305; https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01443/full.

## So, is a blind person "like an AI that can't see"?
Partly, and the differences are the interesting part.
- **Like a text-only LLM:** both learn a lot about the visual world without visual input. Blind people's knowledge of appearance is rich and mostly right, except colour.
- **Unlike a text-only LLM:** blind people are not learning from words alone. Touch, sound, movement and space ground their concepts, and they **infer** appearance from causal knowledge (what an animal is, how it lives). In Bedny's data, blind people share more with sighted people than with a model built from text statistics. So "understanding without seeing" is real, but it comes from grounded, multi-sense experience plus inference, not from language patterns alone.
- **Do they "see"?** Not in the ordinary sense: they have no visual experience. But their brains build **sense-independent (supramodal) representations** of shape, space and category, often in the same areas sighted people use for vision. In that sense they "know what it looks like" without seeing it.
- **And SPARC?** SPARC receives pixels, so it has a crude retina. But it has no 3D world, no touch and no actions, and since round 2 only small eye movements. Its "seeing" is narrower than a blind person's grounded knowledge, in a different way.

## What this means for our project
- **IT should be supramodal.** Object codes should not belong to one sense. Stage 6 (grounded symbols) should bind several senses to the same IT and hippocampal codes, as the blind brain does.
- **Understanding does not require one specific sense**, but it does require grounding plus inference. That supports the stage ladder: perception and memory, then prediction (inference about what follows), then structure (cognitive maps).
- **A ready-made test, the "Molyneux test"**: train SPARC on digits in one modality (images), then present the same digits in a second modality (e.g. a coarse "tactile" rendering or spoken digits) and measure (a) zero-shot cross-modal recognition (expected: low, like the Prakash children) and (b) how fast cross-modal binding is learned (expected: fast, within a few examples). That would be a stage-6 benchmark with a clear human comparison.
- **Cross-modal plasticity**: an area deprived of its usual input should be able to take on another sense's job. In SPARC that means V1/IT learning rules that work on any structured input, which our local rules already do.

## Status
Research note only. "Molyneux test" added to the stage-6 plan (docs/roadmap.md).
