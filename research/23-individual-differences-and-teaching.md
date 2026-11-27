# 23 — People learn differently: what the evidence says about teaching

User question (2026-11-27): truly understanding something is what keeps it long-term, but not every human learns the same way; some need more explanation, some something else.

## Evidence [abstract unless noted]
- **"Learning styles" (visual / auditory / kinesthetic, with matched teaching) is a myth.** The review found virtually no evidence for the matching ("meshing") hypothesis, and some sound studies contradicted it: Pashler, McDaniel, Rohrer & Bjork 2008, https://pubmed.ncbi.nlm.nih.gov/26162104/
- **What does differ: the learner's current state, especially prior knowledge.** Expertise-reversal effect: worked examples help novices, but become redundant for learners with more knowledge, for whom problem solving works better (Kalyuga et al. 2003; review https://link.springer.com/article/10.1007%2Fs10648-007-9054-3; https://en.wikipedia.org/wiki/Expertise_reversal_effect). So "some need more explanation" is true, by knowledge state, not by fixed type.
- **Adaptive one-to-one teaching helps, but less than the legend.** Bloom's "2 sigma" (1984, https://Www.Gwern.net/doc/psychology/1984-bloom.pdf) was a small, unreplicated study; meta-analyses find tutoring ≈ 0.33–0.37 SD (https://www.educationnext.org/two-sigma-tutoring-separating-science-fiction-from-science-fact/).
- **Self-explanation builds understanding.** Prompting students to explain each step improved recall and transfer vs re-reading (Chi et al. 1994, https://pcl.sitehost.iu.edu/rgoldsto/courses/cogscilearning/chiselfexplanation.pdf); it can be taught and works broadly.
- **Learners steer themselves by learning progress.** Curiosity as intrinsic reward for learning progress; humans choose tasks where they improve fastest; midbrain/dopamine involvement (Oudeyer, Gottlieb & Lopes 2016, https://www.pyoudeyer.com/oudeyerGottliebLopesPBR16.pdf; Ten et al. 2021, https://pmc.ncbi.nlm.nih.gov/articles/PMC8514490).
- **Machines: curriculum learning** (easy → hard; Bengio et al. 2009) can help, but the best order depends on the task, sometimes hard → easy (survey https://arxiv.org/pdf/2101.10382).

## Implications for SPARC
SPARC has one fixed teacher (a fixed stream, random reminders). Two brain-like additions:
1. **Adaptive teacher (world layer, docs/rules.md):** the benchmark watches what SPARC gets wrong and gives targeted reminders / more examples there. Test in LIFE: targeted vs random reminders at equal budget.
2. **Curious learner (in the model):** SPARC requests examples where its learning progress (or uncertainty) is highest, via a dopamine-like learning-progress signal. This is learning how to learn, linked to stage 5 (reward).
3. **Self-explanation analogue:** sleep replay is like self-testing; a version that predicts and checks itself during sleep would be closer to explaining to oneself.
