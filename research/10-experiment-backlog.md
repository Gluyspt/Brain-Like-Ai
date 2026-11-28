# 10 — Experiment backlog (triage under the research → test rule)

Rule (docs/rules.md, user directive 2026-11-27): every research idea worth trying gets tested, unless (1) more important current-stage work comes first, in which case it is **queued** in priority order, or (2) it belongs to a later stage, in which case it is **parked** with a stage tag. Every idea has a status and a reason. Results go back into the research note and docs/iteration-log.md.

Status values: **running**, **queued** (with priority), **tested** (result), **adopted**, **dormant** (flag kept off; partner it probably needs), **rejected** (broken, not brain-like, or harmful even with its partner), **parked → stage N**.

## Combination round (end of each round)
All adopted + dormant flags on, then leave-one-out. Next one: when stage 1 adopts its next change. Fair re-test queued: **early LTP + selective capture of replayed hippocampal traces during sleep** (after R4-4).

## Running
| Idea | From | Note |
|---|---|---|

## Queued: current stages 1–3, in priority order
| P | Idea | Brain mechanism | Target | From |
|---|---|---|---|---|
| 1 | **Delayed relearning check** in LIFE (relearning judged after sleep and after a further learning phase) | savings, desirable difficulties | benchmark validity for relearning | 22 |
| 2 | **Context inference for stage-1 memory**: a reminder reactivates the old class's context | latent causes, COIN | relearning, few-shot drop | 19, 15 |
| 3 | **Adaptive teacher vs curious learner vs random reminders** (equal budget) | tutoring / expertise reversal; curiosity = learning-progress reward | retention, relearning | 23 |
| 4 | **Structure → savings**: do items embedded in a structure (maps) relearn faster than isolated ones? | levels of processing, elaboration | understanding vs memorising | 22, 17 |
| 5 | **LBL0**: form categories with zero labels, give names only at the end | unsupervised category learning (P5) | how unsupervised stage 1 really is | (2026-11-26 discussion) |
| 6 | **Prioritised replay** of currently confused classes in sleep | hippocampal replay prioritisation | Fashion FGT, few-shot drop | 03 |
| 7 | **IT-level sleep replay** (re-align prototypes and readouts) | sleep replay, synaptic homeostasis | Fashion FGT | 03 |
| 8 | **Self-testing in sleep** (predict and check stored traces, like self-explanation) | retrieval practice, self-explanation | retention | 23 |
| 8b | **BTSP-like one-shot binding** for few-shot classes (plateau-gated, seconds-scale eligibility, bidirectional) | BTSP | few-shot, old-class drop | 24 |
| 8c | **Behavioural tagging**: keep hippocampal traces of weakly learned items when a surprising event occurs nearby | tag and capture, dopamine | few-shot retention | 24 |
| 9 | **KC readout reserved for few-shot classes** (near-zero old-class drop in It 8) | fly mushroom body | few-shot old-class drop | 02 |
| 10 | **Learned hippocampal write gate** (value of a trace = did replaying it reduce errors) | basal-ganglia gating, dopamine | memory efficiency, old-class drop | 07 |
| 11 | **Separate ACh / NE modulators** (expected vs unexpected uncertainty) | Yu & Dayan | reversal speed, stability | 01 |
| 12 | **Graded critical period** (decaying V1 plasticity, reopened by strong novelty) | adult plasticity | Fashion FGT, CIFAR | 01 |
| 13 | **Rethink loop during learning** (second look also when a labelled sample is a near-tie) | ACC/PFC | few-shot, look-alikes | 05 |
| 14 | **Predictive-coding feedback IT → V1** (during the V1 critical period) | predictive coding | features, CIFAR | 05 |
| 15 | **Use-dependent decay of hippocampal episodes** (details fade, categories stay) | active forgetting | memory efficiency | 15 |
| 16 | **BCM sliding threshold** instead of k-WTA in V1 | BCM metaplasticity | locality of competition | 01 |
| 17 | **Adaptive global inhibition** for the KC expansion (fly APL neuron) | fly mushroom body | few-shot | 02 |
| 18 | **Separate hemisphere IT populations + learned gate** (concatenation was rejected in It 10) | hemispheric specialisation, thalamic gating | look-alikes | 04, 06 |

## Parked: later stages
| Stage | Idea | From |
|---|---|---|
| 4 | Gated working-memory buffer (PFC + basal ganglia), n-back / 1-2-AX; **learn the counting-on routine** instead of building it in | 17, 07, 20 |
| 4–5 | **Interrogation loop**: a reasoning system asks memory focused sub-questions over several steps (as in MeMo) | 21 |
| 4 | CA3-style pattern completion from partial codes | 03 |
| 5 | **Reward instead of labels** (dopamine RPE as the modulator); curiosity-driven exploration | 07, 23 |
| 5 | Thinking as simulation: roll out the stage-2 predictor to plan (mazes, shortcuts) | 17 |
| 2–4 | **Memory linking in time** (events close in time share neurons via transient excitability) | 24 |
| 5 | **REM-like recombination phase** in sleep (insight) | 24 |
| 6 | **Molyneux test** (cross-modal recognition and binding); supramodal IT; hearing front-end | 18 |
| 6 | Stage-6 comparison: continual document learning, SPARC-style memory vs MeMo vs RAG | 21 |
| 6 / natural images | Dorsal-like stream (location, size invariance) | 04 |
| scale | Feedback alignment into V2, only with old-class protection (the IT version distorted old classes) | 13 |
| 7 | Raven's- / ARC-style abstract reasoning | 17 |

## Tested (results in docs/iteration-log.md)
| Idea | Result | From |
|---|---|---|
| Event boundaries removed (continuous stream); surprise self-segmentation | **adopted** stream (S2-8/9/F2: test online 0.924, all stage-2 criteria pass); self-segmentation rejected (reset kills context recruitment; binding-only over-segments) | 24 |
| Spacing effect test | tested (R4-3): none in SPARC (massed ≈ spaced), it does not forget between sessions | 2026-11-27 |
| Early / late LTP with capture by sleep or surprise | **dormant** (R4-3): no benefit alone (without capture Fashion FGT 14.0); partner: **selective capture of what the hippocampus replays in sleep** | 24 |
| SHY down-scaling sleep (`ie_sleep=down`) | **candidate** (R4-2/4, 3 seeds): relearning → savings, MNIST FEW5 41.2 → 69.6, Fashion FGT 6.9 → 5.2; old-class drop 1.9 → 5.5 (MNIST), 2.3 → 3.7 (Fashion); needs a partner protecting old classes; partner for sleep found (`merge_protect=used`, R4-8), old-class partner still open | 01, 22, 24 |
| Slow recovery of excitability across nights (`ie_rec`) | **dormant** (R4-4): trades few-shot for old-class drop, no net gain | 01 |
| Outcome-error hippocampal writes (`hip_err`), recency-first replay (`rep_recent`) | **dormant** (R4-6 2×2): sleep still loses 6–14 pts vs wake; partner unknown | 03, 24 |
| Fresh memories not merged at the next sleep (`merge_protect`) | **candidate** (R4-7, R4-8 2×2): removes the sleep loss alone; with SHY the apparent sleep > wake was a task confound (R4-8b: on task (2, 3) sleep − wake −3 to −4.5); ACC F 83.5 / M 96.3; old-class drop 4.6 / 4.9 fails → clean 2×2 in R4-8c | 2026-11-28 diagnosis |
| Schema assimilation (new item placed into an established map in one shot) | **adopted** (S3-5): top-insert novel pairs 1.00 after 1 fact (without 0.62, MLP 0.53–0.82) | 24 |
| Sleep-time homeostasis (ie_sleep) / slow homeostasis (ie_tau 20000) | tested (R4-1): see R4-2 | 22, 24 |
| Top-down conflict bias + second look (rethink loop) | **adopted** (It 12): old-class drop down, Fashion FGT down | 05 |
| Eye movements + trace rule | **adopted** (R2): Fashion FGT 9.5 → 6.3 | 14 |
| Label-gated refinement | **adopted** (R3): small val gain, not on test | 13 |
| V1 critical-period closure | **adopted** (It 9) | 01 |
| V2 layer | tested (R2-1): helps MNIST few-shot / Fashion, not CIFAR; not adopted | 09 |
| Error-driven IT tuning (LVQ-like) | **rejected** (R2-2a): reverses over long streams | 13 |
| Two-hemisphere V1, concatenated | **dormant** (It 10); partner: learned hemispheric gate (P18) | 04 |
| KC pathway as a third vote | **dormant** (It 8); partner: reserved for few-shot classes (P9) | 02 |
| Episodic recall at answer time (ungated / gated / margin) | tested (R2-5, R2-6): trade-off curve; not adopted | 16 |
| Retrieval-triggered reconsolidation | tested (R2-4): partial relearning gain, hurts main stream | 15 |
| Engram tags (silent engrams) | **dormant** (R3-1): no effect alone; partner: context inference for stage 1 (P2) | 15 |
| Per-synapse metaplasticity | **dormant** (R3-2): halves MNIST drop alone, raised Fashion FGT with label gate; partner: retrieval reopening / context inference | 15 |
| Temporal predictor + context inference | **adopted** as stage 2 | 08, 19 |
| Number sense, transitive inference, list linking, same/different | **adopted** as stage 3 | 17, 20 |
| GDumb baseline, CIFAR-10 check | done (B5, C1) | 12 |
| Dense IT code (k 30/50), turnover | rejected (It 6, It 2) | – |
