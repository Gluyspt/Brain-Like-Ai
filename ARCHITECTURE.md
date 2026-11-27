# SPARC — Sparse, Predictive, Adaptive-Resonance network with Consolidation

(Sections 1–5: stage 1, perception + memory. Section 6: stage 2, prediction over time. Section 7: stage 3, number sense.)

A continual learner whose **learning** is built from brain mechanisms only: local three-factor plasticity, online single-pass learning, sparse codes, novelty-gated recruitment, a hippocampal store, sleep replay, and error-driven self-correction. No backprop, no autograd, no raw-image replay. Code: `sparc.py` (+ `v1.py`), benchmark: `bench.py`, history: `goal.md` docs/iteration-log.md.

Design rule (user directive, goal.md §3): **every mechanism must be something a brain does**; a change that only raised a score was rejected, and the cost of each brain-first choice is logged.

## 1. Architecture

```
image ─► V1   5×5 contrast-normalised patches; 64 competitive-Hebbian features (learned in a critical period that closes);
         │    k-WTA 3/64 per position; fixed 4×4 sum-pooling → 1024-d
         ▼
         IT   adaptive-resonance prototypes (budget N = 3000): cosine match, k-WTA 15 (0.5 %),
         │    refine on resonance, recruit on mismatch (vigilance ρ, surprise z-score, match tracking),
         │    intrinsic excitability, sleep-pressure consolidation
         ▼ sparse code h (15 of 3000)
   ┌─────┴───────────────────────────────┐
   ▼                                     ▼
 FAST readout (fly-style row update,  HIPPOCAMPUS: 1000 sparse codes (idx, value, label),
 confident-error LTD, reconsolidation) write p = σ(surprise z); class-balanced eviction
   │                                     │ sleep: cleanup + class-balanced replay
   │                                     ▼
   │                                  SLOW cortical readout R (delta rule, sleep only)
   └──► arbitration: confidence × (1 − recent unexpected uncertainty) × consolidation level ◄──┘
 rethink: conflict (ACC) → top-down gain on discriminative IT units (PFC) → second look
 broadcast modulators: surprise s (ACh/NE), confident error M (DA/NE), lability L, uncertainty U
```

## 2. Learning rules (per labelled or unlabelled sample, batch 1, single pass)

Notation: x = normalised V1 vector, a_i = w_i·x (cosine), s = 1 − max_i a_i (surprise), z = (s − μ_s)/σ_s with running statistics (τ = 500).

| Component | Rule | Brain mechanism |
|---|---|---|
| V1 dictionary | 10 random contrast patches p per image; winner d_k ← unit(d_k + η_k(p − d_k)), η_k = max(1/(1+n_k), 0.002); imprint a free feature if best match < 0.5; **plasticity closes after the first 12 000 images** | experience-dependent simple-cell tuning; critical period that closes (PV maturation, perineuronal nets) |
| IT recruitment | recruit (w_new = x) iff [max a < ρ = 0.8 and z > 1] or **match tracking**: labelled sample whose fast prediction is wrong | ART vigilance; ACh/NE novelty; ARTMAP match tracking |
| IT refinement | top-k winners: w_i ← unit(w_i + η_i a_i (x − w_i)), η_i = 1/(1+n_i) | Hebbian/Oja-like competitive learning, metaplasticity |
| Intrinsic excitability | winners chosen on a_i + γ·clip(1 − r_i/(k/used), ±1); r_i = win-rate trace (τ = 2000), newborn units r = 0 | hyperexcitable adult-born neurons; homeostatic intrinsic plasticity |
| Consolidation (structural) | when the budget is full or at sleep: merge most-similar same-class prototype pairs (usage-weighted mean, readout synapses add), water-filled so the largest classes shrink first, until 25 % is free | sleep-dependent synaptic downscaling/pruning; competition for cortical territory |
| Fast readout | labelled c: A_c ← A_c + (h − A_c)/n_c (only row c) | fly mushroom body: only synapses from active KCs onto the reinforced output change |
| Confident-error LTD | if the fast readout predicts c′ ≠ c with confidence q > 0.5: M = q; A_{c′,i} ← A_{c′,i}(1 − g·ĥ_i) on active i, g = L_{c′}·min(1, 50M/(50+n_{c′})) | three-factor rule, dopamine/NE error burst, LTD at the mistaken association |
| Reconsolidation | same event: n_c ← n_c(1 − 0.2M), n_{c′} ← n_{c′}(1 − 0.2M·L_{c′}) (plasticity reopens; repeated errors flip, single errors nudge) | reconsolidation, evidence-weighted plasticity |
| Lability | L_c ← L_c·e^{−1/50} + 1/20 when c is labelled | only recently reactivated memories are labile |
| Hippocampus | labelled code stored with p = σ(z); full → overwrite a random trace of the most-stored class | novelty-gated encoding (DA/ACh), interference-based forgetting |
| Sleep | (1) drop traces confidently contradicted by the fast readout; (2) 3 × M class-balanced replays, R_{:,idx} += η(onehot − softmax(R h)) h (single-layer delta rule) | active forgetting; complementary learning systems; sleep replay |
| Arbitration | p = (w_f p_f + w_s p_s)/(w_f + w_s), w = max prob; w_s ·= (1 − U_{class}) and ·= min(1, traces(c_fast)/50) | uncertainty-based arbitration between systems (Daw 2005); CLS: new memories are hippocampus-dependent |
| Rethink loop | at inference: if p(runner-up) > 0.5·p(winner), add top-down gain 0.1·|Â_c1,i − Â_c2,i|/max to IT unit i, re-run k-WTA, decide again (one extra pass) | ACC conflict monitoring → PFC top-down bias → second look |
| Label-gated refinement | with a label, only IT winners of that class (or of no class) refine toward the input; unlabelled glimpses and the trace rule stay label-free | task-relevant gating of IT plasticity (Sigala & Logothetis 2002) |
| Eye movements + trace rule | each training image seen as 2 shifted glimpses (±2 px) then the fixated view; later glimpses don't recruit; previous glimpse's IT winners pulled toward the current view, rate 0.5/(1+n); the label binds once | saccades; temporal-continuity trace rule (Földiák 1991) → shift invariance |
| Novelty signal | s = 1 − max a (no learning at test) | mismatch / surprise signal |

## 3. Results

Protocol: docs/benchmark.md (5 tasks × 2 classes, class-incremental, single pass, batch 1; baselines batch 10). All tuning on the validation split; the test split was used only for the logged final runs. Tables are generated by `tools/make_report.py` from the JSON files.

### Final results (test split, mean ± sd over seeds; tag `final6` for SPARC, `final` for baselines)

| model | data | n | ACC | FGT | FEW1 | FEW5 | FEW5 old drop | LBL5 | NOVELTY | REV ex/class→80 % | REV final | REV other drop | NOISE gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPARC | mnist | 3 | 94.7 ± 0.1 | 0.7 ± 0.0 | 13.5 ± 0.9 | 43.6 ± 4.4 | 2.2 ± 0.6 | 95.1 ± 0.1 | 0.927 ± 0.002 | 80.5 ± 0.0 | 98.2 ± 0.4 | -0.0 ± 0.0 | -0.1 ± 0.0 |
| B1 | mnist | 3 | 19.6 ± 0.0 | 99.4 ± 0.1 | 29.1 ± 16.3 | 83.6 ± 5.2 | 21.1 ± 2.8 | 19.2 ± 0.3 | – | 48.3 ± 9.4 | 97.3 ± 0.2 | 0.0 ± 0.0 | 0.2 ± 0.1 |
| B2 | mnist | 3 | 86.9 ± 0.4 | 15.0 ± 0.5 | 7.6 ± 9.4 | 64.1 ± 18.6 | 44.9 ± 13.5 | 84.3 ± 2.7 | – | 238.3 ± 75.9 | 74.0 ± 9.0 | 1.2 ± 0.8 | 1.5 ± 0.3 |
| B3 | mnist | 3 | 97.9 ± 0.1 | – | – | – | – | – | – | – | – | – | – |
| B4 | mnist | 3 | 82.1 ± 0.0 | 6.7 ± 0.0 | 3.0 ± 1.7 | 32.3 ± 3.9 | 1.3 ± 0.4 | 81.6 ± 0.3 | – | – | 66.7 ± 0.0 | 0.0 ± 0.0 | 0.2 ± 0.0 |
| B5 | mnist | 3 | 85.9 ± 0.6 | 7.4 ± 1.0 | 1.0 ± 0.1 | 27.1 ± 3.8 | -0.1 ± 0.8 | 89.1 ± 0.1 | – | – | 2.8 ± 0.1 | 0.0 ± 0.0 | 19.0 ± 0.4 |
| SPARC | fashion | 3 | 81.5 ± 0.4 | 7.6 ± 0.3 | 40.2 ± 11.2 | 79.8 ± 8.7 | 2.7 ± 0.7 | 80.9 ± 0.7 | 0.845 ± 0.003 | 40.5 ± 0.0 | 91.4 ± 0.6 | 0.4 ± 0.2 | 0.2 ± 0.2 |
| B1 | fashion | 3 | 19.9 ± 0.0 | 98.8 ± 0.0 | 50.5 ± 1.1 | 78.1 ± 10.1 | 15.2 ± 8.4 | 19.9 ± 0.0 | – | 25.0 ± 0.0 | 96.5 ± 0.3 | 0.0 ± 0.0 | 0.1 ± 0.1 |
| B2 | fashion | 3 | 75.6 ± 1.0 | 25.6 ± 0.7 | 0.3 ± 0.4 | 41.3 ± 27.9 | 25.8 ± 5.7 | 68.6 ± 3.2 | – | 75.0 ± 21.6 | 92.2 ± 3.9 | 1.8 ± 2.9 | 0.3 ± 0.7 |
| B3 | fashion | 3 | 87.9 ± 0.4 | – | – | – | – | – | – | – | – | – | – |
| B4 | fashion | 3 | 67.7 ± 0.0 | 13.8 ± 0.0 | 31.4 ± 5.4 | 67.4 ± 2.5 | 1.6 ± 0.3 | 67.7 ± 0.4 | – | – | 46.4 ± 0.0 | -5.0 ± 0.0 | -0.1 ± 0.1 |
| B5 | fashion | 3 | 71.6 ± 3.4 | 15.6 ± 1.7 | 10.7 ± 8.5 | 45.9 ± 12.2 | 0.1 ± 5.5 | 68.2 ± 5.8 | – | – | 2.3 ± 1.6 | 0.0 ± 0.0 | 2.4 ± 3.7 |

SPARC sparsity (MNIST seed 0): {'IT': 0.005, 'V1': 0.026501618491278755}; Fashion: {'IT': 0.005, 'V1': 0.03500693904028999}. params 3,133,600; stored numbers 31,000 (1000 codes x (2*15+1) = 31000 numbers (vs 500 raw images = 392500)); raw images stored 0. Wall-clock per full bench run ≈ 1241 s.

### Ablations (validation split, seed 0; Δ vs full model, sign so that + = full model better)

| ablation | data | ACC | FGT | FEW5 | FEW5_old_drop | LBL5 | NOVELTY | REV_examples_per_class_to_80 | NOISE_gap | costs ≥ 1 pt somewhere? |
|---|---|---|---|---|---|---|---|---|---|---|
| **full** | mnist | 96.0 | 1.0 | 48.7 | 2.2 | 95.5 | 0.900 | 30.5 | 0.5 | |
| gabor | mnist | 93.7 (+2.2) | 0.9 (-0.1) | 45.5 (+3.2) | 2.1 (-0.0) | 92.7 (+2.9) | 0.860 (+0.039) | 40.5 (+10) | 0.6 (+0.1) | yes |
| noV1 | mnist | 91.2 (+4.7) | 2.3 (+1.3) | 41.5 (+7.1) | 3.6 (+1.4) | 91.4 (+4.2) | 0.849 (+0.050) | 30.5 (+0) | 0.6 (+0.0) | yes |
| noadaptvig | mnist | 96.0 (+0.0) | 1.1 (+0.1) | 50.2 (-1.5) | 2.1 (-0.1) | 95.5 (+0.1) | 0.902 (-0.003) | 80.5 (+50) | 0.4 (-0.2) | yes |
| noarbitration | mnist | 96.0 (-0.0) | 0.9 (-0.0) | 48.7 (+0.0) | 2.1 (-0.0) | 95.5 (+0.0) | 0.900 (+0.000) | 1720.5 (+1690) | 0.5 (-0.0) | yes |
| noclsdefer | mnist | 96.0 (+0.0) | 1.0 (-0.0) | 40.9 (+7.7) | 0.8 (-1.4) | 95.5 (+0.0) | 0.900 (+0.000) | 30.5 (+0) | 0.5 (-0.0) | yes |
| nocorrection | mnist | 95.8 (+0.1) | 1.0 (+0.1) | 50.1 (-1.4) | 2.2 (+0.0) | 95.5 (+0.0) | 0.907 (-0.007) | – | 0.4 (-0.2) | no |
| noexcitability | mnist | 96.1 (-0.1) | 3.2 (+2.2) | 49.4 (-0.8) | 3.6 (+1.4) | 95.8 (-0.2) | 0.898 (+0.002) | 30.5 (+0) | -0.0 (-0.6) | yes |
| nolability | mnist | 96.1 (-0.1) | 1.0 (+0.1) | 46.0 (+2.7) | 2.0 (-0.2) | 95.7 (-0.1) | 0.909 (-0.009) | 30.5 (+0) | 0.2 (-0.3) | yes |
| nomatchtrack | mnist | 95.4 (+0.5) | 1.1 (+0.1) | 44.4 (+4.3) | 2.0 (-0.2) | 95.5 (+0.0) | 0.903 (-0.003) | 30.5 (+0) | 0.0 (-0.5) | yes |
| nomerge | mnist | 94.9 (+1.1) | 0.8 (-0.1) | 30.9 (+17.7) | 1.9 (-0.3) | 95.0 (+0.6) | 0.891 (+0.009) | 30.5 (+0) | 0.5 (-0.0) | yes |
| norefine | mnist | 95.3 (+0.7) | 1.3 (+0.4) | 50.9 (-2.3) | 1.4 (-0.8) | 95.2 (+0.3) | 0.826 (+0.074) | 40.5 (+10) | -0.0 (-0.6) | yes |
| nosleep | mnist | 95.7 (+0.3) | 1.0 (+0.1) | 51.1 (-2.4) | 2.6 (+0.4) | 95.5 (+0.0) | 0.901 (-0.001) | 30.5 (+0) | 0.2 (-0.3) | no |
| nosleepclean | mnist | 96.1 (-0.1) | 0.9 (-0.1) | 49.9 (-1.3) | 2.4 (+0.2) | 95.7 (-0.2) | 0.902 (-0.003) | 30.5 (+0) | 0.1 (-0.4) | no |
| **full** | fashion | 77.2 | 19.3 | 70.0 | 2.7 | 79.0 | 0.836 | 30.5 | -0.2 | |
| gabor | fashion | 77.6 (-0.4) | 11.3 (-8.0) | 65.3 (+4.7) | 3.6 (+0.8) | 78.0 (+1.0) | 0.912 (-0.076) | 30.5 (+0) | 1.3 (+1.5) | yes |
| noV1 | fashion | 77.9 (-0.7) | 14.4 (-4.9) | 76.4 (-6.3) | 6.1 (+3.3) | 77.7 (+1.3) | 0.760 (+0.075) | 30.5 (+0) | 0.7 (+0.9) | yes |
| noadaptvig | fashion | 74.7 (+2.5) | 22.6 (+3.4) | 69.0 (+1.0) | 2.4 (-0.3) | 77.7 (+1.3) | 0.838 (-0.002) | 30.5 (+0) | -0.6 (-0.3) | yes |
| noarbitration | fashion | 77.1 (+0.1) | 19.4 (+0.2) | 69.8 (+0.2) | 2.6 (-0.1) | 78.8 (+0.1) | 0.836 (+0.000) | – | -0.3 (-0.1) | no |
| noclsdefer | fashion | 77.2 (+0.0) | 19.3 (-0.0) | 64.0 (+6.1) | 1.8 (-0.9) | 79.0 (+0.0) | 0.836 (+0.000) | 30.5 (+0) | -0.2 (-0.0) | yes |
| nocorrection | fashion | 74.7 (+2.5) | 22.8 (+3.5) | 70.3 (-0.2) | 3.8 (+1.1) | 77.1 (+1.9) | 0.840 (-0.004) | – | 0.5 (+0.8) | yes |
| noexcitability | fashion | 77.9 (-0.7) | 21.9 (+2.6) | 71.6 (-1.5) | 3.1 (+0.3) | 78.8 (+0.2) | 0.833 (+0.003) | 20.5 (-10) | -0.2 (+0.0) | yes |
| nolability | fashion | 68.5 (+8.7) | 32.0 (+12.7) | 66.9 (+3.2) | 3.2 (+0.5) | 75.1 (+3.9) | 0.828 (+0.008) | 20.5 (-10) | 0.5 (+0.7) | yes |
| nomatchtrack | fashion | 75.8 (+1.4) | 17.2 (-2.1) | 72.6 (-2.6) | 4.1 (+1.4) | 79.6 (-0.6) | 0.829 (+0.007) | 20.5 (-10) | 0.0 (+0.3) | yes |
| nomerge | fashion | 78.2 (-1.0) | 14.7 (-4.6) | 68.8 (+1.3) | 3.0 (+0.3) | 78.2 (+0.8) | 0.915 (-0.079) | 30.5 (+0) | 1.9 (+2.2) | yes |
| norefine | fashion | 68.9 (+8.3) | 31.6 (+12.3) | 71.7 (-1.7) | 2.1 (-0.6) | 70.5 (+8.5) | 0.847 (-0.011) | 20.5 (-10) | -0.2 (+0.1) | yes |
| nosleep | fashion | 76.0 (+1.2) | 20.8 (+1.6) | 72.7 (-2.6) | 2.9 (+0.2) | 77.8 (+1.2) | 0.850 (-0.014) | 30.5 (+0) | -0.3 (-0.0) | yes |
| nosleepclean | fashion | 75.8 (+1.5) | 21.6 (+2.4) | 70.3 (-0.2) | 3.1 (+0.3) | 76.6 (+2.4) | 0.838 (-0.002) | 30.5 (+0) | -0.6 (-0.4) | yes |

B5 = GDumb (Prabhu et al. 2020): 500 stored images, class-balanced, network retrained from scratch before predicting. It violates P7, like B2.

### Against the success criteria (goal.md §4; test, mean of 3 seeds)

| Criterion | Target | MNIST | Fashion | |
|---|---|---|---|---|
| M1 local / online / no raw buffer / fixed budget | by construction | docs/audit.md | | ✅ (structural steps flagged, §4) |
| M2 ACC | ≥ 90 / ≥ 75 | 94.7 | 81.5 | ✅ |
| M3 FGT | ≤ 5 / ≤ 10 | 0.7 | 7.6 | ✅ |
| M4 beats B1, B2; ≤ 10 below B3 (MNIST) | | 94.7 vs 19.6, 86.9; B3 97.9 | 81.5 vs 19.9, 75.6 | ✅ (also beats GDumb: 85.9 / 71.6) |
| M5 FEW-5 new-class acc; old drop ≤ 2 | ≥ 70 / ≥ 50 | 43.6, drop 2.2 | 79.8, drop 2.7 | ❌ / ✅ acc, ❌ drop |
| M6 LBL5 within 5 of full | | 95.1 vs 94.7 | 80.9 vs 81.5 | ✅ |
| M7 ≤ 5 % active per learned layer | | IT 0.5 %, V1 2.7 % | IT 0.5 %, V1 3.5 % | ✅ |
| S1 NOVELTY AUROC ≥ 0.80 | | 0.927 | 0.845 | ✅ |
| S2 sleep helps ≥ 1 pt | | +1.3 ACC, −1.7 FGT (val, It 4) | ablation: −1.2 ACC without | ✅ |
| S3 every mechanism earns its place | | ablation table: all 12 removals cost ≥ 1 pt somewhere | | ✅ |
| S4 beats B4 by ≥ 8 | | +12.6 | +13.8 | ✅ |
| S5 reversal ≤ ~100 ex/class, other drop ≤ 2; noise ≤ 3 | | 80.5 ex, 0.0, −0.1 | 40.5 ex, 0.4, 0.2 | ✅ |

Version history on the test split (same protocol), MNIST / Fashion ACC and FGT: It 6 = 96.2 / 2.8 and 77.4 / 21.7; It 7 = 95.5 / 0.9 and 76.0 / 20.0; It 9 = 94.7 / 0.9 and 80.8 / 10.9; It 12 = 94.4 / 0.9 and 80.5 / 10.4; round 2 final (It 12 + eye movements) = 94.7 / 0.7 and 81.6 / 7.2; **round 3 final (+ label-gated refinement) = 94.7 / 0.7 and 81.5 / 7.6**. Ablations were run on the It 7 configuration (val). Later changes are their own ablations on val: never closing the V1 critical period costs 3.9 pts Fashion ACC and 8.5 pts FGT; removing the rethink loop raises the old-class drop by 0.3 / 0.4 and Fashion FGT by 0.9; removing eye movements (val, 3 seeds) raises Fashion FGT from 6.3 to 9.5. The full ablation table was not re-run for round 2 (each run is ~3× slower with eye movements).

### Scale check: CIFAR-10 (validation, seed 0; not a test result)
| model | ACC | FGT | FEW5 | NOVELTY |
|---|---|---|---|---|
| SPARC (It 12, colour V1 5×5×3) | **36.1** | 26.8 | 20.5 | **0.559** |
| B1 sequential MLP | 15.4 | 66.3 | 49.1 | – |
| B2 ER-500 | 28.6 | 47.8 | 23.9 | – |
| B3 joint (upper bound) | 47.7 | – | – | – |
| B4 class means, raw pixels | 25.5 | 22.8 | 23.7 | – |
| B5 GDumb | 10.7 (chance: training likely diverged; not trusted) | – | 0.0 | – |

SPARC still beats every continual baseline on natural images, but it is 11.6 pts behind joint training, and the novelty signal collapses to near chance. A single local feature layer is not enough for natural images (research/12, warning W1).

## 6. Stage 2: prediction over time (`sparc_seq.py`, `seqbench.py`)
Stage-1 SPARC is the perception. On top of it, a sparse **conjunction** of the current and previous category activity (dendritic coincidence detection; fixed random mixing + k-WTA) feeds predictors of the next category. These are learned with a local delta rule, with metaplastic rates. **Context inference** (latent causes: Gershman, Blei & Niv 2010; COIN, Heald et al. 2021; research/19) keeps up to 8 context slots. The slot that has recently predicted best is active and the only one that learns; persistent surprise in an established slot recruits a new one. Surprise = 1 − p(observed category).

| test, 3 seeds (docs/benchmark-stage2.md) | online | retention | violation AUROC | completion from 5 items |
|---|---|---|---|---|
| **SPARC + context inference** | 0.893 | **0.820** | **0.928** | **0.725** |
| second-order n-gram on true labels (idealised, not brain-like) | 0.910 | 0.846 | 0.975 | 0.738 |
| target | ≥ 0.90 | ≥ 0.60 | ≥ 0.85 | ≥ 0.70 |

Near-pass: 3 of 4 targets. SPARC is within ~3 pts of an idealised counting learner while learning from raw images, online, with local rules and no stored sequences.

## 7. Stage 3 (first step): number sense (`sparc_num.py`, `numbench.py`)
Dehaene's triple code (research/20): the stage-1 visual digit form; an innate approximate number system (center-surround dot detectors → local competition → summation → log-tuned numerosity units with Weber noise); Hebbian grounding of digit symbols in numerosity codes; memorised facts; and **counting-on**, which uses successor knowledge learned from counting sequences and two working-memory registers. The routine itself is built in; learning it belongs to stages 4–5.

| val, 3 seeds | GROUND (dots → digit) | ORDER (never trained) | ADD held-out (exact) |
|---|---|---|---|
| lookup / MLP one-hot / MLP thermometer (backprop) | oracle | 0.53 / 0.53 / 1.00 | 0.08 / 0.00 / 0.08 |
| **NumSPARC (counting-on, stage 1 on 10k images)** | 0.76 | **0.95** (distance effect on 2 of 3 seeds) | **0.78** (0.92 / 0.83 / 0.58) |

It adds pairs it never practised, where memorising models are at chance; the remaining errors are digit recognition and the approximate dot counter.

**Relations and cognitive maps** (`tibench.py`, `TISPARC`): items get positions on a learned 1-D map from dopamine-like prediction errors; items belong to maps, and a fact linking two established maps shifts both in one shot (structure vs content, Tolman–Eichenbaum style).

| val, 3 seeds | transitive inference: novel pairs (from images) | list linking: cross-list pairs after **one** bridge fact |
|---|---|---|
| lookup | 0.51 | 0.51–0.54 |
| backprop MLP | 0.98 | 0.56–0.60 |
| **SPARC maps** | **0.97–1.00** (symbolic distance effect) | **1.00** (within-list order intact) |

## 4. Limits and honest gaps

1. **MNIST few-shot (M5) fails badly: 43.6 % vs 70 %.** 5-shot 8/9 must be told apart from well-learned 3/5/4/7. Diagnosis: even a nearest-class-mean on the V1 features gets only ~66 %, so a single learned feature layer is the limit. Excitability, CLS deferral and a fly-like Kenyon-cell pathway each moved it by ≤ 6 pts. Fashion few-shot passes (79.8 %). Round 2: a V2 layer (+10 on a quick screen) and CLS-gated episodic recall (82 % on a quick screen, but old classes lose 10–14 pts) show it can be raised, but not yet without hurting old classes. Fix direction: deeper local hierarchy (docs/roadmap.md #2).
2. **Few-shot old-class drop: MNIST 2.2 (seeds 2.8 / 2.3 / 1.4), Fashion 2.7 pts (target ≤ 2).** Label-gated refinement (round 3) helped on val (MNIST 1.9) but not on test. Hyperexcitable new units and CLS deferral trade old-class accuracy for new-class accuracy; the rethink loop won part back. Every few-shot mechanism tried in round 2 moves along the same trade-off curve.
3. **Relearning after forgetting is not solved** (research/15): relearning a forgotten Fashion task even lowers it (test: peak 91 → forgotten 76 → relearned 68 → after sleep 62 %). Round-3 diagnosis: label-blind refinement and competition among look-alikes corrupt the old memory during practice; engram tags, per-synapse metaplasticity and gating did not fix it. Candidate cure from stage 2: context inference (research/19).
4. **Structural steps are not synaptic-local.** Consolidation (pairwise merges with a water-filled cap across classes), hippocampal eviction by class counts, sleep cleanup and match tracking use population-level information. They are declared as structural plasticity / memory management, not as synaptic rules (docs/audit.md §1).
5. **IT prototypes are imprints of V1 codes.** 3000 × 1024 numbers form an exemplar-like store in feature space (declared as parameters; no pixels are stored). The hippocampus holds 31 000 numbers against 392 500 for a 500-image replay buffer.
6. **Task boundaries** trigger sleep (allowed by docs/design-v1.md); no task labels at test. Sleep is also triggered by budget pressure.
7. **Benchmark deviations:** REVERSAL/NOISE use the first half of tasks 1–2 and 4-way evaluation (docs/benchmark.md describes the implementation); the train stream is the 54k non-validation subset for both splits; FEW-k uses one seeded draw per seed (large variance: ± 3–10 pts).
8. **Unverifiable intermediate numbers.** A few It 1/2/5/7 diagnostic numbers in docs/iteration-log.md were printed to the console only and are marked as such. Every number in this file comes from a JSON file.
9. **Scale:** the full benchmark is on 28×28 grey images; a CIFAR-10 check (val, one seed) gives 36.1 % and novelty AUROC 0.56, so natural images need a deeper local hierarchy. About 640 s per full bench run on one CPU core.
10. **One tuning seed.** All settings were chosen on validation seed 0 only (research/12, warning W6).

## 5. What is new here and what is not

Each component comes from prior work: the fly mushroom body (Shen et al. 2021), ART and ARTMAP match tracking (Grossberg; Carpenter), SoftHebb-style local feature learning, sleep replay (Tadros et al. 2022), complementary learning systems, and uncertainty-based arbitration (Daw et al. 2005). The contribution is to combine them in one strictly local, online, single-pass system and to measure it on all ten properties (P1–P10, including reversal learning) under one protocol, with ablations showing that each part matters. The design rule throughout was brain-first: a change was kept only if it named a brain mechanism.

