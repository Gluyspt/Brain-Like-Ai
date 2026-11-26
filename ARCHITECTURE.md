# SPARC — Sparse, Predictive, Adaptive-Resonance network with Consolidation

A continual learner whose **learning** is built from brain mechanisms only: local three-factor plasticity, online single-pass learning, sparse codes, novelty-gated recruitment, a hippocampal store, sleep replay, and error-driven self-correction. No backprop, no autograd, no raw-image replay. Code: `sparc.py` (+ `v1.py`), benchmark: `bench.py`, history: `goal.md` §10.

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
| Novelty signal | s = 1 − max a (no learning at test) | mismatch / surprise signal |

## 3. Results

RESULTS_PLACEHOLDER

## 4. Limits and honest gaps

GAPS_PLACEHOLDER
