# Starting design (hypothesis v1, Nov 2026)

> Historical: the design the loop started from. The current design is in ARCHITECTURE.md.
*Sparse, Predictive, Adaptive-Resonance network with Consolidation.* Treat everything here as a hypothesis the loop is allowed to change.

```
image ─► V1  local patches, divisive normalisation, competitive-Hebbian
         │   dictionary, k-winners-take-all per position, spatial pooling
         ▼
         IT  adaptive-resonance prototype layer: cosine match, k-WTA,
         │   vigilance recruitment; surprise s = 1 − max match
         ▼ sparse code h
   ┌─────┴──────────────────────────────┐
   ▼                                    ▼
 FAST readout (fly-style)          HIPPOCAMPUS: bounded store of sparse
 only true class row, only         codes, written with prob ∝ surprise
 active synapses change                 │ (sleep: balanced replay)
   │                                    ▼
   │                               SLOW cortical readout (delta rule,
   │                               trained only during sleep) + downscaling
   └──────────► combine by confidence ◄─┘
 surprise s (ACh/NE analogue) gates: plasticity, recruitment, memory writes
```

**Learning rules (all local)**
- Generic three-factor form: `Δw_ij = η_i · M · y_i · (x_j − w_ij)` — η_i from the unit's own usage, M a broadcast modulator (surprise or reward), y post, x pre.
- **V1 dictionary** (K ≈ 64 features, 5×5 patches, contrast-normalised): for each sampled patch p, winner k moves `d_k ← normalise(d_k + η_k (p − d_k))`, with `η_k = max(1/n_k, η_min)`. If best match < ρ₁ and unused units remain, imprint p onto a new unit. Encode with k-WTA per position (≈3 of 64) then pool over a 3×3 grid (≈576-d). Ablation: innate fixed Gabor bank (developmental prior) vs learned.
- **IT layer** (budget N ≈ 1500, k ≈ 15 active, i.e. 1 %): `a = W x` (cosine). If `max a < ρ` and free units remain → imprint `w_new = x` (one-shot). Else top-k winners: `w_i ← normalise(w_i + η_i a_i (x − w_i))`, `η_i = 1/(1 + n_i)`. Surprise `s = 1 − max a`.
- **Fast readout**: labelled sample of class c → `A_c ← A_c + (h − A_c)/n_c` (only row c changes). Predict `argmax_c cos(A_c, h)`. No depression of other classes, so no recency bias.
- **Hippocampus**: store (top-k indices, values, label) with probability rising with s; capacity M ≈ 1000 codes (≈ 1000 × 15 × 2 numbers — report this budget against a 500-image raw buffer).
- **Sleep** (after each task, or every N samples): replay stored codes class-balanced; train slow readout R with `ΔR = η (onehot(c) − softmax(R h)) hᵀ`; then multiplicative downscaling (homeostasis). Final score = confidence-weighted mix of fast and slow readouts.
- **Few labels**: V1 and IT learn from every sample; readouts only from labelled ones.

**Known risks and first fixes to try**
1. *Feature drift*: if V1/IT keep moving, stored class means go stale → 1/usage learning rates, slower V1 than IT, or re-anchor during sleep.
2. *Budget exhausted early*: vigilance too strict → adaptive vigilance (recruit when s is in the top x % of recent surprises) or recycle the least-used unit.
3. *Recency bias*: any softmax/delta readout trained online on a class-incremental stream will favour new classes → train it only in sleep with balanced replay.
4. *Few-shot misses*: novel test images may activate old units → check that few-shot examples recruit units; allow several prototypes per class.
5. *Fake wins*: count every stored number; tune on validation only; never use task labels at test.
