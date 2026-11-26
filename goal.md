# goal.md — Build an architecture that learns like a real brain

> **Handoff brief for a Claude Code session.** Read the whole file, set up the agent team (§12), then run the loop in §8.
> Update the iteration log (§10) after every run, failures included. Stop rule is in §8.
> Status (2026-11-26): loop run It 0–9 (+5b, 8). Final SPARC (test, 3 seeds): MNIST 94.7 / FGT 0.9, Fashion 80.8 / FGT 10.9; all SHOULDs pass; MUST gaps: MNIST few-shot, few-shot old-class drop, Fashion FGT 0.9 over. See ARCHITECTURE.md and audit.md.

---

## 1. Why
Today's LLMs and deep nets learn in ways no brain does.

| Brain | Backprop nets / LLMs |
|---|---|
| **Local plasticity**: a synapse changes using only its own pre/post activity plus a broadcast neuromodulator | **Global backprop**: exact error sent backwards through symmetric weights (weight transport); forward pass frozen while it happens |
| Learns **online** from one continuous stream; each moment seen once | Shuffled i.i.d. data, many epochs, then frozen |
| Keeps old skills while learning new ones | **Catastrophic forgetting** when trained sequentially |
| **One-shot** learning of a new thing (hippocampus) | Thousands of gradient steps per new concept (outside the context window) |
| Mostly **unsupervised**; labels and rewards are rare | Needs a dense training signal for every example |
| **Sparse** activity (a few % of neurons active) | Dense activity, high energy |
| **Knows when something is new** (mismatch / surprise signals) | Confidently wrong on novel inputs |
| **Sleeps**: offline replay consolidates memories | No offline consolidation phase |

## 2. Goal
Design a new architecture whose **learning** works like a brain's, prove it on a benchmark that measures brain-like learning, and iterate until it passes.
Scope: the learning machinery, tested at small scale (image streams). Language scale is the roadmap (§11), not this loop.

## 3. Testable definition of "learns like a brain"
> **Brain-first rule (user directive, 2026-11-25).** Every step or decision about the model or its training must make it act more like a real brain. Name the brain mechanism behind each change in the §10 log. A change that only raises an accuracy number with no brain analogue does not belong here, even if it scores better; when a brain-like and a score-only option compete, pick the brain-like one and log the cost. Accuracy targets in §7 are checks that the brain-like machinery works, not the objective.

- **P1 Local:** every weight update uses only pre-synaptic activity, post-synaptic activity and broadcast scalar modulators (three-factor rule). No backprop, no autograd, no weight transport. A single-layer delta rule is allowed (the error lives at the output neuron itself, like a teaching input).
- **P2 Online:** batch size 1, each example seen once, stream order (no shuffling, no epochs), learning and inference interleaved.
- **P3 Continual:** class-incremental stream, no task labels at test time.
- **P4 Fast:** learns a new class from 1–5 examples without disturbing old classes.
- **P5 Mostly unsupervised:** representations learned without labels; labels only bind representations to names.
- **P6 Sparse:** ≤ 5 % of units active per input in every learned layer.
- **P7 Fixed resources:** neuron budget fixed up front; no raw-input storage (no image replay buffer). Compressed internal codes are allowed (hippocampal index) with a declared, reported budget.
- **P8 Novelty-aware:** an internal surprise signal that is high for never-seen classes.
- **P9 Sleep:** an offline replay phase that measurably helps.
- **P10 Self-correcting:** a wrong association can be unlearned when contrary evidence arrives (reversal learning), without damaging unrelated classes. Brain basis: confident-error prediction-error signals (dopamine/noradrenaline) boost plasticity where the mistake happened; recalled memories become labile again (reconsolidation); old links are often outcompeted rather than erased (extinction).

## 4. What we already learnt (research)
The research lives in **`research/`**: one file per brain system, each giving what the brain does, the evidence, how it helps SPARC, how it might work, the test and its status. Start at `research/README.md`. The next experiments, ranked by the gaps they target, are in `research/10-experiment-backlog.md`. Scan 1 (Nov 2026: fly, ART, SoftHebb, sleep replay, HiCL, Nested Learning, BDH) is in files 01–03 and 09. Scan 2 (2026-11-26: hemispheres, PFC, pulvinar, basal ganglia, cerebellum) is in files 04–08.

**Takeaways that drive the design**
1. Forgetting is mostly **interference**. Sparse, high-dimensional codes plus updating only active synapses removes most of it (fly).
2. **Recruit new units for novel input; refine old units only on a good match** (ART). Stability without freezing everything.
3. **Use-dependent plasticity** (learning rate ∝ 1/usage) makes heavily used units stable, like a critical period, while fresh units stay plastic.
4. Local, feedback-free layers can learn good features fast (SoftHebb), so backprop is not required for usable representations.
5. **Sleep with replay** can repair what waking learning damaged (SRC).
6. Gap: the systems above each cover a few properties, and most continual-learning systems still train with backprop over many epochs. In this scan we did not find one model tested on **all** of P1–P9 at once under strict rules. That combined, measured system is the contribution; do not claim the individual parts are new.

## 5. Starting design (hypothesis v1) — working name **SPARC**
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

## 6. Benchmark protocol (`bench.py`)
- **Data**: MNIST and Fashion-MNIST (60k train / 10k test). Hold out 10 % of train as validation for all tuning (`--split val`); touch the test set only for logged results (`--split test`).
- **Stream**: 5 tasks × 2 classes `(0,1)(2,3)(4,5)(6,7)(8,9)`, class-incremental, single pass. SPARC batch 1. Baselines batch 10 (standard online-CL setting, generous to them).
- **ACC**: final 10-way test accuracy.
- **FGT**: for tasks j = 1..4, `max_t acc_j(t) − acc_j(final)`, where `acc_j(t)` is accuracy on task j's test classes after task t (predicting over all classes seen so far); mean over j.
- **FEW-k** (k = 1, 5): learn tasks 1–4, then present only k labelled examples each of classes 8 and 9 (seeded draw, batch 1). Report accuracy on 8/9 test images (10-way) and the accuracy drop on classes 0–7.
- **LBL5**: full stream, only 5 % of examples carry a label.
- **SPARSITY**: mean fraction of non-zero units per learned layer on test inputs.
- **NOVELTY**: after task 4, AUROC of surprise s for unseen classes (8, 9) vs seen (0–7).
- **REVERSAL** (as implemented): learn the first half of tasks 1–2, then stream the second half of task 2 with labels 2↔3 swapped. Every 20 samples, measure 4-way accuracy on 2/3 under the new mapping; report examples per class until ≥ 80 %, the final value, and the change on classes 0/1. **NOISE**: in tasks 1–2, 20 % of first-half labels flipped to the pair partner, second half clean; report 4-way accuracy on classes 0–3 after task 2 vs a clean run.
- **Seeds**: 1 for quick runs, 3 for the final confirmation.
- Also log: wall-clock, parameter count, stored numbers.

**Baselines**
- **B1** backprop MLP 784-400-400-10, same stream (expect catastrophic forgetting).
- **B2** B1 + experience replay of 500 raw images (reservoir; each step 10 current + 10 replayed). Standard CL baseline; violates P7.
- **B3** backprop MLP, joint shuffled data, 10 epochs (upper bound; violates P2, P3).
- **B4** nearest class mean on raw pixels (no feature learning).

Rough sanity ranges from memory, verify before trusting: B1 ≈ 20 % ACC (only the last task's classes get predicted); B3 ≈ 98 % MNIST / ≈ 89 % Fashion; B4 ≈ 82 % MNIST.

## 7. Success criteria
**MUST** (mean of 3 seeds, both datasets unless stated)
- **M1** P1, P2, P6, P7 hold by construction (code audit: no autograd, no shuffling, no raw buffer).
- **M2** ACC ≥ 90 % MNIST, ≥ 75 % Fashion.
- **M3** FGT ≤ 5 pts MNIST, ≤ 10 pts Fashion.
- **M4** beats B1 and B2 on ACC; within 10 pts of B3 on MNIST.
- **M5** FEW-5 novel-class accuracy ≥ 70 % MNIST, ≥ 50 % Fashion, with old-class drop ≤ 2 pts.
- **M6** LBL5 ACC within 5 pts of full-label ACC.
- **M7** ≤ 5 % active units in every learned layer.

**SHOULD**
- **S1** NOVELTY AUROC ≥ 0.80 (MNIST).
- **S2** sleep improves ACC or FGT by ≥ 1 pt.
- **S3** every mechanism earns its place: ablating it costs ≥ 1 pt on some metric.
- **S4** beats B4 by ≥ 8 pts ACC.
- **S5** REVERSAL: new mapping adopted within ~100 examples per class, other classes drop ≤ 2 pts; NOISE: affected-class accuracy within 3 pts of the clean run.

## 8. Loop protocol
Each iteration:
1. **Hypothesis**: which brain mechanism should fix which observed failure (one sentence). Main session only.
2. **Implement** the smallest change. One change per iteration. When two or three fixes compete, give each to a `variant` agent in parallel (§12) and keep the winner.
3. **Run** quick bench (MNIST, seed 0). Full bench (2 datasets × 3 seeds) only once quick passes, handed to a `runner` agent.
4. **Log** the numbers against §7 in §10, after opening the JSON files yourself.
5. **Diagnose** → next hypothesis. Main session only.

Suggested order (deviate when the data says so):
- **It 0** — setup from §12 (git, CLAUDE.md, agent files); a `builder` writes data.py / baselines.py / bench.py; a `runner` runs B1–B4; check the sanity ranges in §6.
- **It 1** — IT layer + fast readout on raw pixels (no V1): tests the core anti-forgetting mechanism.
- **It 2** — metaplasticity + vigilance recruitment tuning if FGT is high.
- **It 3** — add V1 if ACC is below target (learned vs innate-Gabor ablation).
- **It 4** — hippocampus + sleep; measure S2.
- **It 5** — few-shot, few-label, novelty metrics; fix whichever fails.
- **It 5b** — self-correction (P10): confident-mistake depression of active→wrong-class links (three-factor rule, M = confident error), reopen plasticity of the units active in the mistake (reconsolidation), evidence-weighted links so one bad label only nudges, sleep cleanup of conflicting hippocampal memories. Test with REVERSAL and NOISE. Known weakness it fixes: the fast readout only ever strengthens the labelled class, so a wrong link is only diluted, never removed.
- **It 6+** — full run, ablations (S3), then Fashion-specific fixes.

**Stop** when every MUST passes on the full bench → run ablations → write `ARCHITECTURE.md` (design, equations, brain mapping, results table, limits). Or stop after **12 iterations** and report the best version plus an honest gap analysis.

## 9. Environment notes
- Both data sources returned HTTP 200 from a sandbox on 2026-11-26:
  - MNIST: `https://raw.githubusercontent.com/fgnt/mnist/master/<file>`
  - Fashion: `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/<file>`
  - `<file>` ∈ `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`. Parse with `gzip` + `numpy.frombuffer` (images: skip 16-byte header, labels: 8).
  - Offline fallback: `sklearn.datasets.load_digits()` (8×8, no download).
- numpy + scikit-learn are enough. Baselines can be a small numpy MLP; use PyTorch only if already installed (never inside SPARC).
- Speed (rough estimate): a per-sample Python loop with ~1500 IT units should be about 1 ms/sample, so about a minute per 60k pass on CPU. Batch the test-set encoding.
- Suggested layout:
  ```
  goal.md        this file (keep the log current)
  data.py        load datasets, build the class-incremental stream
  baselines.py   numpy MLP (B1–B3), nearest class mean (B4)
  sparc.py       the architecture (no autograd anywhere)
  bench.py       protocol + metrics → results/<model>_<data>_<seed>.json
  results/
  ```
  e.g. `python bench.py --model sparc --data mnist --seed 0 --split val --quick`

## 10. Iteration log
| # | Hypothesis / change | MNIST ACC | FGT | Other | Agents (model/effort) | Verdict |
|---|---|---|---|---|---|---|
| – | Research + design only; nothing run yet | – | – | – | – | start at It 0 |
| 0 | Setup (§12) + data.py/baselines.py (builder) + bench.py (main). Baselines MNIST val seed 0, full stream: **B1** ACC 19.6 / FGT 99.4; **B2** (ER-500) ACC 85.5 / FGT 17.1 / FEW5 66.2 (old drop 24.8); **B3** joint 97.9; **B4** NCM 81.2 / FGT 7.5 | – | – | All in §6 sanity ranges | builder: sonnet/medium (general-purpose: .claude/agents files only register after a session restart); baselines run as a plain background command (no runner needed) | OK |
| 1 | IT ART layer (N=1500, k=15, rho=0.8, eta=1/(1+n)) + fly fast readout on raw centred pixels | 85.6 (quick 85.6) | 6.8 | quick: FEW1 14.5, FEW5 34.9 (drop 3.7), LBL5 84.8, NOV 0.81, sparsity 1.0 %. Diagnosis (console only, tag reused → not in JSON): budget full at sample 12.5k of 54k; units/class 365…50; n_it=6000 → 93.2 / 3.6; refine=False → 79.4 / 10.4. In JSON: rho=0.7 → 88.3 / 5.7 (full at 24.8k) | main | budget exhaustion is the main failure → It 2 |
| 2 | Budget control. (a) turnover: recycle least-used unit after grace; (b) adaptive vigilance: recruit only if s > μ_s + z·σ_s; (c) sleep-pressure consolidation: when budget full (and at task end) merge most-similar same-class prototype pairs, water-filled so the largest classes shrink first, until 25 % is free | 89.6 (c + b z=1, adopted) | 6.8 | (a) grace 1000: 90.6/4.3; (a)+(b): 91.0/3.3 **but** (a) grace 5000: 19.4/99.1 (eats every old unit) → unsafe; (a) with only n≤1/3 units recyclable: 85–86; (b) z=2 alone: 89.9/6.1; (c) without fairness (global most-similar): 70–77, class 1 shrinks to 6 units (console only, JSON overwritten). Remaining loss: 4→9, 2/3/5→8 confusions in pixel space | main (flag grid run directly, not variant agents: 10-line flag changes, and worktrees would branch from the default branch) | adopt (c)+(b); representation is now the bottleneck → It 3 V1 |
| 3 | V1 layer before IT: 5×5 contrast-normalised patches, K=64, k-WTA 3/position, 3×3 sum-pool (576-d). Learned competitive-Hebbian dictionary (1/usage rate = critical period) vs innate Gabor bank; rho swept | 92.7 (learned, rho 0.8) | 5.9 | Gabor rho 0.8: 91.0/4.4; learned rho 0.7/0.9/0.95: 89.1/92.8/92.3; full metrics (learned): FEW1 20.9, FEW5 45.0 (old drop 4.8), LBL5 92.8, NOV 0.878, sparsity IT 1.0 % / V1 2.6 %. Budget no longer fills at rho 0.8 (1879 recruits) | main | adopt learned V1 (brain: experience-dependent V1 with critical period; it also scored higher) |
| 4 | Hippocampus (1000 sparse IT codes, idx+val+label = 31k numbers vs 392.5k for ER-500; write p = σ(surprise z-score); full → overwrite a trace of the most-stored class) + sleep after each task: class-balanced replay trains a slow cortical readout with a delta rule (CLS theory); confidence-weighted mix with the fast readout. Consolidation remaps stored indices to merged units | 94.2 | 4.0 | no sleep (same code): 92.9/5.7 → **S2 ✓** (+1.3 ACC, −1.7 FGT). slow-only 94.3/3.6 but FEW5 23; mix 0.5 ≈ conf. FEW5 gets worse with sleep (44.7 → 31.1): the slow readout learns a new class from 5 memories poorly | main | adopt conf mix; few-shot is now the worst failure → It 5 |
| 5 | Few-shot. Diagnosis (scratch scripts, console only): shots do recruit IT units (6/10), but even V1-space NCM gets only 58 % on 5-shot 8/9 → representation-limited. Changes: (a) ARTMAP match tracking: a labelled sample whose fast prediction is wrong raises vigilance and recruits a unit (ACh/NE after unexpected outcome); (b) V1 pooling 3×3 → 4×4 (smaller complex-cell pools; NCM proxy 60 → 67 %). Also fixed a hippocampal index-remap crash (empty code slots) | 94.5 | 4.1 | MNIST: FEW1 12.6, FEW5 56.8 (drop 2.6), LBL5 94.6, NOV 0.894, REV final 70 % (never 80 %), NOISE gap 0.1. grid 3: 94.5/4.1, FEW5 43.8. **Fashion (first run)** grid 4: ACC 74.3, **FGT 26.6**, FEW5 68.1 (drop 1.5), LBL5 75.0, NOV 0.82; per class pullover 23 %, sandal 49 %; forgetting = new class captures a look-alike old class (pullover→coat, coat→shirt, sandal→sneaker) | main | adopt (a)+(b); Fashion FGT is the main failure → confident-error LTD (It 5b) is the brain mechanism for exactly this discrimination problem |
| 5b | Self-correction (P10): (1) confident error (q>0.5) → M=q: LTD at active synapses onto the wrongly predicted output, scaled by its evidence; (2) reconsolidation: both output neurons lose evidence (plasticity reopens); (3) sleep cleanup: hippocampal traces confidently contradicted by current fast readout are dropped. Then (4) **lability gating**: only output neurons labelled in the last ~τ samples are labile (reconsolidation needs reactivation); (5) **arbitration by uncertainty** (Daw et al. 2005): per-class trace of recent confident errors discounts the slow readout | 94.2 | 4.8 | Fashion 75.2 / 24.3. REV → 80 % in 30.5 (MNIST) / 10.5 (Fashion) ex/class (20-sample grid), final 98.9 / 96.9 %, other drop −0.1 / 1.7, NOISE gap 0.1 / 0.9 → **S5 ✓**. Path: (1)–(3) alone: REV ✓ (670 / 30 ex) but Fashion 63.7 / 41 (new classes depress look-alike old ones); +(4) τ=1000: MNIST cost gone, Fashion 69.1/33; τ=200+(5): 72.0/28.3; τ=50+(5): adopted. Fashion baselines (val): B1 20.8, B2 75.3 (FGT 24.2), B3 88.9, B4 68.8 (FGT 12.2) | main | adopt (1)–(5); Fashion FGT 24 ≫ B4 12 → isolate cause (It 6) |
| 6 | Diagnosis of Fashion FGT (val): fast-only 75.1/23.2, slow-only 73.6/27.2, no refinement 68.1/33.8, no merging (N=8000) 77.8/23.4 → forgetting is look-alike capture, not merging or readout; FGT also penalises learning each task well (pullover/dress 97 % alone). Hypothesis: a larger / denser IT population separates look-alikes better. Grid: k 30 / 50 (2–3.3 %): worse on both (MNIST 92.8 / 90.9). **N=3000, k=15**: MNIST 96.1 / 3.1, Fashion 77.8 / 21.9 | 96.1 | 3.1 | N=3000: FEW1 11.8 / 30.9, FEW5 44.4 (drop 2.8) / 69.2 (drop 1.9), NOV 0.898 / 0.833, IT sparsity 0.5 %. FEW5 varies 44–57 between configs (single draw of 5 shots) | main | adopt N=3000 (brain: larger cortical population = more capacity; cost: params 1.19M → 3.13M, ~1.6× wall-clock); go to full 3-seed test bench before more changes |
| F1 | **Full bench, test split, 3 seeds** (commit 6ba0e2a = It 6 config; runner haiku/low, 30 runs, no crashes; numbers recomputed by main from the JSONs, mean ± sd) | **96.2 ± 0.2** | **2.8 ± 0.1** | MNIST: FEW1 12.9, FEW5 38.8 ± 3.8 (drop 1.8), LBL5 96.2, NOV 0.899, REV 30.5 ex/class (final 99.5, other drop 0.0), NOISE gap 0.0. **Fashion**: ACC 77.4 ± 0.4, FGT 21.7 ± 0.5, FEW1 37.1, FEW5 75.8 ± 7.5 (drop 2.4), LBL5 74.0, NOV 0.836, REV 20.5 (final 95.9, drop 1.7), NOISE −0.1. Sparsity IT 0.5 %, V1 2.7 / 3.5 %. Stored 31k numbers; params 3.13M. Baselines (test, 3 seeds) MNIST / Fashion: B1 19.6 / 19.9; B2 86.9 (FGT 15.0, FEW5 64.1 drop 44.9) / 75.6 (FGT 25.6); B3 97.9 / 87.9; B4 82.1 / 67.7. **§7: pass M1 M2 M4 M6 M7 S1 S2 S4 S5, M3-MNIST; fail M3-Fashion (21.7 > 10), M5-MNIST FEW5 (38.8 < 70), M5-Fashion drop (2.4 > 2)** | runner haiku/low; auditor sonnet/high (audit.md) | continue loop: It 7 targets MNIST few-shot |
| 7 | Few-shot diagnosis (MNIST val, scratch script, console only): 5-shot 8/9 → full SPARC 44.9 %, fast-only 51.6 %, V1-feature NCM 65.9 % (old 96.3 / 95.2 / 91.1) → IT exemplar code + slow readout suppress new classes. (a) **intrinsic excitability**: newborn IT units start hyperexcitable, homeostasis pulls each unit's win rate to k/used (γ=0.05; adult-born granule cells, Turrigiano): FEW5 42.1/40.7 (γ .03/.1) but old drop 2.8 → 1.4/1.0. (b) **CLS deferral**: when the fast system's best class has < 50 hippocampal traces (not consolidated), the slow readout's vote is scaled down: FEW5 49.9, drop 3.7 (a first version gating on the slow system's class did nothing). (a)+(b) | 96.0 | **1.0** | Fashion 77.0 / **19.3**; FEW5 49.2 (drop 2.2) / 70.1 (drop 2.9); NOV 0.900 / 0.836 | main | adopt (a)+(b): FGT down on both |
| 8 | Fly mushroom-body pathway: fixed random sparse wiring V1 → 4000 Kenyon cells (fan-in 10, top 200 fire), fly readout from KCs, third vote in the confidence mix | – | – | MNIST val FEW5: with KC 47.1 (drop 1.4), fan-in 20 46.2 (0.8), KC readout alone 50.9 (drop 0.35); without 49.2 (2.2) | main | reject (no gain; flag left off). MNIST few-shot of look-alike digits is limited by one learned feature layer → roadmap §11.2 (depth) |
| F2 | **Full bench, test, 3 seeds, It 7 config** (commit 394929b, incl. the audit's unit-0 sleep fix) | **95.5 ± 0.2** | **0.9 ± 0.0** | MNIST FEW1 19.2, FEW5 45.4 ± 3.1 (drop 2.4), LBL5 95.3, NOV 0.898, REV 30.5 (final 98.9, drop −0.1), NOISE 0.3. Fashion ACC 76.0 ± 0.1, FGT 20.0 ± 0.2, FEW5 80.0 ± 9.4 (drop 4.0), LBL5 75.6, NOV 0.834, REV 33.8 (final 93.6, drop 1.0), NOISE 0.3. vs F1 (It 6): ACC −0.7/−1.4, FGT −1.9/−1.7, FEW5 +6.6/+4.2, old drop +0.6/+1.6 (config chosen on val, not changed on test) | plain background run | M5 drop now fails on both |
| A | **Ablations** (val, seed 0, It 7 config, 13 × 2 runs; `results/report_tables.md`) | – | – | every mechanism costs ≥ 1 pt on some metric when removed (S3 ✓): V1 (MNIST ACC −4.7), merge (MNIST FEW5 −17.7), CLS deferral (FEW5 −7.7/−6.1), excitability (FGT +2.2/+2.6), match tracking (MNIST FEW5 −4.3, Fashion ACC −1.4), adaptive vigilance (Fashion ACC −2.5, MNIST REV 30→80), arbitration (MNIST REV 30→1720), correction (Fashion ACC −2.5, REV never), lability (Fashion ACC −8.7), refine (Fashion ACC −8.3, MNIST NOV −0.074), sleep (Fashion ACC −1.2, FGT +1.6), sleep cleanup (Fashion ACC −1.5). **Surprise**: on Fashion, Gabor V1 FGT 11.3 and no-V1 14.4 vs learned 19.3 at equal ACC; no-merge FGT 14.7 → learned-V1 drift (rate floor 0.002 never closes) makes stored IT prototypes stale | main | It 9: close the V1 critical period |
| 9 | **Close the V1 critical period** (PV-interneuron maturation / perineuronal nets: adult V1 tuning is stable). V1 plasticity stops after v1_cp images; or no rate floor (gradual closing) | 94.9 | 1.0 | cp 12k: **Fashion 81.1 / 10.8** (was 77.2 / 19.3), FEW5 49.2 (drop 2.6) / 71.2 (drop 2.2), NOV 0.913 / 0.858. cp 5k: 94.7/0.9, F 80.6/11.3; cp 2k: 94.2/0.8, F 81.7/10.7 (MNIST FEW5 42.9); eta_min 0: 95.1/0.9, F 80.5/11.8. Learned-then-frozen beats both innate Gabor (F 77.6/11.3, M 93.7) and never-closing | main | adopt cp = 12k (a fixed developmental age, not task-aware) |
| F3 | **Final bench, test, 3 seeds, It 9 config** (commit 8f79f9b) | **94.7 ± 0.1** | **0.9 ± 0.0** | MNIST FEW1 17.8, FEW5 43.7 ± 3.4 (drop 2.2), LBL5 94.7, NOV 0.919, REV 30.5 (final 98.8, drop −0.1), NOISE 0.5. **Fashion ACC 80.8 ± 0.1, FGT 10.9 ± 0.9**, FEW5 81.8 ± 9.9 (drop 3.2), LBL5 80.5, NOV 0.856, REV 33.8 (final 93.5, drop 1.1), NOISE −0.2. §7: pass M1 M2 M4 M6 M7, M3-MNIST, M5-Fashion acc, all S1–S5; **fail M3-Fashion (10.9 > 10), M5-MNIST FEW5 (43.7 < 70), M5 old drop (2.2 / 3.2 > 2)** | main (plain background run) | **STOP** after 10 iterations: remaining gaps need depth (§11.2) or would be score-tuning of γ / gate (brain-first rule). Write-up: ARCHITECTURE.md |

## 11. Roadmap after the loop
1. **Time**: add temporal prediction (predict the next input from the current sparse code with dendrite-like context), test on sequence streams, then character-level text.
2. **Depth**: stack local layers (SoftHebb-style) for CIFAR-10/100.
3. **Reward**: dopamine-like reward-prediction error as the modulator M for reinforcement learning.
4. **Language**: compare a SPARC-style continual learner with BDH and Nested-Learning/Hope on continual language modelling.

## 12. Agent team — when and how to spawn subagents
**Principle:** the main session is the scientist; subagents are lab techs. Judgement never leaves the main session: hypotheses, diagnosis, pass/fail calls, merging code, editing goal.md, the final write-up. Work that is self-contained, fits in a short brief and can be checked from the files it produces goes to the **cheapest model and lowest effort that can do it**.

**Spawn** when a task is self-contained, checkable from its output files, and either parallel (seeds, datasets, ablations, competing variants) or long and mechanical (writing a loader, babysitting a batch of runs).
**Don't spawn** for a few tool calls of work, for anything that needs this session's accumulated context (diagnosis), or for one plain command: a background Bash run is cheaper than an agent.

### Roles
| Role | Use it for | Model | Effort | Isolation | Parallel? |
|---|---|---|---|---|---|
| **Main session** | hypotheses, diagnosis, pass/fail, merging, goal.md, ARCHITECTURE.md | strongest available (Opus) | high–max | – | – |
| **builder** | data.py, baselines.py, metrics, bench.py, plots: any clearly specified module | sonnet | medium | own files only | yes, if files don't overlap |
| **variant** | one competing hypothesis, implemented and quick-benched | sonnet | high | worktree | yes, 2–3 at a time |
| **sweeper** | hyperparameter grid on the validation split | sonnet | medium | – | one per grid |
| **runner** | a fixed list of bench commands (seeds × datasets × ablations), retry crashes, results table | haiku | low | – | yes |
| **scout** | one narrow literature question, e.g. "published ER-500 numbers for online class-incremental Split-MNIST?" | haiku (sonnet if it struggles) | low | – | yes |
| **auditor** | fresh-eyes check of M1 rules and of every reported number vs the JSON files | sonnet | high | – | after big merges + at the end |

**Escalation ladder:** start at the role's default. If an agent fails twice, or its numbers don't reproduce when you re-run seed 0 yourself, retry one step up (haiku → sonnet → opus; low → medium → high) and note it in §10. If an effort level isn't available for a model, use the nearest one.

### Writing the brief (subagents start cold)
A subagent does not see this conversation, the files you have read, or earlier results. It does load CLAUDE.md, so the hard rules live there (setup below) and briefs stay short:
```
GOAL:       <one line>
READ FIRST: goal.md §3 and §6; <other files>
DO:         <steps or exact commands>
DON'T:      edit goal.md; touch files outside <list>; select anything using test metrics; add autograd to sparc.py
OUTPUT:     <files> + report of ≤10 lines: exact command, numbers copied from the JSON, any ambiguity resolved
DONE WHEN:  <check>
```

### Rules
1. **One writer per file.** Only the main session edits goal.md and merges into sparc.py. Subagents write only their own files (results/<run>.json, figs/, their worktree).
2. **Trust files, not prose.** Log a number only after opening its JSON. Re-run seed 0 yourself for any result that changes a decision.
3. **Commit before spawning variants.** Worktree subagents branch from the default branch, not your current HEAD, so uncommitted work is invisible to them. Commit after every iteration anyway (message = the log line).
4. **CPU is the real limit.** Concurrent bench processes ≤ CPU cores, each with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Claude Code allows 20 concurrent subagents by default, far more than a laptop can feed.
5. **No nesting.** The agent files below leave out the Agent tool, so only the main session spawns; cost stays predictable.
6. **Background by default.** Spawned subagents run in the background. Keep working (draft the next hypothesis) and never guess a pending result.
7. **Batch small things.** One runner with 12 commands beats 12 agents with one each.
8. **Don't force one model globally.** Leave `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` and `CLAUDE_CODE_EFFORT_LEVEL` unset, or they override the per-role settings.
9. **Forks** (`/subtask`) inherit the whole conversation: use one only when a subtask truly needs that context; it is not a way to save cost.

### Setup (It 0, step 1)
1. `git init` and commit goal.md (worktree isolation needs git).
2. Create `CLAUDE.md` (every subagent loads it):
   ```markdown
   # Project rules (all agents)
   - Source of truth: goal.md. Only the main session edits it.
   - SPARC stays local: no autograd, no backprop through layers, batch 1, single pass, no raw-image storage.
   - Tune on the validation split only (--split val). The test set is for logged results.
   - Touch only the files your task names. Report numbers copied from results JSON, never estimated.
   ```
3. Create the six files below in `.claude/agents/` (each code block is the whole file, frontmatter on line 1) and check they appear in `/agents`. Per call, the Agent tool's `model` (sonnet / opus / haiku / fable) and `effort` parameters override a file's settings.

`.claude/agents/builder.md`
```markdown
---
name: builder
description: Implements one clearly specified module of the brain-like learning project (data, baselines, metrics, bench harness, plots) and smoke-tests it. Not for design decisions.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
effort: medium
---
Implement exactly the spec in the task and touch only the files it names. Run a smoke test before returning.
Report: files changed, test command and its last 10 output lines, any ambiguity you resolved.
```

`.claude/agents/variant.md`
```markdown
---
name: variant
description: Implements ONE candidate change to sparc.py in an isolated worktree, runs the quick bench and reports numbers plus the diff. Use to test competing hypotheses in parallel.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
effort: high
isolation: worktree
---
Implement only the hypothesis in the task, keeping every P1–P7 rule in goal.md §3.
Run: python bench.py --model sparc --data mnist --seed 0 --split val --quick
Report: worktree path, diff summary, exact command, results JSON path, ACC / FGT / sparsity copied from the JSON, one sentence on what went wrong if anything. Merge nothing.
```

`.claude/agents/sweeper.md`
```markdown
---
name: sweeper
description: Runs a SPARC hyperparameter grid on the validation split and returns the full table. Never selects using test metrics.
tools: Read, Write, Bash, Glob
model: sonnet
effort: medium
---
Run the grid in the task with --split val only, at most the number of parallel processes the task allows, each with OMP_NUM_THREADS=1.
Write results/sweep_<name>.csv. Report its path and the top 5 rows. Change no code defaults: the main session picks.
```

`.claude/agents/runner.md`
```markdown
---
name: runner
description: Runs a given list of bench.py commands with limited parallelism, retries a crashed run once and returns a results table copied from the JSON files. Never edits code.
tools: Bash, Read, Glob
model: haiku
effort: low
---
Run the commands in the task, at most N at a time (N is in the task), each with OMP_NUM_THREADS=1.
If a run crashes, rerun it once; if it crashes again, report its last 20 error lines.
Return a markdown table: command, ACC, FGT and the other metrics, copied from each JSON, never estimated.
```

`.claude/agents/scout.md`
```markdown
---
name: scout
description: Answers one narrow literature question (prior work, published baseline numbers) with links. Says "not found" instead of guessing.
tools: WebSearch, WebFetch, Read
model: haiku
effort: low
---
Answer only the question asked, in at most 8 bullets, with a URL for every claim.
When sources disagree, give each number with its setting (online vs offline, buffer size, class- vs task-incremental).
```

`.claude/agents/auditor.md`
```markdown
---
name: auditor
description: Fresh-eyes reviewer. Checks that SPARC obeys the locality, online and no-raw-storage rules and that every reported number matches a results JSON. Edits nothing but its report.
tools: Read, Grep, Glob, Bash
model: sonnet
effort: high
---
Assume nothing. Check: no autograd or gradient libraries and no backprop through layers in sparc.py; batch 1, single pass, no shuffling; no raw-input buffers; fixed neuron budget; test set never used for selection; every number in goal.md §10 (and ARCHITECTURE.md if present) matches a results JSON.
Write audit.md with PASS/FAIL per item and file:line evidence.
```

### Example fan-out for one iteration
```
It 3 — "add a V1 layer"
main      writes hypothesis + interface spec, commits
 ├─ variant A  (sonnet/high, worktree)  learned competitive-Hebbian dictionary
 └─ variant B  (sonnet/high, worktree)  innate Gabor bank
main      opens both JSONs, merges the winner, commits, logs in §10
 └─ runner     (haiku/low)              2 datasets × 3 seeds → results/*.json   (only if quick bench passed)
main      updates §10
 └─ auditor    (sonnet/high)            M1 audit of the merged code
```

## References
- Shen, Dasgupta, Navlakha — Algorithmic insights on continual learning from fruit flies: https://arxiv.org/abs/2107.07617
- Zou, Zang, Ji — Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma: https://arxiv.org/abs/2502.01427
- Grossberg — Adaptive resonance theory (Scholarpedia): https://doi.org/10.4249/scholarpedia.1569
- da Silva et al. — A survey of ART neural network models: https://arxiv.org/abs/1905.11437
- Masuyama et al. — ART-based topological clustering with self-adjusting vigilance: https://arxiv.org/abs/2511.17983
- Journé et al. — Hebbian deep learning without feedback (SoftHebb): https://arxiv.org/abs/2209.11883
- Tadros et al. — Sleep-like unsupervised replay reduces catastrophic forgetting: https://doi.org/10.1038/s41467-022-34938-7
- Kapoor et al. — HiCL: Hippocampal-inspired continual learning: https://arxiv.org/abs/2508.16651
- Google Research — Introducing Nested Learning: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- Kosowski et al. — The Dragon Hatchling (BDH): https://arxiv.org/abs/2509.26507
- Hua et al. — Hebbian learning with global direction: https://arxiv.org/abs/2601.21367
- Claude Code docs — Subagents (frontmatter, model and effort order, background runs, worktree isolation): https://code.claude.com/docs/en/sub-agents
