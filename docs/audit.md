# SPARC audit (auditor agent, sonnet/high, at commit 6ba0e2a)

Saved by the main session; the auditor had no Write tool. Condensed from its report; main-session responses in *italics*.

## 1. P1 locality — PASS (synaptic), FLAG (structural)
- No gradient libraries (sparc.py imports numpy + v1 only; v1.py numpy only). No error flows back into V1 or IT weights.
- Synaptic rules checked, all local: V1 winner update (v1.py:71-72), IT refine (sparc.py:149-153), recruitment gated by broadcast scalars (sparc.py:138-145, v1.py:66-69), fast readout row update (sparc.py:165), sleep delta rule (sparc.py:241-244, allowed), neuromodulator traces (sparc.py:121-125, 157-161).
- Note: confident-error LTD normalises by the max of active pre activity (sparc.py:196-197): a divisive input normalisation, not strictly pre/post. *Accepted as divisive normalisation; documented.*
- Structural, non-local (not synaptic weight updates): consolidate() water-filling + pairwise merges (sparc.py:265-331); hippocampal overwrite of the most-represented class (sparc.py:210-213); _sleep_clean() (sparc.py:249-263); match tracking compares a full readout prediction to the label and broadcasts one scalar (sparc.py:128-137). *Declared as structural plasticity / memory management in ARCHITECTURE.md limits.*
- Minor bug: empty hippocampal slots map to unit 0; `R[np.ix_(...)] +=` does not accumulate duplicates, so a real unit-0 entry can lose its update in sleep (sparc.py:206, 244). *Fixed after the final runs (np.add.at); final numbers are from 6ba0e2a.*
- end_task() gives training-time task boundaries (allowed by docs/design-v1.md; no task labels at test).

## 2. P2 online — PASS
batch_size 1 (sparc.py:30, bench.py:34-47); single pass in dataset order (data.py:40,47-53); no shuffle/epochs; only sleep replay repeats (allowed). Eval data never reaches learn(); few-shot shots come from the train pool.

## 3. P7 fixed resources — PASS with caveat
No raw-input buffer; IT budget guard (sparc.py:138) and V1 guard (v1.py:66) hold in every JSON (it_used ≤ it_budget). Hippocampus = k idx + k val + label; stored_numbers 31,000 = 1000·31 correct; params 3,133,600 = 3000·1024 + 2·10·3000 + 64·25 correct.
Caveat: IT prototypes are one-shot imprints of V1 codes (3.07M numbers, an exemplar-like store in feature space), declared as parameters. `raw_images_stored=0` is a constant. *Documented in limits.*

## 4. Test set never used for selection — PASS
All tuning JSONs are split=val; test runs only for tag `final` with the It 6 config chosen on val. Note: the train stream is the 54k subset for both splits.

## 5. Iteration-log numbers vs JSON
Matches for rows 0, 1 (main), 2 (grid), 3, 4, 5, 5b, 6.
FAIL / unverifiable (no JSON): It 1 n_it=6000 → 93.2/3.6, refine=False → 79.4/10.4, "full at 12.5k", "units/class 365…50" (tag `diag` was reused and overwritten); It 2 "(c) without fairness 70–77, class 1 → 6 units" (overwritten by the fair version, same tag); It 5 "shots recruit 6/10", "NCM 58 %", "NCM proxy 60→67 %" (printed by scratch scripts only). *Marked in docs/iteration-log.md as "console only, not in JSON".*
Minor: REV examples 30/10 are 30.5/10.5 in JSON; rounding of 92.35 → 92.4 and 94.35 → 94.4. *Corrected.*
Reproducibility: It 2 turnover flags were removed in the It 4 rewrite. *Noted.*

## 6. Protocol (bench.py vs docs/benchmark.md)
ACC, FGT, LBL5, NOVELTY (AUROC checked against brute force) PASS. FEW-k: sleep is called after the shots; old baseline is 8-way, post-shot 10-way; single draw per seed. REVERSAL/NOISE deviate from the benchmark wording: only first halves of tasks 1–2 are learned before the swap / noise, eval is 4-way after task 2, 20-sample grid. *docs/benchmark.md now describes the implemented protocol.*
Sparsity: V1 figure is the learned per-position k-WTA code; the fixed sum-pooled vector feeding IT is denser. *Both reported in ARCHITECTURE.md.*

## 7. Brain-first rule
It 6 (N 1500 → 3000) named no mechanism and no cost. *Added to docs/iteration-log.md: larger cortical population (capacity); cost params 1.19M → 3.13M, ~1.6× wall-clock.*
