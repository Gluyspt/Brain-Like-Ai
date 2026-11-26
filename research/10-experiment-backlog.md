# 10 — Experiment backlog

Ordered by expected value (item 2, top-down bias, is now first in line; item 3, a V2 layer, is the most direct test of the single-layer limit) for the open gaps, measured at It 9 on the test split: MNIST FEW-5 is 43.7 (target 70); the few-shot old-class drop is 2.2 / 3.2 (target ≤ 2); Fashion FGT is 10.9 (target ≤ 10). Every item names its brain mechanism (brain-first rule).

| # | Experiment | Brain mechanism | Gap it targets | Cost | File |
|---|---|---|---|---|---|
| 1 | ~~Two-hemisphere V1, concatenated at IT~~: **rejected** (It 10: no few-shot gain, Fashion FGT +2–5). Variant still open: separate hemisphere IT populations with a learned gate | hemispheric specialisation + thalamic gating | Fashion look-alikes | medium | 04, 06 |
| 2 | **Top-down conflict bias** on IT competition, plus a "second look" at test time | PFC bias signals, predictive feedback | MNIST few-shot, old-class drop | medium | 05 |
| 3 | **Second learned layer (V2)**: now top priority (CIFAR-10: 36.1 % vs 47.7 joint, novelty 0.56) between V1 and IT, same competitive-Hebbian rule, its own critical period | cortical hierarchy (V1 → V2 → V4 → IT) | MNIST few-shot (single-layer limit) | medium–large | 01, 09 |
| 4 | **IT-level sleep replay** (re-align prototypes and readouts) | sleep replay, synaptic homeostasis | Fashion FGT | medium | 03 |
| 5 | **KC readout reserved for few-shot classes** | fly mushroom body | old-class drop | small | 02 |
| 6 | **Learned hippocampal write gate** | basal-ganglia gating, dopamine | old-class drop, memory efficiency | medium | 07 |
| 7 | **Separate ACh / NE modulators** | expected vs unexpected uncertainty | reversal speed, stability | small | 01 |
| 8 | **Temporal predictor** (cerebellum-like) | forward models | roadmap: sequences, better surprise | large | 08 |
| 9 | **Reward instead of labels** | dopamine reward-prediction error | roadmap: reinforcement | medium | 07 |

## How to run an item
1. Write the hypothesis in `goal.md` docs/loop.md style: mechanism → failure it should fix.
2. Implement it behind a flag that is off by default, and quick-test on `--split val`.
3. Log in `goal.md` docs/iteration-log.md with the brain mechanism and its cost.
4. Adopt it only if it helps without breaking the brain-first rule; then rerun the ablations and the 3-seed test bench.

## Done or rejected (for reference)
Turnover (rejected, unsafe; It 2); dense IT codes with k = 30/50 (rejected, worse; It 6); KC third vote (rejected, no gain; It 8); critical-period closure (adopted, It 9).
