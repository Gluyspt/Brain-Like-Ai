# Loop protocol

## Round 2 (from 2026-11-26): the improved loop

Lessons from round 1 (docs/iteration-log.md, research/12) and the change each one caused:

| Round-1 lesson | Change in round 2 |
|---|---|
| Diagnostic runs reused a tag and overwrote each other (audit) | every run gets a **unique tag**: `r2_<iter>_<variant>` |
| MNIST-only quick runs hid Fashion problems until late (It 5) | the quick screen runs **MNIST + Fashion + CIFAR-10** in parallel |
| Settings tuned on seed 0 only (warning W6) | an adopted change must hold on **val seeds 0–2** before the test bench |
| MNIST alone says little about scale (warning W1) | CIFAR-10 is a tracked target, not a one-off check |
| Changes were adopted for score (It 6 named no mechanism at first) | each row names the brain mechanism + research/ file + cost, **before** running |
| Ablations ran once, on an older config | ablations re-run at the end of the round on the final config |
| Work only reached `main` at the end | merge to `main` after every adopted iteration (docs/rules.md) |

### Each iteration
1. **Hypothesis** (main session): mechanism → the failure it should fix → which metric should move. Write the log row first, with the mechanism, its research/ file and the expected cost.
2. **Implement** behind a flag that is off by default. Check that MNIST predictions are unchanged with the flag off.
3. **Screen**: `--quick --split val --seed 0 --only main,few` on mnist, fashion and cifar10 in parallel (3 processes).
4. **Confirm** (only for a candidate to adopt): full stream on val, seeds 0–2, both MNIST and Fashion. Adopt only if it helps on average and breaks no passing criterion.
5. **Log** in docs/iteration-log.md from the JSON files; commit, push, merge to `main`.
6. **Diagnose** → next hypothesis.

### Round-2 targets (in addition to goal.md §4)
- Close the three open MUSTs: Fashion FGT ≤ 10, Fashion few-shot old-class drop ≤ 2, MNIST FEW-5 ≥ 70.
- CIFAR-10 (val): ACC within 5 pts of joint training (B3 = 47.7) and NOVELTY ≥ 0.70.

### Budget and stop
At most **8 iterations** in round 2. Stop early when all targets pass on the full test bench (3 seeds), or when two iterations in a row give no gain. At the end: ablations on the final config, the auditor, and ARCHITECTURE.md updated.

### Planned order (deviate when the data says so)
- **R2-1 V2 layer** (research/10 item 3): a second competitive-Hebbian layer on locally pooled V1 codes (simple → complex → simple, Hubel & Wiesel), with a later critical period than V1.
- **R2-2** top-down feedback into V1/V2 (predictive coding) or IT-level sleep replay, whichever the R2-1 diagnosis points to.

## Round 1 (2026-11-25/26, done: It 0–12)
Each iteration:
1. **Hypothesis**: which brain mechanism should fix which observed failure (one sentence). Main session only.
2. **Implement** the smallest change. One change per iteration. When two or three fixes compete, give each to a `variant` agent in parallel (docs/agents.md) and keep the winner.
3. **Run** quick bench (MNIST, seed 0). Full bench (2 datasets × 3 seeds) only once quick passes, handed to a `runner` agent.
4. **Log** the numbers against the success criteria (goal.md §4) in docs/iteration-log.md, after opening the JSON files yourself.
5. **Diagnose** → next hypothesis. Main session only.

Round-1 order: It 0 setup and baselines; It 1 IT and fast readout; It 2 budget control; It 3 V1; It 4 hippocampus and sleep; It 5 few-shot; It 5b self-correction; It 6+ capacity, excitability, critical period, rethink loop. Stop rule: every MUST passes, or 12 iterations. Round 1 stopped at 12.
