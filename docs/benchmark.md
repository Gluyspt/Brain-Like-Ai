# Benchmark protocol (`bench.py`)
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
- **LIFE** (`--only life`, research/15), lifelike memory. Each class's train images split 90 % stream / 5 % reminders / 5 % relearning (never reused).
  - **REMIND**: the 5-task stream with ~2 % labelled reminders of earlier classes inserted at random; report REMIND_ACC and REMIND_FGT.
  - **SAVINGS**: plain stream; first-learning curves on a 200-image probe per task (every 50 samples). Afterwards pick the most-forgotten task among 1–4, relearn it from its relearning images (batch-size steps, probe every 10). Report SAV_n_first and SAV_n_relearn (examples to reach 95 % of the task's old peak), SAV_ratio = first / relearn (> 1 means savings), SAV_relearned_to and SAV_other_drop.

