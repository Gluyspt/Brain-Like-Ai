# Stage-2 benchmark: prediction over time (`seqbench.py`)

Sequence worlds built from MNIST images. Each world is a hidden rule over digit classes:

| World | Rule | Order |
|---|---|---|
| W1_count | next = cur + 1 mod 10 | 1 |
| W2_plus3 | next = cur + 3 mod 10 | 1 |
| W3_perm | next = π(cur), a fixed random permutation | 1 |
| W4_fib | next = prev + cur mod 10 | 2 |

**Stream:** the worlds are learned one after another, each for `--steps` items (default 3000), online and single pass, as episodes of `--episode` items (default 30) with random starts. Since 2026-11-27 (S2-8) the episodes follow each other **as one continuous stream with no signalled boundary** (`--boundaries none`, default; real experience has no reset signal). `--boundaries given` restores the old `reset_context` between episodes. Before 2026-11-26 each world was one continuous sequence; that visited only 7 (W3) and 3 (W4) of the digit pairs, so test sequences asked about transitions never shown (benchmark flaw, fixed). Every step shows a fresh training image of the current class with its label. Before each item the model must predict the next class (`predict_next`).

**Metrics** (results/seq_<model>[_<tag>]_<seed>_<split>.json):
- **online_end_acc**: next-item accuracy over the last 500 steps of each world while learning it.
- **retention**: after all worlds, 20 short test sequences (30 items) per world on evaluation images, with learning off (`observe_frozen`). The first 3 predictions are a warm-up (the model must infer which world it is in from recent history), and deviant items are excluded.
- **violation_auroc**: in the same test sequences, 10 % of items are replaced by a random wrong class. AUROC of `surprise_last` for deviant vs regular items.
- **completion**: given 2 items, the model rolls out 8 steps, feeding back an image of its own predicted class (imagination). Accuracy of the rolled-out classes. With 2 items the world is often ambiguous, so even a perfect learner is limited; **completion5** gives 5 items first (world identifiable) and is the real imagination test.

**Baselines:** chance, `ngram1`, `ngram2`: counting tables on the true labels (idealised statistical learners, not brain-like).

| model (val, seed 0, 3000 steps) | online | retention | violation | completion |
|---|---|---|---|---|
| chance | 0.142 | 0.105 | 0.500 | 0.100 |
| ngram1 | 0.538 | 0.287 | 0.734 | 0.234 |
| ngram2 | 0.963 | 0.614 | 0.855 | 0.186 |

**Stage-2 pass criteria (proposed):** mean online ≥ 0.90; mean retention ≥ 0.60 (at least the ngram2 level, without storing sequences); violation AUROC ≥ 0.85; completion from 5 items ≥ 0.70; stage-1 benchmark still passing. (The 2-item completion is reported but has no target: it is ambiguity-limited.)
