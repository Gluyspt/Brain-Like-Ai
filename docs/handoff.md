# Handoff (2026-11-28): cloud session → your laptop

Read this first in the new session, then `CLAUDE.md`, `docs/rules.md`, `goal.md`, and the last ~15 rows of `docs/iteration-log.md`.

## Where things stand
- **Stage 1** (perception + memory): test MNIST 94.7 / FGT 0.7, Fashion 81.5 / 7.6. Open MUSTs: MNIST few-shot (43.6 test vs 70) and old-class drop (< 2). Relearning is the active problem (round 4).
- **Stage 2** (prediction): all four criteria pass on test with the continuous stream (S2-F2: online 0.924, retention 0.863, violation 0.951, completion5 0.860). Formally done once stage 1 passes.
- **Stage 3** (structure): number sense, transitive inference, list linking, same/different, **schema assimilation** (S3-5, adopted).
- Rules added this session (all in `docs/rules.md`): copy the function, not the limitation; dormant, not rejected (+ combination round); comparisons always old vs new, labelled; human comparison (two questions, `docs/human-signatures.md`); Marr's levels (problem strictly, algorithm closely, implementation freely).

## Round 4 (relearning) so far
| Finding | Row |
|---|---|
| Relearning lowers accuracy because homeostatic excitability drifts while awake | R4-diag, R4-diag2 |
| **SHY sleep** (`ie_sleep=down`): first real savings (Fashion 73.5 → 91.0), MNIST FEW5 41.2 → 69.6 (val, 3 seeds); cost: old-class drop 5.5 / 3.7 | R4-2, R4-4 → **candidate** |
| Sleep vs wake: SPARC's sleep harmed a just-relearned memory (opposite of humans) | R4-5 |
| Cause: sleep's IT merging compresses the freshly relearned units | R4-diag3 (`tools/sleepdiag.py`) |
| **merge_protect=used**: sleep loss gone (sleep − immediate −4.0 → +1.0); a day awake still beats sleep | R4-7 → **candidate** |
| Dormant (kept, flag off): early LTP, outcome-error writes, recency-first replay, slow excitability recovery, spacing has no effect | R4-3, R4-4, R4-6 |

## Next steps, in order (nothing is running; all results are committed)
1. ~~Finish R4-7~~ done in the cloud: 'used' beats 'new' (Fashion s1: sleep − immediate +0.5 vs −1.5), so use `merge_protect=used`.
2. ~~R4-8~~ **done on the laptop**: the pair are partners (Fashion sleep > wake in all 3 seeds, ACC F 83.5 / M 96.3), but old-class drop F 4.6 / M 4.9 still fails; see the R4-8 row in `docs/iteration-log.md`. Original plan: **R4-8: 2×2 SHY × merge_protect** (the two candidates may be partners: SHY fixes relearning, merge_protect fixes sleep). Arms: neither (exists), SHY only (exists: `r4_iedown`, `r4_5_sw_shy`), protect only (R4-7), **both** (to run): `--set ie_sleep=down merge_protect=used`, `--only main,few,life,sleepwake`, val, Fashion seeds 0–2 + MNIST seeds 0–2. Judge with old vs new tables and the two human questions. Target: savings, sleep ≥ immediate, old-class drop < 2.
3. **Old-class drop under SHY** (the blocker for adopting SHY): diagnose like R4-diag3. On a copy after few-shot learning, compare excitability (`r`, `r_eff`) of old- vs new-class units and which units win old-class test images.
4. **Fair re-test of early LTP** (dormant) with its partner: lock in only what the hippocampus replays in sleep (selective capture), not every change.
5. **Numba conversion of the inner loop** (`docs/todo.md`, "Faster code, same results"): convert `_learn_view` hot parts and V1 encode per view; **exact-match check** of every number in a results JSON before/after (the way shared training was checked: `--only main,few,space --quick` MNIST seed 0). Time it with no other runs going.
6. **Combination round** for stage 1 once one of the candidates is adopted: all adopted + dormant flags on, then leave-one-out.
7. Queue after that: `research/10-experiment-backlog.md` (P1 delayed relearning check, P2 context inference for stage 1, P3 adaptive teacher vs curious learner, BTSP one-shot binding, hippocampal behavioural tagging, ...).
8. Paper (user's question, 2026-11-28): estimate 3–6 weeks to a preprint / workshop paper (claim: one local-learning system with continual learning, relearning and relational transfer with human-like signatures; scaling as future work). Not started; user declined a published-work section in ARCHITECTURE.md for now.

## How to run on the laptop (Windows)
- Set up on 2026-11-28: project venv `.venv` (numpy, numba, scipy; numba needs scipy for `np.dot`), all three datasets cached, checks pass (B4 0.8118, SPARC quick 0.9422). Always run through `.\.venv\Scripts\python.exe` (`CLAUDE.md`; steps in `docs/todo.md`, "Moving to your own PC").
- Parallel runs: `.\.venv\Scripts\python.exe tools/run_list.py runs/<list>.txt -j <n>` (no xargs needed; one thread per job; retries once). Laptop: 8 physical / 16 logical cores, 14 GB RAM; 6 at once leaves room.
- Smoke: `python bench.py --model sparc --data mnist --seed 0 --split val --quick` → ACC ≈ 0.94.
- Full `--only main,few,life,sleepwake` run ≈ 13–15 min per run on one core (cloud, 4 in parallel).
- Commit on the working branch; fast-forward `main` when a task is complete; never force-push `main`.
