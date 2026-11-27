# To-do: engineering, speed and running on your own PC

Research to-dos live in `research/10-experiment-backlog.md` and `docs/roadmap.md`. This file is for engineering.

## Speed (measured 2026-11-26, Fashion, eye movements on, 1 core under load)
| Step | Status | Effect |
|---|---|---|
| 1. numpy cleanup: V1 pooling as one matmul; V1 patches computed once per view and reused for learning and encoding | **done** (predictions identical) | learn ~2.6 → ~2.1 ms/image (~20 %) |
| 2. Numba JIT for the V1 per-patch learning loop (`v1._learn_patches`, falls back to plain Python without numba) | **done** (predictions identical) | small on top of step 1 |
| 3. Fewer, larger operations in `SPARC._learn_view` (precompute normalised readout rows, avoid repeated `_unit`/`norm` calls, process the 3 glimpses of an image together where the rule allows) | todo | est. 1.5–2× |
| 4. Numba for V1's per-position k-WTA and for the hippocampus/sleep loops | todo | est. small–medium |
| 5. **Rust core** (PyO3 extension) for the whole per-sample loop: V1 + IT + readouts + hippocampus | **later**, when the design stabilises, or at stages that need scale (natural images, language) | est. 10–30× per run |

Why per-run speed matters: SPARC learns one sample at a time (rule P2), so time goes into many small calls, not big matrix operations. Python overhead is most of the cost, which compiled code removes. Rule for every speed change: check that predictions are identical before and after (`regress.py`-style check: same seed, same stream, compare all predictions).

## Profile (2026-11-27, cProfile, Fashion, `--quick --only main`, eye movements on, 1 thread; 4 other runs on the 4 cores, so absolute times are inflated, shares are what matter)
Total 66.6 s, 8.7 M Python function calls, 30 000 views learned (10 000 images × 3 glimpses).

| Where | Time (cumulative) | Share | What it is |
|---|---|---|---|
| `SPARC._learn_view` (training, per view) | 44.7 s | 67 % | |
| ├ V1 encode of the view (`_pre` → `v1.encode`: patches, patch × dictionary matmul, per-position k-WTA via `np.partition`, pooling) | 22.3 s | 34 % | small matmul + partition per view |
| └ IT step (own time 20.1 s: matvecs W·x up to 4× per view, k-WTA, readouts, hippocampus) + V1 learning 2.3 s | ~22 s | 33 % | many small ops |
| evaluation (`SPARC.encode`, batched, 27 calls) | 16.7 s | 25 % | already batched |
| rest (setup, data) | ~5 s | 8 % | |

Small-op overhead is large: `norm` 161 k calls (3.5 s), `_unit` 108 k (3.1 s), `take/put_along_axis` 100 k (2 s), `partition` / `argpartition` 141 k (6.4 s). Per view ≈ 1.5 ms of training work spread over ~250 Python calls.

## GPU (CUDA) plan
No GPU on the cloud machine (no `nvidia-smi`). Verdict: **little gain for training** (batch 1, single pass: each step is ~1–2 ms of many small sequential ops; a GPU kernel launch costs ~5–20 µs each and the Python loop stays), **large gain for evaluation and for many runs at once**.
| Step | Gain on GPU | How |
|---|---|---|
| G1. Evaluation (25 % of a run) | large | CuPy drop-in for `SPARC.encode` / `predict` (already batched); keep weights on the GPU between evaluations |
| G2. Many models in lockstep (seeds, sweeps) | large | stack S models' weights (S, N, D) and step them together; batch 1 per model is kept (the batch rule is within one model) |
| G3. Training loop | ~none to 2× | only after steps 3–4 above; keep V1 + IT of one model on the GPU to avoid transfers |
| G4. Scale (N ≥ 30 k IT units, larger images, CIFAR at scale) | grows with size | per-step matmuls become big enough to fill the GPU |
Order on your own PC: speed steps 3 → 4 first (CPU, helps every machine), then G1, then G2 if you sweep a lot. Install: `pip install cupy-cuda12x`; use `xp = cupy if available else numpy` in `v1.py` / `sparc.py`; check predictions identical (within float tolerance) before/after.

## Moving to your own PC
1. **Get the code**: `git clone` the repo; work on a branch; `main` holds the merged state.
2. **Python**: 3.11+ with `numpy` (required) and `numba` (optional, faster). No GPU and no torch needed. `pip install numpy numba`.
3. **Data** (`data_cache/`, not in git; `data.py` downloads MNIST and Fashion-MNIST automatically):
   - MNIST: `https://raw.githubusercontent.com/fgnt/mnist/master/<file>`
   - Fashion: `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/<file>`
   - CIFAR-10: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` (or the mirror `https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz`) into `data_cache/cifar_dl/`; check md5 `c58f30108f718f92721af3b95e74349a`. The loader uses a restricted unpickler, so keep it that way.
4. **Cores**: the cloud machine had 4. Run one bench process per core, each with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, e.g. `xargs -P <cores> -I{} sh -c '{}' < runs/<list>.txt`. More cores = more experiments in parallel.
5. **Expected times** (one core): full stage-1 bench run (main, few, lbl, rev) ≈ 20–25 min with eye movements, ≈ 10 min without; quick screen ≈ 2–4 min; stage-2 `seqbench.py` ≈ 5–15 min.
6. **Claude Code locally**: open the repo folder. `CLAUDE.md` and `.claude/agents/*.md` load automatically (agent types register at session start). Rules: `docs/rules.md`; loop: `docs/loop.md`.
7. **Check the setup**: `python bench.py --model b4 --data mnist --seed 0 --split val` should give ACC ≈ 0.81 in seconds; `python bench.py --model sparc --data mnist --seed 0 --split val --quick` gives ACC ≈ 0.95.

## Other engineering to-dos
- Result files keep growing (~300 JSONs): consider `results/<round>/` subfolders and an index.
- Re-run the full ablation table for the current config once runs are faster (skipped in round 2 for time).
- `tools/confirm.py` is hard-wired to the It 12 baseline tags; generalise it.
