# Brain-Like-Ai

SPARC: a learning architecture built only from brain mechanisms (local plasticity, sparse codes, a hippocampus, sleep replay, self-correction), with no backprop. It is tested on class-incremental image streams (MNIST, Fashion-MNIST, CIFAR-10).

| Start here | |
|---|---|
| `goal.md` | the goal, what "learns like a brain" means (P1–P10), success criteria, status |
| `ARCHITECTURE.md` | the current design, equations, results and honest gaps |
| `docs/` | rules, benchmark protocol, loop, iteration log, roadmap, agents, environment, audit |
| `research/` | brain systems, prior attempts and warnings, experiment backlog, glossary |

Code: `sparc.py` and `v1.py` (model); `bench.py`, `data.py` and `baselines.py` (benchmark, the "world"); `tools/` (reports). Run lists are in `runs/`, outputs in `results/`.

```
python bench.py --model sparc --data mnist --seed 0 --split val --quick
python tools/summarize.py results/sparc_*_val.json
```
Needs numpy only. Data downloads are described in `docs/environment.md`.
