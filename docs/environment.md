# Environment notes
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
