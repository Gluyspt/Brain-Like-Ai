"""Stage-3 benchmark: transitive inference (docs/roadmap.md stage 3, research/17).

python tibench.py --model lookup|mlp|sparc_ti --seed 0 [--split val]

Seven arbitrary, visually distinct objects (Fashion-MNIST classes trouser, dress, coat, sandal, bag, boot, T-shirt; fresh images each trial) get a hidden rank A>B>…>G.
Training shows ONLY adjacent pairs (A?B, B?C, …), the model picks one and gets told which was higher.
Test: every pair, including non-adjacent pairs never shown (e.g. B?E). The classic animal/human result is
transitive choice on novel pairs and a symbolic-distance effect (far pairs easier).
"""
import argparse, json, os
import numpy as np
from data import load

N = 7
ITEMS = [1, 3, 4, 5, 8, 9, 0]          # Fashion classes used as the 7 objects (look-alikes pullover/shirt excluded)


class Lookup:
    def __init__(self, seed): self.rng = np.random.RandomState(seed); self.mem = {}
    def choose(self, i, j, xi, xj):
        k = (min(i, j), max(i, j))
        return self.mem[k] if k in self.mem else (i if self.rng.rand() < 0.5 else j)
    def feedback(self, i, j, xi, xj, winner): self.mem[(min(i, j), max(i, j))] = winner


class MLP:
    """Backprop MLP on one-hot (pair) inputs, trained online on the feedback (non-brain baseline)."""
    def __init__(self, seed, h=32, lr=0.2):
        r = np.random.RandomState(seed); self.W1 = r.randn(2 * N, h) * 0.3; self.W2 = r.randn(h) * 0.3; self.lr = lr
    def _x(self, i, j): x = np.zeros(2 * N); x[i] = 1; x[N + j] = 1; return x
    def _p(self, x): h = np.maximum(x @ self.W1, 0); return 1 / (1 + np.exp(-(h @ self.W2))), h
    def choose(self, i, j, xi, xj): return i if self._p(self._x(i, j))[0] > 0.5 else j
    def feedback(self, i, j, xi, xj, winner):
        x = self._x(i, j); p, h = self._p(x); g = p - (winner == i)
        gh = g * self.W2 * (h > 0); self.W2 -= self.lr * g * h; self.W1 -= self.lr * np.outer(x, gh)


def make(name, seed, ov):
    if name == 'lookup': return Lookup(seed)
    if name == 'mlp': return MLP(seed)
    if name == 'sparc_ti':
        from sparc_num import TISPARC
        return TISPARC(seed=seed, **ov)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='lookup'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val', choices=['val', 'test']); ap.add_argument('--trials', type=int, default=600)
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default=''); ap.add_argument('--out', default='results')
    a = ap.parse_args()
    from bench import parse_val
    ov = {k: parse_val(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    rng = np.random.RandomState(a.seed)
    Xtr, ytr, Xev, yev = load('fashion', a.split)
    order = rng.permutation(N)                      # object order[r] has rank r (0 = highest)
    rank = np.empty(N, int); rank[order] = np.arange(N)
    tr = {c: np.where(ytr == ITEMS[c])[0] for c in range(N)}; ev = {c: np.where(yev == ITEMS[c])[0] for c in range(N)}
    img = lambda pool, X, c: X[pool[c][rng.randint(len(pool[c]))]][None]
    model = make(a.model, a.seed, ov)
    if hasattr(model, 'setup'):
        model.setup(Xtr, ytr, rng)
    adj = [(order[r], order[r + 1]) for r in range(N - 1)]
    learn_curve = []
    for t in range(a.trials):
        i, j = adj[rng.randint(len(adj))]
        if rng.rand() < 0.5: i, j = j, i
        xi, xj = img(tr, Xtr, i), img(tr, Xtr, j)
        c = model.choose(i, j, xi, xj)
        win = i if rank[i] < rank[j] else j
        learn_curve.append(c == win)
        model.feedback(i, j, xi, xj, win)
    res = dict(model=a.model, seed=a.seed, split=a.split, overrides=ov, tag=a.tag,
               train_last100=float(np.mean(learn_curve[-100:])))
    acc = {}
    for d in range(1, N):
        hits = []
        for r in range(N - d):
            for _ in range(20):
                i, j = order[r], order[r + d]
                if rng.rand() < 0.5: i, j = j, i
                hits.append(model.choose(i, j, img(ev, Xev, i), img(ev, Xev, j)) == (i if rank[i] < rank[j] else j))
        acc[d] = float(np.mean(hits))
    res['by_distance'] = acc
    res['TI_adjacent'] = acc[1]
    res['TI_novel'] = float(np.mean([acc[d] for d in range(2, N)]))
    # the key inner pairs (exclude the end items A and G, whose answers are trivially "always/never chosen")
    inner = []
    for r1 in range(1, N - 1):
        for r2 in range(r1 + 2, N - 1):
            for _ in range(20):
                i, j = order[r1], order[r2]
                if rng.rand() < 0.5: i, j = j, i
                inner.append(model.choose(i, j, img(ev, Xev, i), img(ev, Xev, j)) == (i if rank[i] < rank[j] else j))
    res['TI_novel_inner'] = float(np.mean(inner))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ti_{a.model}{'_' + a.tag if a.tag else ''}_{a.seed}_{a.split}.json")
    json.dump(res, open(path, 'w'), indent=1)
    print(f"train {res['train_last100']:.2f}  adjacent {acc[1]:.3f}  novel {res['TI_novel']:.3f}  novel-inner {res['TI_novel_inner']:.3f}  by distance "
          + ' '.join(f'{d}:{acc[d]:.2f}' for d in acc) + f'  wrote {path}')


if __name__ == '__main__':
    main()
