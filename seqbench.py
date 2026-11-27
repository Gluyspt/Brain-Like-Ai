"""Stage-2 benchmark: prediction over time in sequence worlds made of MNIST images (docs/benchmark-stage2.md).

python seqbench.py --model sparc|ngram1|ngram2|chance --seed 0 --split val [--steps 3000] [--set key=val ...] [--tag t]

Each world is a hidden rule over digit classes; each step shows a fresh image of the current class with its label.
The model predicts the next class BEFORE seeing it. Worlds are learned one after another, online, single pass.
Interface: model.observe(x, y) -> learns; model.predict_next() -> (10,) probabilities; model.surprise_last() -> float
(surprise at the item just observed, computed before learning from it); model.reset_context() between sequences.
"""
import argparse, json, os, time
import numpy as np
from data import load

PERM = np.random.RandomState(7).permutation(10)
WORLDS = {
    'W1_count': lambda p, c: (c + 1) % 10,
    'W2_plus3': lambda p, c: (c + 3) % 10,
    'W3_perm': lambda p, c: int(PERM[c]),
    'W4_fib': lambda p, c: (p + c) % 10,          # second order: needs the previous item too
}


def gen(rule, n, rng, deviant=0.0):
    """Class sequence of length n from a rule; with prob `deviant` an item is replaced by a random other class."""
    p, c = rng.randint(10), rng.randint(10)
    seq, dev = [p, c], [False, False]
    for _ in range(n - 2):
        nxt = WORLDS[rule](p, c)
        is_dev = rng.rand() < deviant
        if is_dev:
            nxt = int(rng.choice([k for k in range(10) if k != nxt]))
        seq.append(nxt); dev.append(is_dev)
        p, c = c, nxt
    return np.array(seq), np.array(dev)


class Pool:
    """Draw images of a class without replacement (wraps around if exhausted)."""
    def __init__(self, X, y, rng):
        self.X, self.idx, self.ptr, self.rng = X, {c: rng.permutation(np.where(y == c)[0]) for c in range(10)}, {c: 0 for c in range(10)}, rng

    def draw(self, c):
        i = self.idx[c][self.ptr[c] % len(self.idx[c])]
        self.ptr[c] += 1
        return self.X[i:i + 1]


# ---------------- baselines ----------------
class NGram:
    """Counting n-gram on true labels (an idealised statistical learner; not brain-like). Order 1 or 2."""
    def __init__(self, order=1, **kw):
        self.order, self.T, self.hist = order, {}, []

    def reset_context(self):
        self.hist = []

    def _key(self):
        return tuple(self.hist[-self.order:]) if len(self.hist) >= self.order else None

    def predict_next(self):
        k = self._key()
        cnt = self.T.get(k) if k is not None else None
        if cnt is None or cnt.sum() == 0:
            return np.full(10, 0.1)
        return cnt / cnt.sum()

    def observe(self, x, y):
        k = self._key()
        self._s = 1.0 - self.predict_next()[y]
        if k is not None:
            self.T.setdefault(k, np.zeros(10))[y] += 1
        self.hist.append(int(y))

    def observe_frozen(self, x, y):
        self._s = 1.0 - self.predict_next()[y]
        self.hist.append(int(y))

    def surprise_last(self):
        return self._s


class Chance(NGram):
    def predict_next(self):
        return np.full(10, 0.1)


def make(name, seed, overrides):
    if name == 'sparc':
        from sparc_seq import SeqSPARC
        return SeqSPARC(seed=seed, **overrides)
    if name == 'sparc_ctx':
        from sparc_seq import CtxSeqSPARC
        return CtxSeqSPARC(seed=seed, **overrides)
    if name == 'ngram1':
        return NGram(1)
    if name == 'ngram2':
        return NGram(2)
    return Chance()


def auroc(neg, pos):
    s = np.concatenate([neg, pos]); r = np.empty(len(s)); r[s.argsort(kind='mergesort')] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_seq(model, pool, seq, learn=True, record=False, reset=True):
    """Present a class sequence; returns per-step (correct prediction, surprise). reset=False: no event
    boundary is given before it (the stream continues; the model must segment it itself)."""
    if reset:
        model.reset_context()
    hits, surp = [], []
    for t, c in enumerate(seq):
        if t >= 2:
            hits.append(int(np.argmax(model.predict_next()) == c))
        x = pool.draw(int(c))
        if learn:
            model.observe(x, int(c))
        else:
            model.observe_frozen(x, int(c)) if hasattr(model, 'observe_frozen') else model.observe(x, int(c))
        surp.append(model.surprise_last())
    return np.array(hits), np.array(surp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sparc')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--episode', type=int, default=30, help='items per training episode (event boundaries between)')
    ap.add_argument('--boundaries', default='none', choices=['given', 'none'],
                    help='none: training is one continuous stream, episode jumps are not signalled')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    from bench import parse_val
    ov = {k: parse_val(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    Xtr, ytr, Xev, yev = load('mnist', a.split)
    rng = np.random.RandomState(a.seed)
    train_pool, eval_pool = Pool(Xtr, ytr, rng), Pool(Xev, yev, np.random.RandomState(a.seed + 1))
    model = make(a.model, a.seed, ov)
    T0 = time.time()
    res = dict(model=a.model, seed=a.seed, split=a.split, steps=a.steps, overrides=ov, tag=a.tag, worlds=list(WORLDS),
               boundaries=a.boundaries)
    names = list(WORLDS)
    online = {}
    # 1) learn the worlds one after another (online, single pass). Each world is experienced as episodes of
    #    `--episode` items with random starts, separated by event boundaries (reset_context), so every part of
    #    the rule is visited (a single long sequence stays inside one cycle of the rule).
    for w in names:
        hits = []
        for _ in range(a.steps // a.episode):
            seq, _ = gen(w, a.episode, rng)
            h, _ = run_seq(model, train_pool, seq, learn=True, reset=a.boundaries == 'given')
            hits += list(h)
        online[w] = float(np.mean(hits[-500:]))
        print(f'  learn {w}: last-500 next-item acc {online[w]:.3f}', flush=True)
    res['online_end_acc'] = online
    # 2) retention: short test sequences from each world on eval images, learning off; first 5 items = context
    ret, viol, comp = {}, {}, {}
    erng = np.random.RandomState(a.seed + 2)
    for w in names:
        accs, neg, pos = [], [], []
        for _ in range(20):
            seq, dev = gen(w, 30, erng, deviant=0.1)
            hits, surp = run_seq(model, eval_pool, seq, learn=False)
            ok = ~dev[2:]
            ok[:3] = False                                   # warm-up: infer the world from recent history
            # predictions after a deviant are judged against the rule from the true (deviant) history
            accs.append(hits[ok].mean() if ok.any() else np.nan)
            neg += list(surp[2:][~dev[2:]][3:]); pos += list(surp[2:][dev[2:]])
        ret[w] = float(np.nanmean(accs))
        viol[w] = auroc(np.array(neg), np.array(pos)) if pos else None
        # 3) completion: given g items (g = 2: world often ambiguous; g = 5: world identifiable), roll out
        #    8 predicted classes, feeding back an image of the model's own prediction (imagination)
        for g in (2, 5):
          cacc = []
          for _ in range(20):
            seq, _ = gen(w, g + 8, erng)
            model.reset_context()
            for t in range(g):
                model.observe_frozen(eval_pool.draw(int(seq[t])), int(seq[t])) if hasattr(model, 'observe_frozen') else model.observe(eval_pool.draw(int(seq[t])), int(seq[t]))
            correct = 0
            for t in range(g, g + 8):
                pred = int(np.argmax(model.predict_next()))
                correct += pred == seq[t]
                x = eval_pool.draw(pred)                     # imagine the predicted item
                model.observe_frozen(x, pred) if hasattr(model, 'observe_frozen') else model.observe(x, pred)
            cacc.append(correct / 8)
          comp[w if g == 2 else w + '_from5'] = float(np.mean(cacc))
        print(f'  test {w}: retention {ret[w]:.3f}  violation AUROC {viol[w]}  completion {comp[w]:.3f} / from 5 items {comp[w + "_from5"]:.3f}', flush=True)
    res.update(retention=ret, violation_auroc=viol, completion=comp,
               mean_online=float(np.mean(list(online.values()))), mean_retention=float(np.mean(list(ret.values()))),
               mean_violation=float(np.mean([v for v in viol.values() if v is not None])),
               mean_completion=float(np.mean([comp[w] for w in names])),
               mean_completion5=float(np.mean([comp[w + '_from5'] for w in names])), wall_clock_s=time.time() - T0,
               n_seg=getattr(model, 'n_seg', None), n_ctx_new=getattr(model, 'n_ctx_new', None))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"seq_{a.model}{'_' + a.tag if a.tag else ''}_{a.seed}_{a.split}.json")
    json.dump(res, open(path, 'w'), indent=1)
    print(f"mean: online {res['mean_online']:.3f} retention {res['mean_retention']:.3f} "
          f"violation {res['mean_violation']:.3f} completion {res['mean_completion']:.3f} / from 5 {res['mean_completion5']:.3f}  wrote {path}")


if __name__ == '__main__':
    main()
