"""Stage-3 benchmark: number sense (docs/roadmap.md stage 3, research/20).

python numbench.py --model lookup|mlp_onehot|mlp_thermo|sparc_num --seed 0 [--split val]

Three tests, each separating memorising from understanding:
  1. GROUND   dot-array images (1–9 dots) are paired with digit images during training; test: a new dot array →
              which digit? (symbol grounding; 9-way)
  2. ORDER    learned only from counting sequences (1,2,3,…); test: which of two digits is larger, for every pair
              (never trained), and the distance effect (accuracy for far vs close pairs)
  3. ADD      "a, b → a+b" (1 ≤ a, b; a+b ≤ 9); train on a subset of pairs, test on HELD-OUT pairs
              (a lookup table is at chance on held-out pairs; structure is needed to generalise)
Models get class labels for training items (they recognise digits, stage 1) and must learn the relations.
"""
import argparse, json, os, time
import numpy as np


# ---------------- dot arrays ----------------
def dot_image(n, rng, side=28):
    """n non-overlapping dots of random radius (1.5–3 px) at random positions; values in [0, 1]."""
    img = np.zeros((side, side), np.float32)
    yy, xx = np.mgrid[0:side, 0:side]
    centres = []
    tries = 0
    while len(centres) < n and tries < 1000:
        tries += 1
        r = rng.uniform(1.5, 3.0)
        cy, cx = rng.uniform(r + 1, side - r - 1, size=2)
        if all((cy - y) ** 2 + (cx - x) ** 2 > (r + rr + 1.5) ** 2 for y, x, rr in centres):
            centres.append((cy, cx, r))
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1.0
    return img.reshape(1, -1)


PAIRS = [(a, b) for a in range(1, 9) for b in range(1, 9) if a + b <= 9]      # 36 pairs


def split_pairs(rng):
    """Hold out 1/3 of the addition pairs (stratified by sum where possible)."""
    p = list(PAIRS)
    rng.shuffle(p)
    k = len(p) // 3
    return p[k:], p[:k]


# ---------------- baselines (operate on true class labels; not brain-like) ----------------
class Lookup:
    """Memorises training facts; guesses uniformly otherwise."""
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed); self.facts = {}; self.dot = {}

    def train_ground(self, dots_n, digit):    self.dot[dots_n] = digit      # oracle: knows dot count
    def ground(self, dots_n):                 return self.dot.get(dots_n, self.rng.randint(1, 10))
    def train_count(self, seq):               self.succ = {seq[i]: seq[i + 1] for i in range(len(seq) - 1)}
    def larger(self, a, b):                   return a if self.rng.rand() < 0.5 else b     # no comparison facts
    def train_add(self, a, b, s):             self.facts[(a, b)] = s
    def add(self, a, b):                      return self.facts.get((a, b), self.rng.randint(2, 10))


class MLP:
    """Backprop MLP baseline for ADD (one-hot or thermometer inputs) and ORDER (trained on nothing: chance)."""
    def __init__(self, seed, code='onehot', hidden=64, epochs=300, lr=0.1):
        self.rng = np.random.RandomState(seed); self.code, self.h, self.epochs, self.lr = code, hidden, epochs, lr
        self.X, self.Y = [], []; self.dot = {}

    def _enc(self, v):
        e = np.zeros(9, np.float32)
        if self.code == 'onehot':
            e[v - 1] = 1
        else:
            e[:v] = 1                                    # thermometer = number line given for free
        return e

    def train_ground(self, dots_n, digit):    self.dot[dots_n] = digit
    def ground(self, dots_n):                 return self.dot.get(dots_n, self.rng.randint(1, 10))
    def train_count(self, seq):               pass
    def larger(self, a, b):
        if self.code == 'thermo':                        # thermometer codes make magnitude readable
            return a if self._enc(a).sum() > self._enc(b).sum() else b
        return a if self.rng.rand() < 0.5 else b
    def train_add(self, a, b, s):             self.X.append(np.concatenate([self._enc(a), self._enc(b)])); self.Y.append(s)

    def fit(self):
        X = np.array(self.X); Y = np.array(self.Y)
        W1 = self.rng.randn(18, self.h) * 0.3; b1 = np.zeros(self.h); W2 = self.rng.randn(self.h, 10) * 0.3; b2 = np.zeros(10)
        for _ in range(self.epochs):
            H = np.maximum(X @ W1 + b1, 0); Z = H @ W2 + b2; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            G = P.copy(); G[np.arange(len(Y)), Y] -= 1; G /= len(Y)
            gW2 = H.T @ G; gH = G @ W2.T * (H > 0); gW1 = X.T @ gH
            W2 -= self.lr * gW2; b2 -= self.lr * G.sum(0); W1 -= self.lr * gW1; b1 -= self.lr * gH.sum(0)
        self.W = (W1, b1, W2, b2)

    def add(self, a, b):
        if not hasattr(self, 'W'):
            self.fit()
        W1, b1, W2, b2 = self.W
        x = np.concatenate([self._enc(a), self._enc(b)])
        return int(np.argmax(np.maximum(x @ W1 + b1, 0) @ W2 + b2))


def make(name, seed, ov):
    if name == 'lookup':
        return Lookup(seed)
    if name == 'mlp_onehot':
        return MLP(seed, 'onehot')
    if name == 'mlp_thermo':
        return MLP(seed, 'thermo')
    if name == 'sparc_num':
        from sparc_num import NumSPARC
        return NumSPARC(seed=seed, **ov)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='lookup')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    from bench import parse_val
    ov = {k: parse_val(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    rng = np.random.RandomState(a.seed)
    model = make(a.model, a.seed, ov)
    T0 = time.time()
    res = dict(model=a.model, seed=a.seed, split=a.split, overrides=ov, tag=a.tag)
    # 1) GROUND: 300 training pairings (dot array of n, then the digit n); test 200 new dot arrays
    imgs = None
    if hasattr(model, 'train_ground_img'):
        from data import load
        Xtr, ytr, Xev, yev = load('mnist', a.split)
        pools = {c: list(np.where(ytr == c)[0]) for c in range(10)}
        epools = {c: list(np.where(yev == c)[0]) for c in range(10)}
        imgs = (Xtr, pools, Xev, epools)
        model.setup(imgs, rng)
    for _ in range(300):
        n = rng.randint(1, 10)
        if imgs:
            model.train_ground_img(dot_image(n, rng), n)
        else:
            model.train_ground(n, n)
    hits = []
    for _ in range(200):
        n = rng.randint(1, 10)
        g = model.ground_img(dot_image(n, rng)) if imgs else model.ground(n)
        hits.append(g == n)
    res['GROUND'] = float(np.mean(hits))
    # 2) ORDER: counting sequences only; then compare every pair (a ≠ b) of 1..9
    for _ in range(20):
        model.train_count(list(range(1, 10)))
    acc, close, far = [], [], []
    for x in range(1, 10):
        for y in range(1, 10):
            if x == y:
                continue
            ok = model.larger(x, y) == max(x, y)
            acc.append(ok); (close if abs(x - y) <= 2 else far).append(ok)
    res.update(ORDER=float(np.mean(acc)), ORDER_close=float(np.mean(close)), ORDER_far=float(np.mean(far)))
    # 3) ADD: train on 2/3 of the pairs (each 10×), test held-out pairs and training pairs
    train, held = split_pairs(rng)
    for _ in range(10):
        for (x, y) in train:
            model.train_add(x, y, x + y)
    res['ADD_train'] = float(np.mean([model.add(x, y) == x + y for x, y in train]))
    outs = [(model.add(x, y), x + y) for x, y in held]
    res['ADD_heldout'] = float(np.mean([o == t for o, t in outs]))
    res['ADD_heldout_pm1'] = float(np.mean([abs(o - t) <= 1 for o, t in outs]))
    res['n_heldout'] = len(held)
    res['wall_clock_s'] = time.time() - T0
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"num_{a.model}{'_' + a.tag if a.tag else ''}_{a.seed}_{a.split}.json")
    json.dump(res, open(path, 'w'), indent=1)
    print(f"GROUND {res['GROUND']:.3f}  ORDER {res['ORDER']:.3f} (close {res['ORDER_close']:.3f}, far {res['ORDER_far']:.3f})  "
          f"ADD train {res['ADD_train']:.3f} held-out {res['ADD_heldout']:.3f} (±1 {res['ADD_heldout_pm1']:.3f})  wrote {path}")


if __name__ == '__main__':
    main()
