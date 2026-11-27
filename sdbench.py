"""Stage-3 benchmark: same/different abstract rule (honeybees, monkeys, children; Giurfa et al. 2001 [known]).

python sdbench.py --model lookup|mlp|sparc_sd --seed 0 [--split val]

Pairs of Fashion-MNIST images; answer SAME if both show the same kind of object, else DIFFERENT. Feedback is
given on pairs from classes 0–4 only; the test uses classes 5–9 (never seen in training): the rule must
transfer to new items. 50 % of pairs are same, 50 % different (different images even for 'same').
"""
import argparse, json, os
import numpy as np
from data import load

TRAIN_C, TEST_C = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]


class Lookup:
    def __init__(self, seed): self.rng = np.random.RandomState(seed); self.mem = {}
    def answer(self, x1, x2, c1, c2): return self.mem.get((c1, c2), self.rng.rand() < 0.5)
    def feedback(self, x1, x2, c1, c2, same): self.mem[(c1, c2)] = same


class MLP:
    """Backprop MLP on the two images' pixels (concatenated), trained online (non-brain baseline)."""
    def __init__(self, seed, h=64, lr=0.05):
        r = np.random.RandomState(seed); self.W1 = r.randn(1568, h) * 0.03; self.b1 = np.zeros(h); self.w2 = r.randn(h) * 0.1; self.lr = lr
    def _f(self, x1, x2):
        x = np.concatenate([x1.ravel(), x2.ravel()]); hh = np.maximum(x @ self.W1 + self.b1, 0)
        return x, hh, 1 / (1 + np.exp(-(hh @ self.w2)))
    def answer(self, x1, x2, c1, c2): return self._f(x1, x2)[2] > 0.5
    def feedback(self, x1, x2, c1, c2, same):
        x, hh, p = self._f(x1, x2); g = p - same; gh = g * self.w2 * (hh > 0)
        self.w2 -= self.lr * g * hh; self.W1 -= self.lr * np.outer(x, gh); self.b1 -= self.lr * gh


class SDSPARC:
    """Relational rule: a 'same' decision read from the similarity of the two items' graded IT activation
    profiles (how strongly every prototype responds), a relational feature independent of item identity.
    The decision boundary is learned from feedback as two running means (typical similarity on 'same' and on
    'different' trials, Hebbian prototypes); answer = nearer prototype. Brain basis: match enhancement/
    suppression signals and abstract same/different neurons in PFC (Wallis, Anderson & Miller 2001 [known]).
    Perception = stage-1 SPARC exposed to images without labels (no same/different feedback on new classes)."""
    def __init__(self, seed=0, stage1_n=6000, feat='dense', **stage1):
        from sparc import SPARC
        self.s = SPARC(seed=seed, glimpses=1, **stage1); self.n1, self.feat = stage1_n, feat
        self.mu = {True: 0.0, False: 0.0}; self.cnt = {True: 0, False: 0}
    def setup(self, Xtr, ytr, rng):
        for i in np.sort(rng.permutation(len(Xtr))[:self.n1]):
            self.s.learn(Xtr[i:i + 1], None)              # unlabelled exposure: perception only
    def _sim(self, x1, x2):
        H, A = self.s.encode(np.concatenate([x1, x2]))
        V = A if self.feat == 'dense' else H
        V = V - V.mean(1, keepdims=True) if self.feat == 'dense' else V
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        return float(V[0] @ V[1])
    def answer(self, x1, x2, c1, c2):
        if min(self.cnt.values()) == 0:
            return True
        s = self._sim(x1, x2)
        return abs(s - self.mu[True]) < abs(s - self.mu[False])
    def feedback(self, x1, x2, c1, c2, same):
        s = self._sim(x1, x2); self.cnt[same] += 1
        self.mu[same] += (s - self.mu[same]) / self.cnt[same]


def make(name, seed, ov):
    if name == 'lookup': return Lookup(seed)
    if name == 'mlp': return MLP(seed)
    if name == 'sparc_sd': return SDSPARC(seed=seed, **ov)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='lookup'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val', choices=['val', 'test']); ap.add_argument('--trials', type=int, default=2000)
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default=''); ap.add_argument('--out', default='results')
    a = ap.parse_args()
    from bench import parse_val
    ov = {k: parse_val(v) for k, v in (kv.split('=', 1) for kv in a.set)}
    rng = np.random.RandomState(a.seed)
    Xtr, ytr, Xev, yev = load('fashion', a.split)
    model = make(a.model, a.seed, ov)
    if hasattr(model, 'setup'): model.setup(Xtr, ytr, rng)
    pools = lambda y, cs: {c: np.where(y == c)[0] for c in cs}
    trp, tep_tr, tep_new = pools(ytr, TRAIN_C), pools(yev, TRAIN_C), pools(yev, TEST_C)
    def pair(P, X, cs):
        same = rng.rand() < 0.5
        c1 = cs[rng.randint(len(cs))]; c2 = c1 if same else cs[rng.choice([k for k in range(len(cs)) if cs[k] != c1])]
        i1, i2 = rng.choice(P[c1]), rng.choice(P[c2])
        while i2 == i1: i2 = rng.choice(P[c2])
        return X[i1][None], X[i2][None], c1, c2, same
    hits = []
    for t in range(a.trials):
        x1, x2, c1, c2, same = pair(trp, Xtr, TRAIN_C)
        hits.append(model.answer(x1, x2, c1, c2) == same); model.feedback(x1, x2, c1, c2, same)
    def test(P, cs):
        return float(np.mean([model.answer(*(lambda p: (p[0], p[1], p[2], p[3]))(p := pair(P, Xev, cs))) == p[4] for _ in range(1000)]))
    res = dict(model=a.model, seed=a.seed, split=a.split, overrides=ov, tag=a.tag,
               train_last500=float(np.mean(hits[-500:])), SD_trained_classes=test(tep_tr, TRAIN_C), SD_novel_classes=test(tep_new, TEST_C))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"sd_{a.model}{'_' + a.tag if a.tag else ''}_{a.seed}_{a.split}.json")
    json.dump(res, open(path, 'w'), indent=1)
    print(f"train {res['train_last500']:.3f}  test on trained classes {res['SD_trained_classes']:.3f}  on NOVEL classes {res['SD_novel_classes']:.3f}  wrote {path}")


if __name__ == '__main__':
    main()
