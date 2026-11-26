"""Numpy baselines B1-B5. Common interface: name, batch_size, learn, end_task, predict, stats."""
import numpy as np


class MLP:
    name = 'b1'
    batch_size = 10

    def __init__(self, seed, lr=0.05, dim=784, sizes=None):
        sizes = sizes or (dim, 400, 400, 10)
        self.rng = np.random.RandomState(seed)
        self.lr = lr
        self.W = [(self.rng.randn(a, b) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _forward(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return acts

    def _grads(self, X, y):
        acts = self._forward(X)
        z = acts[-1]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        d = p / len(y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def _step(self, X, y):
        gW, gb = self._grads(X, y)
        for i in range(len(self.W)):
            self.W[i] -= self.lr * gW[i]
            self.b[i] -= self.lr * gb[i]

    def learn(self, X, y):
        if y is None:
            return
        self._step(X, y)

    def end_task(self):
        pass

    def predict(self, X, allowed):
        allowed = list(allowed)
        out = np.empty(len(X), np.int64)
        for s in range(0, len(X), 5000):
            z = self._forward(X[s:s + 5000])[-1][:, allowed]
            out[s:s + 5000] = np.asarray(allowed, np.int64)[z.argmax(1)]
        return out

    def _nparams(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    def stats(self):
        return dict(params=self._nparams(), stored_numbers=0, raw_images_stored=0)


class ReplayMLP(MLP):
    name = 'b2'
    CAP = 500

    def __init__(self, seed, lr=0.05, dim=784):
        super().__init__(seed, lr, dim)
        self.dim = dim
        self.bX = np.zeros((self.CAP, dim), np.float32)
        self.by = np.zeros(self.CAP, np.int64)
        self.n_seen = 0

    def learn(self, X, y):
        if y is None:
            return
        m = min(self.n_seen, self.CAP)
        if m > 0:
            r = self.rng.randint(0, m, size=len(X))
            Xb, yb = np.concatenate([X, self.bX[r]]), np.concatenate([y, self.by[r]])
        else:
            Xb, yb = X, y
        self._step(Xb, yb)
        for xi, yi in zip(X, y):
            if self.n_seen < self.CAP:
                j = self.n_seen
            else:
                j = self.rng.randint(0, self.n_seen + 1)
            if j < self.CAP:
                self.bX[j], self.by[j] = xi, yi
            self.n_seen += 1

    def stats(self):
        return dict(params=self._nparams(), stored_numbers=self.CAP * (self.dim + 1), raw_images_stored=self.CAP)


class JointMLP(MLP):
    name = 'b3'

    def fit_joint(self, X, y, epochs=10, batch=64):
        lr, mom = 0.01, 0.9
        n = len(X)
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                gW, gb = self._grads(X[idx], y[idx])
                for i in range(len(self.W)):
                    self.vW[i] = mom * self.vW[i] - lr * gW[i]
                    self.vb[i] = mom * self.vb[i] - lr * gb[i]
                    self.W[i] += self.vW[i]
                    self.b[i] += self.vb[i]


class NCM:
    name = 'b4'
    batch_size = 10

    def __init__(self, seed=0, dim=784):
        self.dim = dim
        self.sums = np.zeros((10, dim), np.float64)
        self.cnt = np.zeros(10, np.int64)

    def learn(self, X, y):
        if y is None:
            return
        for c in np.unique(y):
            m = y == c
            self.sums[c] += X[m].sum(0)
            self.cnt[c] += m.sum()

    def end_task(self):
        pass

    def predict(self, X, allowed):
        allowed = np.asarray(list(allowed), np.int64)
        mu = (self.sums[allowed] / np.maximum(self.cnt[allowed], 1)[:, None]).astype(np.float32)
        d = (mu ** 2).sum(1)[None, :] - 2 * X @ mu.T  # ||x||^2 constant per row
        return allowed[d.argmin(1)]

    def stats(self):
        return dict(params=0, stored_numbers=10 * self.dim, raw_images_stored=0)


class GDumb:
    """GDumb (Prabhu et al. 2020): greedy class-balanced memory; fresh MLP trained on memory at query time."""
    name = 'b5'
    batch_size = 10

    def __init__(self, seed, mem=500, dim=784):
        self.seed, self.mem, self.dim = seed, mem, dim
        self.rng = np.random.RandomState(seed)
        self.items = []  # list of (x, y)
        self.cnt = {}
        self.net = None
        self.dirty = True

    def learn(self, X, y):
        if y is None:
            return
        for xi, yi in zip(X, y):
            yi = int(yi)
            if len(self.items) < self.mem:
                self.items.append((xi.copy(), yi))
            elif self.cnt.get(yi, 0) < self.mem / (len(self.cnt) + (yi not in self.cnt)):
                big = max(self.cnt.values())
                cands = [j for j, (_, c) in enumerate(self.items) if self.cnt[c] == big]
                j = cands[self.rng.randint(len(cands))]
                self.cnt[self.items[j][1]] -= 1
                self.items[j] = (xi.copy(), yi)
            else:
                continue
            self.cnt[yi] = self.cnt.get(yi, 0) + 1
            self.dirty = True

    def end_task(self):
        pass

    def _train(self, epochs=30, batch=16, lr=0.05, mom=0.9):
        net = MLP(self.seed, dim=self.dim)
        X = np.stack([x for x, _ in self.items]).astype(np.float32)
        y = np.array([c for _, c in self.items], np.int64)
        rng = np.random.RandomState(self.seed)
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                idx = perm[s:s + batch]
                gW, gb = net._grads(X[idx], y[idx])
                for i in range(len(net.W)):
                    net.vW[i] = mom * net.vW[i] - lr * gW[i]
                    net.vb[i] = mom * net.vb[i] - lr * gb[i]
                    net.W[i] += net.vW[i]
                    net.b[i] += net.vb[i]
        return net

    def predict(self, X, allowed):
        if self.dirty or self.net is None:
            self.net = self._train()
            self.dirty = False
        return self.net.predict(X, allowed)

    def stats(self):
        net = self.net or MLP(self.seed, dim=self.dim)
        return dict(params=net._nparams(), stored_numbers=self.mem * (self.dim + 1), raw_images_stored=self.mem)


def make_baseline(name, seed, dim=784):
    return {'b1': MLP, 'b2': ReplayMLP, 'b3': JointMLP, 'b4': NCM, 'b5': GDumb}[name](seed, dim=dim)
