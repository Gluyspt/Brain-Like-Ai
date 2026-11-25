"""SPARC: Sparse, Predictive, Adaptive-Resonance network with Consolidation.

All plasticity is local (pre activity, post activity, broadcast scalar modulators).
No autograd, no backprop through layers. Batch 1, single pass. No raw-input storage.
"""
import numpy as np


def _unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


class SPARC:
    name = 'sparc'
    batch_size = 1

    def __init__(self, seed=0, n_it=1500, k_it=15, rho=0.80, eta_min=0.0, n_cls=10, center=True, refine=True):
        self.refine = refine
        self.full_at = -1
        self.rng = np.random.RandomState(seed)
        self.N, self.k, self.rho, self.eta_min, self.C = n_it, k_it, rho, eta_min, n_cls
        self.center = center
        self.D = 784
        # IT prototype layer (adaptive resonance)
        self.W = np.zeros((self.N, self.D), np.float32)
        self.n = np.zeros(self.N)            # usage counts (metaplasticity)
        self.used = 0                        # units recruited so far (budget = N)
        # fast readout: per-class running mean of sparse codes (fly-style, only row c changes)
        self.A = np.zeros((self.C, self.N), np.float32)
        self.nc = np.zeros(self.C)
        self.n_recruit = 0
        self.n_seen = 0

    # ---------- encoding ----------
    def _pre(self, X):
        X = np.asarray(X, np.float32)
        if self.center:  # subtractive + divisive normalisation (retina-like)
            X = X - X.mean(1, keepdims=True)
        return _unit(X)

    def _code(self, a):
        """k-WTA on activations a (n, used) -> sparse code (n, N)."""
        n, u = a.shape
        H = np.zeros((n, self.N), np.float32)
        if u == 0:
            return H
        k = min(self.k, u)
        idx = np.argpartition(-a, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(a, idx, 1)
        np.put_along_axis(H, idx, np.maximum(vals, 0), 1)
        return H

    def encode(self, X):
        x = self._pre(X)
        a = x @ self.W[:self.used].T
        return self._code(a), a

    # ---------- learning ----------
    def learn(self, X, y=None):
        x = self._pre(X)[0]
        self.n_seen += 1
        u = self.used
        a = self.W[:u] @ x if u else np.zeros(0, np.float32)
        best = a.max() if u else -1.0
        if best < self.rho and u < self.N:
            # novelty -> recruit a fresh unit, one-shot imprint
            self.W[u] = x
            self.n[u] = 1
            self.used += 1
            self.n_recruit += 1
            if self.used == self.N:
                self.full_at = self.n_seen
            a = self.W[:self.used] @ x
        elif self.refine and u:
            # resonance -> refine top-k winners, rate from own usage
            k = min(self.k, u)
            win = np.argpartition(-a, k - 1)[:k]
            eta = np.maximum(1.0 / (1.0 + self.n[win]), self.eta_min)
            g = (eta * np.maximum(a[win], 0))[:, None]
            self.W[win] = _unit(self.W[win] + g * (x[None] - self.W[win]))
            self.n[win] += 1
            a = self.W[:self.used] @ x
        if y is not None:
            c = int(np.asarray(y).ravel()[0])
            h = self._code(a[None])[0]
            self.nc[c] += 1
            self.A[c] += (h - self.A[c]) / self.nc[c]

    def end_task(self):
        pass

    # ---------- inference ----------
    def predict(self, X, allowed):
        H, _ = self.encode(X)
        allowed = np.asarray(allowed)
        S = _unit(H) @ _unit(self.A[allowed]).T
        return allowed[S.argmax(1)]

    def surprise(self, X):
        _, a = self.encode(X)
        return 1.0 - a.max(1)

    def sparsity(self, X):
        H, _ = self.encode(X)
        return {'IT': float((H != 0).mean())}

    def stats(self):
        p = self.W.size + self.A.size
        return dict(params=int(p), stored_numbers=0, raw_images_stored=0,
                    it_used=int(self.used), it_budget=int(self.N))

    def diagnostics(self):
        return dict(it_used=int(self.used), n_recruit=int(self.n_recruit), n_seen=int(self.n_seen), full_at=int(self.full_at),
                    units_per_class=[int(v) for v in np.bincount(self.A[:, :self.used].argmax(0), minlength=self.C)])
