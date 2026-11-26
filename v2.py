"""V2: second cortical stage. Complex-cell pooling of the V1 code map (local max over 2x2),
then a competitive-Hebbian dictionary over 3x3 neighbourhoods of complex cells (conjunctions of
V1 features: corners, junctions, curves), k-WTA per position, fixed sum-pooling.
Hubel & Wiesel simple -> complex -> simple hierarchy (as in the neocognitron / HMAX).
Local learning only; its own critical period (closes later than V1's)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


class V2:
    def __init__(self, rng, v1, K=128, k=5, grid=4, pool=2, p=3, rho=0.5, eta_min=0.002, n_patch=10):
        self.rng, self.v1, self.K, self.k, self.grid, self.pool, self.p = rng, v1, K, k, grid, pool, p
        self.rho, self.eta_min, self.n_patch = rho, eta_min, n_patch
        self.C = v1.S // pool                     # complex-cell map side
        self.S = self.C - p + 1                   # V2 positions per side
        self.D = np.zeros((K, p * p * v1.K), np.float32)
        self.n = np.zeros(K)
        self.used = 0
        r = np.minimum(np.arange(self.S) * grid // self.S, grid - 1)
        self.region = (r[:, None] * grid + r[None, :]).ravel()
        self.out_dim = grid * grid * K

    def _patches(self, X):
        M = self.v1.code_map(X)                                   # (n, S1, S1, K1)
        n, q, c = len(X), self.pool, self.C
        M = M[:, :c * q, :c * q].reshape(n, c, q, c, q, -1).max((2, 4))   # complex cells: local max
        P = sliding_window_view(M, (self.p, self.p), axis=(1, 2)).reshape(n, self.S * self.S, -1)
        nrm = np.linalg.norm(P, axis=-1, keepdims=True)
        return P / (nrm + 1e-8), nrm[..., 0] > 0

    def learn(self, x):
        P, m = self._patches(x[None])
        cand = np.where(m[0])[0]
        if len(cand) == 0:
            return
        for i in self.rng.choice(cand, min(self.n_patch, len(cand)), replace=False):
            p = P[0, i]
            if self.used:
                a = self.D[:self.used] @ p
                w = int(a.argmax()); best = a[w]
            else:
                best = -1
            if best < self.rho and self.used < self.K:
                self.D[self.used] = p; self.n[self.used] = 1; self.used += 1
            else:
                eta = max(1.0 / (1.0 + self.n[w]), self.eta_min)
                self.D[w] = _unit(self.D[w] + eta * (p - self.D[w]))
                self.n[w] += 1

    def encode(self, X, return_code_density=False, chunk=128):
        out = np.zeros((len(X), self.grid * self.grid, self.K), np.float32)
        u, dens = self.used, []
        if u == 0:
            return (out.reshape(len(X), -1), 0.0) if return_code_density else out.reshape(len(X), -1)
        k = min(self.k, u)
        for s in range(0, len(X), chunk):
            P, m = self._patches(X[s:s + chunk])
            A = P @ self.D[:u].T
            thr = np.partition(A, u - k, axis=-1)[..., u - k:u - k + 1]
            A = np.where((A >= thr) & (A > 0) & m[..., None], A, 0)
            if return_code_density:
                dens.append((A != 0).sum(-1).mean() / self.K)
            for r in range(self.grid * self.grid):
                out[s:s + chunk, r, :u] = A[:, self.region == r].sum(1)
        out = out.reshape(len(X), -1)
        return (out, float(np.mean(dens))) if return_code_density else out
