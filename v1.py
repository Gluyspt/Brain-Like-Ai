"""V1: local patches, divisive normalisation, competitive-Hebbian dictionary (or innate Gabor bank),
k-winners-take-all per position, fixed spatial sum-pooling. Local learning only."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def gabor_bank(K=64, p=5, seed=0):
    ys, xs = np.mgrid[-(p // 2):p // 2 + 1, -(p // 2):p // 2 + 1].astype(np.float32)
    F = []
    n_or = K // 4
    for th in np.linspace(0, np.pi, n_or, endpoint=False):
        for lam in (3.0, 5.0):
            for ph in (0.0, np.pi / 2):
                xr = xs * np.cos(th) + ys * np.sin(th)
                yr = -xs * np.sin(th) + ys * np.cos(th)
                g = np.exp(-(xr ** 2 + 0.5 * yr ** 2) / (2 * 1.5 ** 2)) * np.cos(2 * np.pi * xr / lam + ph)
                F.append(g.ravel() - g.mean())
    return _unit(np.array(F[:K], np.float32))


class V1:
    def __init__(self, rng, K=64, p=5, k=3, grid=3, mode='learned', rho1=0.5, eta_min=0.002,
                 n_patch=10, contrast=0.1, lowsf=False, img_side=28, img_ch=1):
        self.rng, self.K, self.p, self.k, self.grid, self.mode = rng, K, p, k, grid, mode
        self.rho1, self.eta_min, self.n_patch, self.contrast = rho1, eta_min, n_patch, contrast
        self.lowsf = lowsf          # low spatial frequency 'hemisphere': sees a 2x2-averaged 14x14 image
        self.img_side, self.ch = img_side, img_ch
        side = img_side // 2 if lowsf else img_side
        self.side = side
        self.S = side - p + 1
        if mode == 'gabor':
            g = gabor_bank(K, p)                       # luminance filters, same on every channel
            self.D = _unit(np.repeat(g[:, None, :], img_ch, 1).reshape(K, -1))
            self.used = K
        else:
            self.D = np.zeros((K, img_ch * p * p), np.float32)
            self.used = 0
        self.n = np.zeros(K)
        # pooling region of each position
        r = np.minimum(np.arange(self.S) * grid // self.S, grid - 1)
        self.region = (r[:, None] * grid + r[None, :]).ravel()
        self.out_dim = grid * grid * K

    def _patches(self, X):
        """X (n, side*side*ch, HWC) -> normalised patches (n, S*S, ch*p*p) and contrast mask."""
        s, c = self.img_side, self.ch
        I = X.reshape(-1, s, s, c)
        if self.lowsf:
            I = I.reshape(-1, s // 2, 2, s // 2, 2, c).mean((2, 4))
        P = sliding_window_view(I, (self.p, self.p), axis=(1, 2)).reshape(len(X), self.S * self.S, -1)
        P = P - P.mean(-1, keepdims=True)
        nrm = np.linalg.norm(P, axis=-1, keepdims=True)
        return P / (nrm + 1e-8), (nrm[..., 0] > self.contrast)

    def learn(self, x):
        if self.mode == 'gabor':
            return
        P, m = self._patches(x[None])
        cand = np.where(m[0])[0]
        if len(cand) == 0:
            return
        for i in self.rng.choice(cand, min(self.n_patch, len(cand)), replace=False):
            p = P[0, i]
            if self.used:
                a = self.D[:self.used] @ p
                w = int(a.argmax())
                best = a[w]
            else:
                best = -1
            if best < self.rho1 and self.used < self.K:
                self.D[self.used] = p
                self.n[self.used] = 1
                self.used += 1
            else:
                eta = max(1.0 / (1.0 + self.n[w]), self.eta_min)
                self.D[w] = _unit(self.D[w] + eta * (p - self.D[w]))
                self.n[w] += 1

    def code_map(self, X):
        """Per-position k-WTA code before pooling: (n, S, S, K) (input to V2)."""
        n, u = len(X), self.used
        out = np.zeros((n, self.S, self.S, self.K), np.float32)
        if u == 0:
            return out
        k = min(self.k, u)
        P, m = self._patches(X)
        A = P @ self.D[:u].T
        thr = np.partition(A, u - k, axis=-1)[..., u - k:u - k + 1]
        A = np.where((A >= thr) & (A > 0) & m[..., None], A, 0)
        out[..., :u] = A.reshape(n, self.S, self.S, u)
        return out

    def encode(self, X, return_code_density=False, chunk=256):
        out = np.zeros((len(X), self.grid * self.grid, self.K), np.float32)
        dens = []
        u = self.used
        if u == 0:
            return out.reshape(len(X), -1)
        k = min(self.k, u)
        for s in range(0, len(X), chunk):
            P, m = self._patches(X[s:s + chunk])
            A = P @ self.D[:u].T                                  # (b, pos, u)
            thr = np.partition(A, u - k, axis=-1)[..., u - k:u - k + 1]
            A = np.where((A >= thr) & (A > 0) & m[..., None], A, 0)   # k-WTA per position
            if return_code_density:
                dens.append((A != 0).sum(-1).mean() / self.K)
            # sum-pool over fixed regions
            for r in range(self.grid * self.grid):
                out[s:s + chunk, r, :u] = A[:, self.region == r].sum(1)
        out = out.reshape(len(X), -1)
        if return_code_density:
            return out, float(np.mean(dens))
        return out
