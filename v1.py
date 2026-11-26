"""V1: local patches, divisive normalisation, competitive-Hebbian dictionary (or innate Gabor bank),
k-winners-take-all per position, fixed spatial sum-pooling. Local learning only."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:                                   # optional JIT for the per-patch learning loop (docs/todo.md)
    from numba import njit
except ImportError:                    # plain Python fallback: identical rule, slower
    njit = None


def _learn_patches(D, n, used, Ps, rho1, eta_min, K):
    """Competitive-Hebbian dictionary update for a few patches in sequence (same rule as before)."""
    for t in range(Ps.shape[0]):
        p = Ps[t]
        best = -1.0
        w = 0
        if used > 0:
            a = D[:used] @ p
            w = int(np.argmax(a))
            best = a[w]
        if best < rho1 and used < K:
            D[used] = p
            n[used] = 1
            used += 1
        else:
            eta = max(1.0 / (1.0 + n[w]), eta_min)
            d = D[w] + eta * (p - D[w])
            D[w] = d / (np.sqrt((d * d).sum()) + 1e-8)
            n[w] += 1
    return used


if njit is not None:
    _learn_patches = njit(cache=True)(_learn_patches)


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
        self.Pool = (self.region[None, :] == np.arange(grid * grid)[:, None]).astype(np.float32)  # (regions, positions)
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

    def learn(self, x, pm=None):
        if self.mode == 'gabor':
            return
        P, m = self._patches(x[None]) if pm is None else pm
        cand = np.where(m[0])[0]
        if len(cand) == 0:
            return
        sel = self.rng.choice(cand, min(self.n_patch, len(cand)), replace=False)
        Ps = np.ascontiguousarray(P[0, sel], dtype=np.float32)
        self.used = int(_learn_patches(self.D, self.n, self.used, Ps, np.float32(self.rho1), self.eta_min, self.K))

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

    def encode(self, X, return_code_density=False, chunk=256, pm=None):
        out = np.zeros((len(X), self.grid * self.grid, self.K), np.float32)
        dens = []
        u = self.used
        if u == 0:
            return out.reshape(len(X), -1)
        k = min(self.k, u)
        for s in range(0, len(X), chunk):
            P, m = self._patches(X[s:s + chunk]) if pm is None else pm
            A = P @ self.D[:u].T                                  # (b, pos, u)
            thr = np.partition(A, u - k, axis=-1)[..., u - k:u - k + 1]
            A = np.where((A >= thr) & (A > 0) & m[..., None], A, 0)   # k-WTA per position
            if return_code_density:
                dens.append((A != 0).sum(-1).mean() / self.K)
            # sum-pool over fixed regions
            out[s:s + chunk, :, :u] = np.matmul(self.Pool, A)          # sum-pool over fixed regions
        out = out.reshape(len(X), -1)
        if return_code_density:
            return out, float(np.mean(dens))
        return out
