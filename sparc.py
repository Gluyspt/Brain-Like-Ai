"""SPARC: Sparse, Predictive, Adaptive-Resonance network with Consolidation.

All plasticity is local (pre activity, post activity, broadcast scalar modulators).
No autograd, no backprop through layers. Batch 1, single pass. No raw-input storage.

Brain mapping (see goal.md §5 / ARCHITECTURE.md):
  V1        experience-dependent simple-cell dictionary, critical period via 1/usage learning rate
  IT        adaptive-resonance prototypes: refine on match, recruit on mismatch (ACh/NE-gated vigilance)
  fast      fly mushroom-body style readout: only the labelled class's synapses from active units change
  hippo     bounded store of sparse IT codes (index + values + label), write probability rises with surprise
  sleep     interleaved, class-balanced replay trains a slow cortical readout (complementary learning
            systems); structural consolidation merges redundant prototypes (sleep pressure = full budget)
"""
import numpy as np
from v1 import V1


def _unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class SPARC:
    name = 'sparc'
    batch_size = 1

    def __init__(self, seed=0,
                 # IT layer
                 n_it=1500, k_it=15, rho=0.80, center=True, refine=True, adapt_z=1.0, adapt_tau=500,
                 # V1
                 v1='learned', v1_K=64, v1_k=3, v1_grid=4,
                 # structural consolidation
                 merge_frac=0.25, merge_at_task_end=True,
                 # hippocampus + sleep (hip_M=0 disables)
                 hip_M=1000, hip_gain=1.0, sleep_epochs=3, sleep_eta=0.5, downscale=0.0,
                 mix='conf', beta=20.0, n_cls=10, match_track=True,
                 correct=False, err_theta=0.5, reopen=0.2, ltd=1.0, sleep_clean=False, clean_theta=0.9):
        self.rng = np.random.RandomState(seed)
        self.N, self.k, self.rho, self.C = n_it, k_it, rho, n_cls
        self.center, self.refine, self.adapt_z, self.adapt_tau = center, refine, adapt_z, adapt_tau
        self.merge_frac, self.merge_at_task_end = merge_frac, merge_at_task_end
        self.v1 = None if v1 == 'none' else V1(self.rng, K=v1_K, k=v1_k, grid=v1_grid, mode=v1)
        self.D = 784 if self.v1 is None else self.v1.out_dim
        # IT prototype layer (adaptive resonance)
        self.W = np.zeros((self.N, self.D), np.float32)
        self.n = np.zeros(self.N)            # usage counts (metaplasticity)
        self.used = 0                        # units recruited so far (budget = N)
        self.s_mu, self.s_var = 0.2, 0.01    # running surprise statistics (neuromodulator baseline)
        # fast readout: per-class running mean of sparse codes (only row c changes)
        self.A = np.zeros((self.C, self.N), np.float32)
        self.nc = np.zeros(self.C)
        # hippocampus: sparse code store
        self.M, self.hip_gain = hip_M, hip_gain
        self.h_idx = np.zeros((max(hip_M, 1), k_it), np.int64)
        self.h_val = np.zeros((max(hip_M, 1), k_it), np.float32)
        self.h_lab = np.full(max(hip_M, 1), -1, np.int64)
        self.h_n = 0
        # slow cortical readout, trained only in sleep
        self.R = np.zeros((self.C, self.N), np.float32)
        self.sleep_epochs, self.sleep_eta, self.downscale = sleep_epochs, sleep_eta, downscale
        self.mix, self.beta = mix, beta
        self.slept = False
        self.match_track = match_track
        self.n_mt = 0
        # self-correction (P10)
        self.correct, self.err_theta, self.reopen, self.ltd = correct, err_theta, reopen, ltd
        self.sleep_clean, self.clean_theta = sleep_clean, clean_theta
        self.n_conf_err = self.n_cleaned = 0
        # bookkeeping
        self.n_recruit = self.n_seen = self.n_merged = self.n_consol = self.n_sleep = self.n_writes = 0
        self.full_at = -1

    # ---------- encoding ----------
    def _pre(self, X):
        X = np.asarray(X, np.float32)
        if self.v1 is not None:
            X = self.v1.encode(X)
        if self.center:  # subtractive + divisive normalisation
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

    # ---------- learning (awake) ----------
    def learn(self, X, y=None):
        if self.v1 is not None:
            self.v1.learn(np.asarray(X, np.float32)[0])
        x = self._pre(X)[0]
        self.n_seen += 1
        u = self.used
        a = self.W[:u] @ x if u else np.zeros(0, np.float32)
        best = a.max() if u else -1.0
        s = 1.0 - best
        z = (s - self.s_mu) / np.sqrt(self.s_var)          # surprise relative to expectation
        novel = best < self.rho
        if self.adapt_z is not None and u:
            novel = novel and z > self.adapt_z
        r = 1.0 / self.adapt_tau
        d = s - self.s_mu
        self.s_mu += r * d
        self.s_var += r * (d * d - self.s_var)
        c = None if y is None else int(np.asarray(y).ravel()[0])
        if not novel and self.match_track and c is not None and u:
            # match tracking: teaching signal contradicts the top-down prediction -> raise vigilance
            seen = np.where(self.nc > 0)[0]
            if self.nc[c] == 0:
                novel = True
            else:
                h0 = _unit(self._code(a[None]))[0]
                pred = seen[np.argmax(_unit(self.A[seen]) @ h0)]
                novel = pred != c
            self.n_mt += int(novel)
        if novel and self.used < self.N:
            # mismatch -> recruit a fresh unit, one-shot imprint
            j = self.used
            self.W[j] = x
            self.n[j] = 1
            self.used += 1
            self.n_recruit += 1
            a = self.W[:self.used] @ x
        elif self.refine and u:
            # resonance -> refine top-k winners, rate from own usage
            k = min(self.k, u)
            win = np.argpartition(-a, k - 1)[:k]
            eta = 1.0 / (1.0 + self.n[win])
            g = (eta * np.maximum(a[win], 0))[:, None]
            self.W[win] = _unit(self.W[win] + g * (x[None] - self.W[win]))
            self.n[win] += 1
            a = self.W[:self.used] @ x
        if c is not None:
            h = self._code(a[None])[0]
            if self.correct and self.nc[c] > 0:
                self._confident_error(h, c)
            self.nc[c] += 1
            self.A[c] += (h - self.A[c]) / self.nc[c]
            if self.M > 0:
                p = 1.0 / (1.0 + np.exp(-self.hip_gain * z))   # novelty-gated memory write
                if self.rng.rand() < p:
                    self._hip_write(h, c)
        if self.used == self.N and self.merge_frac > 0:     # sleep pressure: budget full
            self.consolidate()

    def _confident_error(self, h, c):
        """Three-factor correction. A confident wrong prediction releases an error modulator M (NE/DA-like):
        (1) LTD at the active synapses onto the wrongly predicted output neuron, scaled by that neuron's
            own plasticity (evidence-weighted: a well-confirmed link only gets nudged);
        (2) reconsolidation: both output neurons involved lose part of their accumulated evidence count,
            so repeated errors re-open their plasticity while a single bad label barely matters."""
        seen = np.where(self.nc > 0)[0]
        if len(seen) < 2:
            return
        hn = h / (np.linalg.norm(h) + 1e-8)
        pf = _softmax(self.beta * (_unit(self.A[seen]) @ hn))
        j = int(pf.argmax())
        cp, q = int(seen[j]), float(pf[j])
        if cp == c or q < self.err_theta:
            return
        M = q
        act = h > 0
        g = min(1.0, self.ltd * M * 50.0 / (50.0 + self.nc[cp]))
        self.A[cp, act] *= (1.0 - g * hn[act] / (hn[act].max() + 1e-8))
        self.nc[c] = max(1.0, self.nc[c] * (1.0 - self.reopen * M))
        self.nc[cp] = max(1.0, self.nc[cp] * (1.0 - self.reopen * M))
        self.n_conf_err += 1

    # ---------- hippocampus ----------
    def _hip_write(self, h, c):
        idx = np.argpartition(-h, self.k - 1)[:self.k]
        val = h[idx]
        idx = np.where(val > 0, idx, 0)      # empty slots point at unit 0 with value 0
        if self.h_n < self.M:
            j = self.h_n
            self.h_n += 1
        else:  # full: overwrite a random trace of the most-represented class (interference/decay)
            cnt = np.bincount(self.h_lab[:self.h_n], minlength=self.C)
            cand = np.where(self.h_lab[:self.h_n] == cnt.argmax())[0]
            j = int(self.rng.choice(cand))
        self.h_idx[j], self.h_val[j], self.h_lab[j] = idx, val, c
        self.n_writes += 1

    # ---------- sleep ----------
    def end_task(self):
        if self.merge_frac > 0 and self.merge_at_task_end and self.used > (1 - self.merge_frac) * self.N:
            self.consolidate()
        if self.M > 0 and self.h_n > 0:
            self.sleep()

    def sleep(self):
        """Offline replay: class-balanced reactivation of hippocampal codes trains the slow readout
        with a single-layer delta rule (error lives at the output neuron), then synaptic downscaling."""
        self.n_sleep += 1
        if self.sleep_clean:
            self._sleep_clean()
        if self.h_n == 0:
            return
        lab = self.h_lab[:self.h_n]
        classes = np.unique(lab)
        by_c = {c: np.where(lab == c)[0] for c in classes}
        n_rep = self.sleep_epochs * self.h_n
        for _ in range(n_rep):
            c = classes[self.rng.randint(len(classes))]
            j = by_c[c][self.rng.randint(len(by_c[c]))]
            idx, val = self.h_idx[j], self.h_val[j]
            val = val / (np.linalg.norm(val) + 1e-8)
            logit = self.R[classes][:, idx] @ val
            p = _softmax(logit)
            err = (classes == c).astype(np.float32) - p
            self.R[np.ix_(classes, idx)] += self.sleep_eta * err[:, None] * val[None]
        if self.downscale > 0:
            self.R *= (1.0 - self.downscale)
        self.slept = True

    def _sleep_clean(self):
        """Replay each trace against current fast (recent) knowledge; traces whose label is confidently
        contradicted are dropped (newer confirmed evidence wins; active forgetting during sleep)."""
        seen = np.where(self.nc > 0)[0]
        n = self.h_n
        H = np.zeros((n, self.N), np.float32)
        np.add.at(H, (np.repeat(np.arange(n), self.k), self.h_idx[:n].ravel()), self.h_val[:n].ravel())
        pf = _softmax(self.beta * (_unit(H) @ _unit(self.A[seen]).T))
        pred, q = seen[pf.argmax(1)], pf.max(1)
        keep = ~((pred != self.h_lab[:n]) & (q > self.clean_theta))
        m = int(keep.sum())
        self.n_cleaned += n - m
        for arr in (self.h_idx, self.h_val, self.h_lab):
            arr[:m] = arr[:n][keep]
        self.h_n = m

    def consolidate(self):
        """Structural consolidation: free merge_frac of the budget by merging the most similar
        prototype pairs within the classes holding more than a fair (water-filled) share.
        Merged unit = usage-weighted mean; readout synapses add; hippocampal indices follow the unit."""
        self.n_consol += 1
        if self.full_at < 0 and self.used == self.N:
            self.full_at = self.n_seen
        u = self.used
        target = int((1 - self.merge_frac) * self.N)
        if u <= target:
            return
        lab = np.where(self.A[:, :u].max(0) > 0, self.A[:, :u].argmax(0), -1)
        groups = {g: np.where(lab == g)[0] for g in np.unique(lab)}
        cnt = np.array([len(v) for v in groups.values()])
        lo, hi = 0, cnt.max()                      # water-filling cap
        while lo < hi:
            c = (lo + hi + 1) // 2
            if np.minimum(cnt, c).sum() <= target:
                lo = c
            else:
                hi = c - 1
        cap = max(lo, 1)
        into = np.arange(u)                        # merged-into pointer
        keep_all = []
        for g, idx in groups.items():
            idx = list(idx)
            while len(idx) > cap:
                Wg = self.W[idx]
                S = Wg @ Wg.T
                np.fill_diagonal(S, -2)
                need = len(idx) - cap
                iu = np.triu_indices(len(idx), 1)
                sv = S[iu]
                taken = np.zeros(len(idx), bool)
                drop = set()
                for o in np.argsort(-sv):
                    if len(drop) >= need:
                        break
                    i, j = iu[0][o], iu[1][o]
                    if taken[i] or taken[j]:
                        continue
                    taken[i] = taken[j] = True
                    gi, gj = idx[i], idx[j]
                    ni, nj = self.n[gi], self.n[gj]
                    self.W[gi] = _unit(ni * self.W[gi] + nj * self.W[gj])
                    self.n[gi] = ni + nj
                    self.A[:, gi] += self.A[:, gj]
                    self.R[:, gi] += self.R[:, gj]
                    into[into == gj] = gi
                    drop.add(j)
                    self.n_merged += 1
                idx = [v for t, v in enumerate(idx) if t not in drop]
            keep_all += idx
        keep = np.sort(np.array(keep_all, int))
        m = len(keep)
        newpos = np.full(u, -1)
        newpos[keep] = np.arange(m)
        remap = newpos[into]
        self.W[:m] = self.W[keep]; self.W[m:u] = 0
        self.n[:m] = self.n[keep]; self.n[m:u] = 0
        for Mx in (self.A, self.R):
            Mx[:, :m] = Mx[:, keep]; Mx[:, m:u] = 0
        if self.h_n:
            hi = self.h_idx[:self.h_n]
            hi[hi >= u] = 0
            self.h_idx[:self.h_n] = np.maximum(remap[hi], 0)
        self.used = m

    # ---------- inference ----------
    def class_probs(self, X, allowed):
        H, _ = self.encode(X)
        allowed = np.asarray(allowed)
        Hn = _unit(H)
        pf = _softmax(self.beta * (Hn @ _unit(self.A[allowed]).T))
        if not self.slept or self.mix == 'fast':
            return pf
        ps = _softmax(Hn @ self.R[allowed].T)
        if self.mix == 'slow':
            return ps
        if self.mix == 'conf':
            wf, ws = pf.max(1, keepdims=True), ps.max(1, keepdims=True)
            return (wf * pf + ws * ps) / (wf + ws)
        lam = float(self.mix)
        return (1 - lam) * pf + lam * ps

    def predict(self, X, allowed):
        allowed = np.asarray(allowed)
        return allowed[self.class_probs(X, allowed).argmax(1)]

    def surprise(self, X):
        _, a = self.encode(X)
        return 1.0 - a.max(1)

    def sparsity(self, X):
        H, _ = self.encode(X)
        out = {'IT': float((H != 0).mean())}
        if self.v1 is not None:
            out['V1'] = self.v1.encode(np.asarray(X, np.float32), return_code_density=True)[1]
        return out

    def stats(self):
        p = self.W.size + self.A.size + self.R.size + (self.v1.D.size if self.v1 is not None else 0)
        hip = self.M * (2 * self.k + 1)
        return dict(params=int(p), stored_numbers=int(hip), raw_images_stored=0,
                    hippocampus_budget=f'{self.M} codes x (2*{self.k}+1) = {hip} numbers '
                                       f'(vs 500 raw images = {500 * 785})',
                    it_used=int(self.used), it_budget=int(self.N))

    def diagnostics(self):
        return dict(it_used=int(self.used), n_recruit=int(self.n_recruit), n_seen=int(self.n_seen),
                    n_merged=int(self.n_merged), n_consol=int(self.n_consol), full_at=int(self.full_at),
                    n_sleep=int(self.n_sleep), n_match_track=int(self.n_mt), n_conf_err=int(self.n_conf_err), n_cleaned=int(self.n_cleaned), hip_writes=int(self.n_writes), hip_size=int(self.h_n),
                    hip_per_class=[int(v) for v in np.bincount(self.h_lab[:self.h_n], minlength=self.C)],
                    units_per_class=[int(v) for v in np.bincount(self.A[:, :self.used].argmax(0), minlength=self.C)])
