"""Stage 2: SPARC with prediction over time (docs/roadmap.md, research/08, research/17).

Perception is the stage-1 SPARC (keeps learning online). On top of it:
  context    sparse conjunction of the previous and current IT codes (dendritic coincidence detection):
             z = kWTA((U h_t) * (V h_{t-1})), U, V fixed random (innate wiring)
  predictor  next IT code  ĥ_{t+1} = P_h h_t + P_z z_t, learned with a local delta rule:
             the error at each IT unit is (observed − predicted) code, as in predictive coding /
             cerebellar forward models; single layer, error at the output (allowed by P1)
  answer     the predicted code is read by stage 1's own fast readout (the model recognises its imagined
             next item)
  surprise   1 − cos(predicted, observed) (prediction error), computed before learning
No backprop, batch 1, single pass, no raw-input storage.
"""
import numpy as np
from sparc import SPARC, _unit, _softmax


class SeqSPARC:
    def __init__(self, seed=0, kz=20, nz=4000, eta=0.5, use_z=True, use_h=True, beta_pred=20.0, ctx='h', beta_q=20.0, meta=False, eta_min=0.02, **stage1):
        self.s = SPARC(seed=seed, **stage1)
        self.rng = np.random.RandomState(seed + 11)
        self.kz, self.nz, self.eta, self.use_z, self.use_h, self.beta_pred = kz, nz, eta, use_z, use_h, beta_pred
        # ctx='h': conjunctions of raw IT codes; ctx='q': of category-level activity (class evidence),
        # so different images of the same digit share their context (category cells feed sequence memory)
        self.ctx, self.beta_q = ctx, beta_q
        # metaplasticity of predictor synapses: rate 1/(1 + uses of the context unit), floor eta_min
        self.meta, self.eta_min = meta, eta_min
        self.U = self.V = None
        self.reset_context()

    # ---- helpers ----
    def _init(self):
        N = self.s.N
        D = 10 if self.ctx == 'q' else N
        self.U = self.rng.randn(self.nz, D).astype(np.float32)
        self.V = self.rng.randn(self.nz, D).astype(np.float32)
        self.Ph = np.zeros((N, N), np.float32)
        self.Pz = np.zeros((N, self.nz), np.float32)
        self.nzu = np.zeros(self.nz)
        self.nhu = np.zeros(N)

    def reset_context(self):
        self._qc = self._qp = None
        self.prev = None          # previous IT code (sparse: idx, val)
        self.pred = None          # predicted next code (dense, N)
        self._s = 0.0

    def _sparse(self, x):
        H, _ = self.s.encode(x)
        h = H[0]
        idx = np.nonzero(h)[0]
        val = h[idx]
        val = val / (np.linalg.norm(val) + 1e-8)
        return idx, val

    def _q(self, code):
        """Category-level activity: softmax of the cue's match to each learned class (stage-1 readout)."""
        seen = np.where(self.s.nc > 0)[0]
        q = np.zeros(10, np.float32)
        if len(seen):
            h = np.zeros(self.s.N, np.float32); h[code[0]] = code[1]
            q[seen] = _softmax(self.beta_q * (_unit(self.s.A[seen]) @ h))
        return np.arange(10), q

    def _z(self, cur, prev):
        if prev is None:
            return None
        if self.ctx == 'q':
            cur, prev = self._qc, self._qp
        a = self.U[:, cur[0]] @ cur[1]
        b = self.V[:, prev[0]] @ prev[1]
        p = a * b
        idx = np.argpartition(-p, self.kz - 1)[:self.kz]
        val = np.maximum(p[idx], 0)
        val = val / (np.linalg.norm(val) + 1e-8)
        return idx, val

    def _predict(self, cur, z):
        N = self.s.N
        out = np.zeros(N, np.float32)
        if self.use_h:
            out += self.Ph[:, cur[0]] @ cur[1]
        if self.use_z and z is not None:
            out += self.Pz[:, z[0]] @ z[1]
        return out

    # ---- interface ----
    def predict_next(self):
        if self.pred is None or not np.any(self.pred > 0) or not np.any(self.s.nc > 0):
            return np.full(10, 0.1)
        seen = np.where(self.s.nc > 0)[0]
        sc = _unit(self.s.A[seen]) @ _unit(np.maximum(self.pred, 0))
        p = np.full(10, 0.0)
        p[seen] = _softmax(self.beta_pred * sc)
        return p

    def _step(self, x, y, learn):
        if learn:
            self.s.learn(x, np.array([y]))
        if self.U is None:
            self._init()
        cur = self._sparse(x)
        # surprise at the observed item, from the prediction made at the previous step
        if self.pred is not None and np.any(self.pred != 0):
            obs = np.zeros(self.s.N, np.float32); obs[cur[0]] = cur[1]
            self._s = 1.0 - float(_unit(obs) @ _unit(self.pred))
            if learn:   # delta rule on the inputs that made the prediction (stored from the previous step)
                err = obs - self.pred
                pc, pz = self._last_inputs
                if self.use_h:
                    eh = np.maximum(1.0 / (1.0 + self.nhu[pc[0]]), self.eta_min) if self.meta else self.eta
                    self.Ph[:, pc[0]] += np.outer(err, pc[1]) * eh
                    self.nhu[pc[0]] += 1
                if self.use_z and pz is not None:
                    ez = np.maximum(1.0 / (1.0 + self.nzu[pz[0]]), self.eta_min) if self.meta else self.eta
                    self.Pz[:, pz[0]] += np.outer(err, pz[1]) * ez
                    self.nzu[pz[0]] += 1
        else:
            self._s = 1.0
            if learn and self._last_inputs is not None and self.pred is not None:
                obs = np.zeros(self.s.N, np.float32); obs[cur[0]] = cur[1]
                pc, pz = self._last_inputs
                if self.use_h:
                    self.Ph[:, pc[0]] += self.eta * np.outer(obs, pc[1])
                if self.use_z and pz is not None:
                    self.Pz[:, pz[0]] += self.eta * np.outer(obs, pz[1])
        if self.ctx == 'q':
            self._qp = getattr(self, '_qc', None)
            self._qc = self._q(cur)
        z = self._z(cur, self.prev) if (self.ctx != 'q' or self._qp is not None) else None
        self.pred = self._predict(cur, z)
        self._last_inputs = (cur, z)
        self.prev = cur

    def observe(self, x, y):
        self._step(x, y, learn=True)

    def observe_frozen(self, x, y):
        self._step(x, y, learn=False)

    def surprise_last(self):
        return self._s

    _last_inputs = None


class CtxSeqSPARC(SeqSPARC):
    """S2-3: context (latent-cause) inference over sequence predictors.

    Each context slot k has its own predictor of the next *category* activity, q̂ = P_k z (z = sparse conjunction
    of current and previous category activity). Every step each slot's prediction quality is tracked as an
    exponential average of the log-probability it gave the observed category; the best slot is the active context
    and the only one that learns (delta rule, metaplastic rate). If even the best slot has been predicting badly
    for a while, a fresh slot is recruited (vigilance at the context level, like ART recruitment of units).
    Brain basis: latent-cause / contextual inference (Gershman, Blei & Niv 2010; Heald, Lengyel & Wolpert 2021),
    hippocampal remapping. research/17.
    """

    def __init__(self, seed=0, K=8, lam=0.2, ctx_vig=-1.6, grace=30, min_age=0, seg_thr=None, seg_age=200, **kw):
        kw.setdefault('ctx', 'q'); kw.setdefault('use_h', False); kw.setdefault('meta', True)
        super().__init__(seed=seed, **kw)
        self.K, self.lam, self.ctx_vig, self.grace = K, lam, ctx_vig, grace
        self.min_age = min_age   # a context must have learned this many steps before its failure can recruit a new one
        self.n_ctx_new = 0
        # event segmentation from surprise (Zacks et al. 2007 event segmentation theory; DuBrow & Davachi 2013:
        # less binding across boundaries; research/24): when an established context badly fails to predict the
        # item, the model marks an event boundary itself, does not learn that transition, and starts afresh
        self.seg_thr, self.seg_age, self.n_seg = seg_thr, seg_age, 0

    def _init(self):
        super()._init()
        self.Pk = np.zeros((self.K, 10, self.nz), np.float32)
        self.nk = np.zeros((self.K, self.nz))
        self.used_k = 1
        self.score = np.zeros(self.K)       # running log-likelihood per context
        self.active = 0
        self.since_new = 0
        self.age = np.zeros(self.K)

    def reset_context(self):
        super().reset_context()
        if getattr(self, 'score', None) is not None:
            self.score[:] = 0.0
        self.qpred = None

    def predict_next(self):
        if getattr(self, 'qpred', None) is None:
            return np.full(10, 0.1)
        p = np.maximum(self.qpred[self.active], 0) + 1e-3
        return p / p.sum()

    def _step(self, x, y, learn):
        if learn:
            self.s.learn(x, np.array([y]))
        if self.U is None:
            self._init()
        cur = self._sparse(x)
        q_obs = self._q(cur)[1]
        if self.qpred is not None:
            # every context scores how well it predicted the observed category
            pk = np.maximum(self.qpred[:self.used_k], 0) + 1e-3
            pk = pk / pk.sum(1, keepdims=True)
            ll = np.log(pk @ q_obs + 1e-6)
            self.score[:self.used_k] = (1 - self.lam) * self.score[:self.used_k] + self.lam * ll
            self._s = 1.0 - float(pk[self.active] @ q_obs)
            boundary = (self.seg_thr is not None and learn and self.age[self.active] >= self.seg_age
                        and float(pk[self.active] @ q_obs) < self.seg_thr)
            if boundary:
                self.n_seg += 1
                self.score[:] = 0.0                  # new event: context evidence starts afresh
                self._qc = None                      # no binding across the boundary
                self.prev = None
            elif learn and self._last_z is not None:
                self.since_new += 1
                best = int(np.argmax(self.score[:self.used_k]))
                if (self.score[best] < self.ctx_vig and self.since_new > self.grace and self.used_k < self.K
                        and self.age[best] >= self.min_age):
                    k = self.used_k; self.used_k += 1          # persistent surprise -> a new context (latent cause)
                    self.score[k] = 0.0; best = k; self.since_new = 0; self.n_ctx_new += 1
                self.active = best
                zi, zv = self._last_z
                err = q_obs - self.qpred[self.active]
                ez = np.maximum(1.0 / (1.0 + self.nk[self.active, zi]), self.eta_min)
                self.Pk[self.active][:, zi] += np.outer(err, zv * ez)
                self.nk[self.active, zi] += 1
                self.age[self.active] += 1
            else:
                self.active = int(np.argmax(self.score[:self.used_k]))
        else:
            self._s = 1.0
        self._qp = getattr(self, '_qc', None)
        self._qc = (np.arange(10), q_obs)
        z = self._z(cur, self.prev) if self._qp is not None else None
        self._last_z = z
        if z is not None:
            self.qpred = np.stack([self.Pk[k][:, z[0]] @ z[1] for k in range(self.K)])
        else:
            self.qpred = None
        self.prev = cur

    _last_z = None
