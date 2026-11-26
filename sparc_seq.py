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
