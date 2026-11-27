"""Stage 3 (first step): number sense, after Dehaene's triple-code model (research/20).

  visual digit form    stage-1 SPARC recognises digit images (trained briefly on MNIST at setup)
  magnitude (ANS)      innate approximate number system: center-surround dot detectors (retina/LGN-like),
                       individuated by local competition, summed, and read by numerosity-tuned units on a log
                       scale (Dehaene & Changeux 1993; Nieder & Miller 2003); decisions carry Weber noise
  grounding            Hebbian links digit symbol ↔ numerosity code, learned from (dot array, digit) pairings
  verbal facts         trained addition facts are memorised (associative), like the verbal code
  comparison           on the grounded magnitudes (noisy) → a distance effect is expected
  addition (unseen)    magnitudes of a and b are combined (summation in the accumulator), then read back as a
                       digit through the grounding links → generalises to pairs never practised
No backprop; all learned links are local Hebbian / running means.
"""
import numpy as np
from sparc import SPARC, _unit


def _conv2(img, k):
    """'same' 2-D convolution via shifted sums (small kernel)."""
    r = k.shape[0] // 2
    pad = np.pad(img, r)
    out = np.zeros_like(img)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            if k[dy, dx] != 0:
                out += k[dy, dx] * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def _dog(sc, ss, size=9):
    y, x = np.mgrid[-(size // 2):size // 2 + 1, -(size // 2):size // 2 + 1]
    g = lambda s: np.exp(-(x ** 2 + y ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    return (g(sc) - g(ss)).astype(np.float32)


class NumSPARC:
    def __init__(self, seed=0, weber=0.12, n_pref=20, stage1_n=3000, thr=0.25, route='count', **stage1):
        self.rng = np.random.RandomState(seed)
        self.weber, self.thr = weber, thr
        self.pref = np.arange(1, n_pref + 1)                 # numerosity-tuned units (preferred numbers)
        self.G = np.zeros((10, n_pref), np.float32)          # digit symbol -> numerosity code (grounding)
        self.ng = np.zeros(10)
        self.facts = {}                                      # verbal arithmetic facts
        self.route = route                                   # 'count': counting-on; 'ans': magnitude sum
        self.T = np.zeros((10, 10))                          # learned successor transitions (Hebbian counts)
        self.stage1_n = stage1_n
        self.s = SPARC(seed=seed, glimpses=1, **stage1)      # visual digit form
        self.kernels = [_dog(1.0, 2.5), _dog(1.6, 3.5)]      # center-surround detectors at two scales

    # ---- visual digit form ----
    def setup(self, imgs, rng):
        Xtr, pools, Xev, epools = imgs
        self.Xev, self.epools = Xev, epools
        idx = np.sort(np.concatenate([pools[c][:self.stage1_n // 10] for c in range(10)]))
        ytr = np.empty(len(Xtr), int)
        for c in range(10):
            ytr[pools[c]] = c
        for i in idx:                                        # brief stage-1 learning so digits are recognised
            self.s.learn(Xtr[i:i + 1], ytr[i:i + 1])
        self.s.end_task()

    def read_digit(self, d):
        """Recognise a fresh image of digit d with the stage-1 readout (the model never sees the true label)."""
        i = self.epools[d][self.rng.randint(len(self.epools[d]))]
        return int(self.s.predict(self.Xev[i:i + 1], list(range(10)))[0])

    # ---- magnitude: approximate number system ----
    def numerosity(self, img):
        """Estimate of the number of items: center-surround responses, local competition, summation."""
        I = img.reshape(28, 28)
        best = None
        for k in self.kernels:
            R = _conv2(I, k)
            # an item = a local maximum of the detector response (3x3 neighbourhood) above threshold
            P = np.pad(R, 1, constant_values=-np.inf)
            nb = np.stack([P[1 + dy:29 + dy, 1 + dx:29 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
            peaks = (R > nb.max(0)) & (R > self.thr * R.max())
            n = int(peaks.sum())
            best = n if best is None else max(best, n) if n <= 12 else best
        return max(best, 1)

    def code(self, n):
        """Numerosity-tuned population on a log scale (Weber-like tuning width)."""
        return np.exp(-(np.log(self.pref) - np.log(max(n, 1e-3))) ** 2 / (2 * (2 * self.weber) ** 2)).astype(np.float32)

    def magnitude(self, d):
        """Expected numerosity linked to digit symbol d (population-vector read-out of the grounding)."""
        g = self.G[d]
        return float((g * self.pref).sum() / (g.sum() + 1e-8)) if g.sum() > 0 else 0.0

    # ---- benchmark interface ----
    def train_ground_img(self, dot_img, digit):
        r = self.code(self.numerosity(dot_img))
        d = self.read_digit(digit)                           # the digit as recognised by stage 1
        self.ng[d] += 1
        self.G[d] += (r - self.G[d]) / self.ng[d]            # Hebbian running mean: symbol ↔ magnitude

    def ground_img(self, dot_img):
        r = _unit(self.code(self.numerosity(dot_img)))
        return int(np.argmax(_unit(self.G) @ r))

    def train_count(self, seq):
        """Successor knowledge from counting: Hebbian transitions between recognised digit symbols."""
        rec = [self.read_digit(d) for d in seq]
        for x, y in zip(rec[:-1], rec[1:]):
            self.T[x, y] += 1

    def succ(self, d):
        return int(np.argmax(self.T[d])) if self.T[d].sum() > 0 else None

    def larger(self, a, b):
        da, db = self.read_digit(a), self.read_digit(b)
        ma, mb = self.magnitude(da), self.magnitude(db)
        # noisy comparison on a log scale (Weber noise) -> distance effect
        na = np.log(max(ma, 1e-3)) + self.weber * self.rng.randn()
        nb = np.log(max(mb, 1e-3)) + self.weber * self.rng.randn()
        return a if na > nb else b

    def train_add(self, a, b, s):
        self.facts[(self.read_digit(a), self.read_digit(b))] = s

    def add(self, a, b):
        da, db = self.read_digit(a), self.read_digit(b)
        if (da, db) in self.facts:                           # verbal-fact route
            return self.facts[(da, db)]
        if self.route == 'count' and self.T.sum() > 0:
            # counting-on: two working-memory registers (running total, steps taken); step with the learned
            # successor until the step register matches b (the routine is built in: PFC/BG control stand-in)
            total, steps = da, None
            for _ in range(9):
                nt = self.succ(total)
                steps = 1 if steps is None else self.succ(steps)
                if nt is None or steps is None:
                    break
                total = nt
                if steps == db:
                    return total
            return total
        m = self.magnitude(da) + self.magnitude(db)          # magnitude route: accumulate both quantities
        r = _unit(self.code(m))
        return int(np.argmax(_unit(self.G) @ r))


class TISPARC:
    """Transitive inference by a learned ordinal map (stage 3; research/17).

    Objects are recognised by stage-1 SPARC (Fashion). Each recognised object has a 'position' on a 1-D map
    (a value-like scalar), learned locally from choice feedback: after each trial the chosen/unchosen items'
    positions move apart by a prediction error (Rescorla–Wagner-style value transfer, as in models of animal
    transitive inference; cf. ordinal / rank coding in hippocampus and PFC). Choice = the item further up the
    map (with decision noise). Novel pairs are answered from the map, never from stored pairs.
    """
    def __init__(self, seed=0, lr=0.15, noise=0.05, stage1_n=6000, oracle=False, linked=True, margin=0.3, map_age=50, assimilate=True, **stage1):
        self.rng = np.random.RandomState(seed); self.lr, self.noise, self.stage1_n = lr, noise, stage1_n
        self.oracle = oracle      # True: use the true object identity (isolates relational learning from perception)
        self.pos = np.zeros(32)                # position of each item within its map
        # structure vs content (Tolman-Eichenbaum-style): items belong to maps (lists = connected groups of
        # items learned together); each map has its own offset. A fact linking two maps moves the whole map.
        self.linked, self.margin = linked, margin
        self.parent = {}                       # union-find over items trained together -> map membership
        self.off = {}                          # offset of each map
        self.age = {}                          # learning steps per map; only established maps link as wholes
        self.map_age = map_age
        # schema assimilation (Tse et al. 2007; research/24): a new item met together with an item of an
        # established map is placed into that map in one shot; the map itself does not move. Each new item
        # keeps only two numbers: the highest value it beat (lo) and the lowest value that beat it (hi).
        self.assimilate = assimilate
        self.lo, self.hi = {}, {}
        self.s = SPARC(seed=seed, glimpses=1, **stage1)

    def setup(self, Xtr, ytr, rng):
        if self.oracle:
            return
        idx = np.sort(np.concatenate([np.where(ytr == c)[0][:self.stage1_n // 10] for c in range(10)]))
        for i in idx:
            self.s.learn(Xtr[i:i + 1], ytr[i:i + 1])
        self.s.end_task()

    def _rec(self, x):
        return int(self.s.predict(x, list(range(10)))[0])

    def choose(self, i, j, xi, xj):
        a, b = (i, j) if self.oracle else (self._rec(xi), self._rec(xj))
        da = self.value(a) + self.noise * self.rng.randn(); db = self.value(b) + self.noise * self.rng.randn()
        return i if da >= db else j

    def feedback(self, i, j, xi, xj, winner):
        a, b = (i, j) if self.oracle else (self._rec(xi), self._rec(xj))
        w, l = (a, b) if winner == i else (b, a)
        if w == l:
            return
        p = 1.0 / (1.0 + np.exp(-(self.value(w) - self.value(l)) / 0.1))  # predicted P(w above l)
        err = 1.0 - p                                                      # prediction error (dopamine-like)
        if (self.linked and self._mapped(w) and self._mapped(l) and self._map(w) != self._map(l)
                and self.age.get(self._map(w), 0) >= self.map_age and self.age.get(self._map(l), 0) >= self.map_age):
            # the two items live in different, already-formed maps: the error moves the whole maps
            # one-shot: shift both maps just enough that the linking fact holds with a margin
            mw, ml = self._map(w), self._map(l)
            need = self.value(l) - self.value(w) + self.margin
            if need > 0:
                self.off[mw] = self.off.get(mw, 0.0) + need / 2
                self.off[ml] = self.off.get(ml, 0.0) - need / 2
            return
        if self.assimilate and self.linked:
            est = lambda x: (self._mapped(x) and x not in self.lo and x not in self.hi
                             and self.age.get(self._map(x), 0) >= self.map_age)
            newish = lambda x: not self._mapped(x) or x in self.lo or x in self.hi
            if (est(l) and newish(w)) or (est(w) and newish(l)):
                new, old = (w, l) if est(l) and newish(w) else (l, w)
                v = self.value(old)
                if new == w:
                    self.lo[new] = max(self.lo.get(new, -np.inf), v)
                else:
                    self.hi[new] = min(self.hi.get(new, np.inf), v)
                lo, hi = self.lo.get(new, -np.inf), self.hi.get(new, np.inf)
                tgt = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (lo + self.margin if np.isfinite(lo) else hi - self.margin)
                r = self._map(old)
                self.pos[new] = tgt - self.off.get(r, 0.0)     # one-shot placement in map coordinates
                self.parent[new] = r                            # joins the map; the map does not move
                return
        if self.linked:
            self._union(w, l)
            r = self._map(w); self.age[r] = self.age.get(r, 0) + 1
        self.pos[w] += self.lr * err
        self.pos[l] -= self.lr * err

    # ---- maps (structure) ----
    def _find(self, x):
        while self.parent.get(x, x) != x:
            x = self.parent[x]
        return x

    def _mapped(self, x):
        return x in self.parent

    def _map(self, x):
        return self._find(x)

    def _union(self, x, y):
        """Merge two forming maps; members of the absorbed map keep their current values."""
        for z in (x, y):
            self.parent.setdefault(z, z)
        rx, ry = self._find(x), self._find(y)
        if rx == ry:
            return
        shift = self.off.get(ry, 0.0) - self.off.get(rx, 0.0)
        for z in list(self.parent):
            if self._find(z) == ry:
                self.pos[z] += shift
        self.parent[ry] = rx
        self.age[rx] = self.age.get(rx, 0) + self.age.pop(ry, 0)
        self.off.pop(ry, None)

    def value(self, x):
        v = self.pos[x]
        if self.linked and self._mapped(x):
            v += self.off.get(self._map(x), 0.0)
        return v
