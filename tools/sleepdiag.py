"""R4-diag3: what does one night's sleep change right after relearning? Fashion val seed 1, defaults (console only)."""
import sys, copy, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from data import load, task_indices, TASKS
from bench import make_model, feed, acc_on
seed = 1; Xtr, ytr, Xev, yev = load('fashion', 'val'); np.random.seed(seed)
all_cls = list(range(10))
stream_c, relearn_c = {}, {}
for c in all_cls:
    ic = task_indices(ytr, [c], None); k = int(0.9 * len(ic)); h = (len(ic) - k) // 2
    stream_c[c], relearn_c[c] = ic[:k], ic[k + h:]
m = make_model('sparc', seed, {}, dim=Xtr.shape[1])
for t, task in enumerate(TASKS):
    feed(m, Xtr, ytr, np.sort(np.concatenate([stream_c[c] for c in task]))); m.end_task()
jt = (2, 3); prng = np.random.RandomState(seed + 500)
pi = prng.choice(np.where(np.isin(yev, jt))[0], 200, replace=False)
def acc(M, mix=None):
    old = M.mix
    if mix: M.mix = mix
    v = float((M.predict(Xev[pi], all_cls) == yev[pi]).mean()); M.mix = old; return v
others = [c for c in all_cls if c not in jt]
print('forgotten: mix %.3f fast %.3f slow %.3f' % (acc(m), acc(m, 'fast'), acc(m, 'slow')))
w0 = m.n_writes
feed(m, Xtr, ytr, np.sort(np.concatenate([relearn_c[c] for c in jt])))
lab = m.h_lab[:m.h_n]; ht = m.h_t[:m.h_n] if hasattr(m, 'h_t') else None
print('relearned: mix %.3f fast %.3f slow %.3f | hip writes during relearning %d | traces of relearned classes %s of %d' % (
    acc(m), acc(m, 'fast'), acc(m, 'slow'), m.n_writes - w0, [int((lab == c).sum()) for c in jt], m.h_n))
print('other classes before sleep %.4f' % acc_on(m, Xev, yev, others, all_cls))
variants = {'normal': {}, 'no_clean': {'sleep_clean': False}, 'no_downscale': {'downscale': 0.0},
            'no_merge': {'merge_frac': 0.0}, 'no_replay': {'M_skip': True}}
for name, ch in variants.items():
    M = copy.deepcopy(m)
    for k, v in ch.items():
        if k != 'M_skip': setattr(M, k, v)
    if ch.get('M_skip'):
        M.sleep = lambda: None
    M.end_task()
    print('%-13s after sleep: mix %.3f fast %.3f slow %.3f | others %.4f | slept=%s' % (
        name, acc(M), acc(M, 'fast'), acc(M, 'slow'), acc_on(M, Xev, yev, others, all_cls), getattr(M, 'slept', None)))
