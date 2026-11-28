"""R4-diag6: few-shot vs old-class trade-off curves of recall-time mechanisms, on one trained model.

Repeats bench.py's 5-shot part (tasks 1-4, then 5 labelled shots each of classes 8, 9 + one sleep; same RNG draws
as bench, so the default setting must reproduce bench's FEW5 / FEW5_old_drop), then evaluates a grid of recall-only
settings (no learning at test, so one trained model serves every point):
  * excitability bias at recall: ie_gamma at test in {0, 1/4, 1/2, 1} x trained value (0 = ie_recall=False),
  * hippocampal recall vote (research/16): off, or CLS-gated (epi=1, epi_gate=50) with recall-vs-familiarity
    margin delta in {0.1, 0.2, 0.3, 0.5, none = ungated by familiarity}.
For each point: FEW5 (classes 8, 9 among all 10) and old-class drop = base old acc (same recall setting, base model,
old classes allowed) - old acc after the shots (all classes allowed). Lower drop / higher FEW5 better.
Writes results/fewcurve_<tag>_<data>_<seed>_val.json.

    python tools/fewcurve.py --data mnist --seed 0 --tag shy --set ie_sleep=down
"""
import argparse, copy, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from data import load, TASKS
from bench import make_model, run_stream, acc_on
from fewdiag import parse_set

GSCALE = (0.0, 0.25, 0.5, 1.0)
EPI = (None, 0.1, 0.2, 0.3, 0.5, 'nomargin')


def apply(m, gs, ep, g0):
    M = copy.copy(m)                     # recall settings are plain attributes; arrays are not changed at test
    M.ie_gamma = g0 * gs
    if ep is None:
        M.epi = 0.0
    else:
        M.epi, M.epi_gate = 1.0, 50.0
        M.epi_margin = None if ep == 'nomargin' else float(ep)
        M.epi_temp = 0.02
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='mnist'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='defaults'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--cap', type=int, default=None, help='images per class (smoke test only)')
    a = ap.parse_args()
    ov = parse_set(a.set)
    Xtr, ytr, Xev, yev = load(a.data, 'val')
    np.random.seed(a.seed)
    all_cls, old, new = list(range(10)), list(range(8)), [8, 9]
    base = make_model('sparc', a.seed, ov, dim=Xtr.shape[1])
    run_stream(base, Xtr, ytr, Xev, yev, a.cap, tasks=TASKS[:4], log=False)
    rng = np.random.RandomState(a.seed + 100)
    pool8, pool9 = np.where(ytr == 8)[0], np.where(ytr == 9)[0]
    for k in (1, 5):                                   # same draw order as bench.py (k=1 first)
        i8, i9 = rng.choice(pool8, k, replace=False), rng.choice(pool9, k, replace=False)
    m = copy.deepcopy(base)
    for i in np.stack([i8, i9], 1).ravel():
        m.learn(Xtr[i:i + 1], ytr[i:i + 1])
    m.end_task()
    g0 = base.ie_gamma
    rec = base.ie_recall
    base.ie_recall = m.ie_recall = True                # the grid sets the recall bias through ie_gamma
    pts = []
    for gs in GSCALE:
        for ep in EPI:
            B, M = apply(base, gs, ep, g0), apply(m, gs, ep, g0)
            bo = acc_on(B, Xev, yev, old, old)
            ao = acc_on(M, Xev, yev, old, all_cls)
            f5 = acc_on(M, Xev, yev, new, all_cls)
            pts.append(dict(g_scale=gs, epi_margin=ep, base_old=bo, after_old=ao, old_drop=bo - ao, FEW5=f5))
            print(f'g x{gs:<5} epi {str(ep):9} FEW5 {100 * f5:5.1f}  old drop {100 * (bo - ao):5.2f}', flush=True)
    res = dict(data=a.data, seed=a.seed, split='val', tag=a.tag, overrides=ov, trained_ie_recall=rec, ie_gamma=g0, points=pts)
    out = os.path.join(ROOT, 'results', f'fewcurve_{a.tag}_{a.data}_{a.seed}_val' + (f'_cap{a.cap}' if a.cap else '') + '.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
