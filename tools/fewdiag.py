"""R4-diag4: why do old classes drop after few-shot learning under SHY sleep?

Repeats bench.py's few-shot part (tasks 1-4, then 5 labelled shots each of classes 8, 9 + one sleep; same RNG draws,
so FEW5_old_drop must match the bench JSON), then looks inside the model after the 5-shot sleep:
  * where the newly wrong old-class images go (to 8/9 or to another old class),
  * which IT units win them (units whose readout says 8/9 = "new units") and those units' excitability bias,
  * test-time what-ifs that change only excitability (no learning): no bias at all; new units' bias set neutral;
    old units' SHY down-regulation undone (r_eff = r); and the readout restricted to old classes.
Writes results/fewdiag_<tag>_<data>_<seed>_val.json.

    python tools/fewdiag.py --data mnist --seed 0 --tag both --set ie_sleep=down merge_protect=used
"""
import argparse, copy, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data import load, TASKS
from bench import make_model, run_stream, acc_on


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        else:
            v = {'True': True, 'False': False, 'None': None}.get(v, v)
        out[k] = v
    return out


def unit_class(m):
    return m.A[:, :m.used].argmax(0)


def bias(m):
    b = m._bias(m.used)
    return np.zeros(m.used, np.float32) if np.isscalar(b) else np.asarray(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='mnist'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='defaults'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--cap', type=int, default=None, help='images per class (smoke test only; bench uses none)')
    a = ap.parse_args()
    ov = parse_set(a.set)
    Xtr, ytr, Xev, yev = load(a.data, 'val')
    np.random.seed(a.seed)
    all_cls, old, new = list(range(10)), list(range(8)), [8, 9]
    base = make_model('sparc', a.seed, ov, dim=Xtr.shape[1])
    run_stream(base, Xtr, ytr, Xev, yev, a.cap, tasks=TASKS[:4], log=False)
    base_old = acc_on(base, Xev, yev, old, old)
    rng = np.random.RandomState(a.seed + 100)
    pool8, pool9 = np.where(ytr == 8)[0], np.where(ytr == 9)[0]
    for k in (1, 5):                                   # same draw order as bench.py (k=1 first)
        i8, i9 = rng.choice(pool8, k, replace=False), rng.choice(pool9, k, replace=False)
    m = copy.deepcopy(base)
    for i in np.stack([i8, i9], 1).ravel():
        m.learn(Xtr[i:i + 1], ytr[i:i + 1])
    m.end_task()
    mo = np.isin(yev, old)
    Xo, yo = Xev[mo], yev[mo]
    p0 = base.predict(Xo, old)                         # base model knows only old classes (bench: allowed = old)
    p1 = m.predict(Xo, all_cls)
    flipped = (p0 == yo) & (p1 != yo)
    res = dict(data=a.data, seed=a.seed, split='val', tag=a.tag, overrides=ov,
               base_old_acc=base_old, after_old_acc=float((p1 == yo).mean()),
               FEW5=acc_on(m, Xev, yev, new, all_cls))
    res['FEW5_old_drop'] = base_old - res['after_old_acc']
    res['n_old_test'] = int(mo.sum()); res['n_flipped'] = int(flipped.sum())
    res['n_gained'] = int(((p0 != yo) & (p1 == yo)).sum())
    res['flipped_to_new_frac'] = float(np.isin(p1[flipped], new).mean()) if flipped.any() else None
    # units and their excitability after the 5-shot sleep
    uc = unit_class(m); b = bias(m); isnew = np.isin(uc, new)
    r = m.r_eff[:m.used] if m.ie_sleep else m.r[:m.used]
    res['units'] = dict(used=int(m.used), n_new=int(isnew.sum()), n_new_recruited=int(m.used - base.used),
                        bias_old_mean=float(b[~isnew].mean()), bias_new_mean=float(b[isnew].mean()) if isnew.any() else None,
                        bias_old_frac_negative=float((b[~isnew] < 0).mean()),
                        r_old_mean=float(r[~isnew].mean()), r_new_mean=float(r[isnew].mean()) if isnew.any() else None,
                        r_target=float(m.k / m.used))
    # winners on the flipped images: share of new units among the k winners, before vs after
    H, _ = m.encode(Xo)
    win_new = (H[:, :m.used][:, isnew] > 0).sum(1) / max(1, m.k)
    res['winners_new_share_flipped'] = float(win_new[flipped].mean()) if flipped.any() else None
    res['winners_new_share_all_old'] = float(win_new.mean())
    # test-time what-ifs (excitability only; no learning)
    def old_acc(M, allowed=all_cls):
        return float((M.predict(Xo, allowed) == yo).mean())
    wi = {}
    M = copy.deepcopy(m); M.ie_gamma = 0.0; wi['no_bias'] = old_acc(M)
    M = copy.deepcopy(m); tgt = M.k / M.used
    for arr in (M.r, M.r_eff):
        arr[:M.used][isnew] = tgt
    wi['new_units_neutral'] = old_acc(M)
    if m.ie_sleep:
        M = copy.deepcopy(m); M.r_eff[:M.used][~isnew] = M.r[:M.used][~isnew]; wi['old_shy_undone'] = old_acc(M)
        M = copy.deepcopy(m); M.r_eff[:M.used] = M.r[:M.used]; wi['all_shy_undone'] = old_acc(M)
    wi['readout_old_only'] = old_acc(m, old)
    res['whatif_old_acc'] = wi
    res['whatif_old_drop'] = {k: base_old - v for k, v in wi.items()}
    # same what-ifs on the base model (before few-shot) for reference
    M = copy.deepcopy(base); M.ie_gamma = 0.0
    res['base_old_acc_no_bias'] = float((M.predict(Xo, old) == yo).mean())
    out = os.path.join(ROOT, 'results', f'fewdiag_{a.tag}_{a.data}_{a.seed}_val' + (f'_cap{a.cap}' if a.cap else '') + '.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1)); print('wrote', out)


if __name__ == '__main__':
    main()
