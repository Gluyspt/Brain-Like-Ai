"""Benchmark protocol for brain-like learning (docs/benchmark.md).

python bench.py --model sparc --data mnist --seed 0 --split val --quick [--set key=val ...] [--tag name]

Metrics: ACC, FGT, FEW-1/FEW-5 (+ old-class drop), LBL5, SPARSITY, NOVELTY (AUROC), wall-clock, params, stored numbers.
Results -> results/<model>[_<tag>]_<data>_<seed>_<split>[_quick].json
"""
import argparse, copy, json, os, time
import numpy as np
from data import load, task_indices, TASKS


def make_model(name, seed, overrides, dim=784):
    if name == 'sparc':
        from sparc import SPARC
        return SPARC(seed=seed, **overrides)
    from baselines import make_baseline
    return make_baseline(name, seed, dim=dim)


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    if v in ('True', 'False'):
        return v == 'True'
    if v == 'None':
        return None
    return v


def feed(model, X, y, idx, label_mask=None):
    """Present samples in stream order with the model's batch size. label_mask[i]=False -> unlabelled."""
    b = model.batch_size
    for s in range(0, len(idx), b):
        j = idx[s:s + b]
        if label_mask is None:
            model.learn(X[j], y[j])
        else:
            m = label_mask[s:s + b]
            if b == 1:
                model.learn(X[j], y[j] if m[0] else None)
            else:  # batch models: only labelled samples are usable
                if m.any():
                    model.learn(X[j[m]], y[j[m]])


def acc_on(model, X, y, classes, allowed):
    m = np.isin(y, classes)
    if m.sum() == 0:
        return float('nan')
    return float((model.predict(X[m], list(allowed)) == y[m]).mean())


def auroc(neg, pos):
    """P(score_pos > score_neg), ties count half (Mann-Whitney)."""
    s = np.concatenate([neg, pos])
    order = s.argsort(kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_stream(model, Xtr, ytr, Xev, yev, cap, label_frac=None, rng=None, tasks=TASKS, log=True):
    nt = len(tasks)
    accm = np.full((nt, nt), np.nan)
    seen = []
    for t, task in enumerate(tasks):
        idx = task_indices(ytr, task, cap)
        mask = None
        if label_frac is not None:
            mask = rng.rand(len(idx)) < label_frac
        t0 = time.time()
        feed(model, Xtr, ytr, idx, mask)
        model.end_task()
        seen += list(task)
        for j in range(t + 1):
            accm[t, j] = acc_on(model, Xev, yev, tasks[j], seen)
        if log:
            print(f'  task {t} {task}: n={len(idx)} {time.time()-t0:.1f}s  acc_per_task=' +
                  ' '.join(f'{a:.3f}' for a in accm[t, :t + 1]), flush=True)
    return accm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sparc')
    ap.add_argument('--data', default='mnist')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--quick', action='store_true', help='cap train at 1000 samples per class')
    ap.add_argument('--cap', type=int, default=None, help='cap train at N samples per class (data-scaling runs)')
    ap.add_argument('--set', nargs='*', default=[], help='model overrides key=val')
    ap.add_argument('--tag', default='')
    ap.add_argument('--only', default='main,few,lbl', help='subset of main,few,lbl,rev')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()

    overrides = dict(kv.split('=', 1) for kv in a.set)
    overrides = {k: parse_val(v) for k, v in overrides.items()}
    np.random.seed(a.seed)
    Xtr, ytr, Xev, yev = load(a.data, a.split)
    cap = a.cap if a.cap else (1000 if a.quick else None)
    parts = a.only.split(',')
    res = dict(model=a.model, data=a.data, seed=a.seed, split=a.split, quick=a.quick, cap=cap,
               overrides=overrides, tag=a.tag)
    T0 = time.time()
    all_cls = list(range(10))

    # ---- main stream: ACC, FGT, SPARSITY ----
    if 'main' in parts:
        print('[main stream]', flush=True)
        model = make_model(a.model, a.seed, overrides, dim=Xtr.shape[1])
        if a.model == 'b3':
            idx = np.concatenate([task_indices(ytr, t, cap) for t in TASKS])
            model.fit_joint(Xtr[idx], ytr[idx], epochs=10)
            res['ACC'] = acc_on(model, Xev, yev, all_cls, all_cls)
            res['FGT'] = None
        else:
            accm = run_stream(model, Xtr, ytr, Xev, yev, cap)
            res['acc_matrix'] = accm.tolist()
            res['ACC'] = acc_on(model, Xev, yev, all_cls, all_cls)
            res['FGT'] = float(np.mean([np.nanmax(accm[j:, j]) - accm[-1, j] for j in range(len(TASKS) - 1)]))
        res['per_class_acc'] = [acc_on(model, Xev, yev, [c], all_cls) for c in all_cls]
        if hasattr(model, 'sparsity'):
            res['SPARSITY'] = model.sparsity(Xev)
        res['stats'] = model.stats()
        if hasattr(model, 'diagnostics'):
            res['diagnostics'] = model.diagnostics()
        print(f"  ACC={res['ACC']:.4f} FGT={res['FGT']}", flush=True)

    # ---- few-shot + novelty: learn tasks 1-4, then k labelled shots of classes 8, 9 ----
    if 'few' in parts and a.model != 'b3':
        print('[few-shot / novelty]', flush=True)
        model = make_model(a.model, a.seed, overrides, dim=Xtr.shape[1])
        run_stream(model, Xtr, ytr, Xev, yev, cap, tasks=TASKS[:4], log=False)
        old = list(range(8))
        base_old = acc_on(model, Xev, yev, old, old)
        res['FEW_base_old_acc'] = base_old
        if hasattr(model, 'surprise'):
            mo, mn = np.isin(yev, old), np.isin(yev, [8, 9])
            s_old, s_new = model.surprise(Xev[mo]), model.surprise(Xev[mn])
            res['NOVELTY'] = auroc(s_old, s_new)
            res['surprise_mean_old_new'] = [float(s_old.mean()), float(s_new.mean())]
        rng = np.random.RandomState(a.seed + 100)
        pool8 = np.where(ytr == 8)[0]
        pool9 = np.where(ytr == 9)[0]
        for k in (1, 5):
            m = copy.deepcopy(model)
            i8 = rng.choice(pool8, k, replace=False)
            i9 = rng.choice(pool9, k, replace=False)
            order = np.stack([i8, i9], 1).ravel()  # 8,9,8,9,...
            for i in order:
                m.learn(Xtr[i:i + 1], ytr[i:i + 1])
            m.end_task()
            res[f'FEW{k}'] = acc_on(m, Xev, yev, [8, 9], all_cls)
            res[f'FEW{k}_old_drop'] = base_old - acc_on(m, Xev, yev, old, all_cls)
            print(f"  FEW{k}={res[f'FEW{k}']:.4f} old_drop={res[f'FEW{k}_old_drop']:.4f}", flush=True)
        if 'NOVELTY' in res:
            print(f"  NOVELTY AUROC={res['NOVELTY']:.4f}", flush=True)

    # ---- few labels: 5% of stream labelled ----
    if 'lbl' in parts and a.model != 'b3':
        print('[5% labels]', flush=True)
        model = make_model(a.model, a.seed, overrides, dim=Xtr.shape[1])
        rng = np.random.RandomState(a.seed + 200)
        run_stream(model, Xtr, ytr, Xev, yev, cap, label_frac=0.05, rng=rng, log=False)
        res['LBL5'] = acc_on(model, Xev, yev, all_cls, all_cls)
        print(f"  LBL5={res['LBL5']:.4f}", flush=True)

    # ---- self-correction (P10): reversal + label noise on tasks 1-2 (classes 0-3) ----
    if 'rev' in parts and a.model != 'b3':
        print('[reversal / noise]', flush=True)
        cls4 = [0, 1, 2, 3]
        halves = {}
        for t in TASKS[:2]:
            idx = task_indices(ytr, t, cap)
            halves[t] = (idx[:len(idx) // 2], idx[len(idx) // 2:])
        # REVERSAL: learn tasks 1-2 on first halves, then the second half of task 2 with labels 2<->3 swapped
        model = make_model(a.model, a.seed, overrides, dim=Xtr.shape[1])
        for t in TASKS[:2]:
            feed(model, Xtr, ytr, halves[t][0])
            model.end_task()
        swap = np.array([0, 1, 3, 2, 4, 5, 6, 7, 8, 9])
        m23 = np.isin(yev, [2, 3])
        before_01 = acc_on(model, Xev, yev, [0, 1], cls4)
        curve, need = [], None
        ridx = halves[TASKS[1]][1]
        b = model.batch_size
        for s in range(0, len(ridx), b):
            j = ridx[s:s + b]
            model.learn(Xtr[j], swap[ytr[j]])
            if (s // b) % max(1, 20 // b) == 0 or s + b >= len(ridx):
                acc_rev = float((model.predict(Xev[m23], cls4) == swap[yev[m23]]).mean())
                curve.append((s + len(j), acc_rev))
                if need is None and acc_rev >= 0.8:
                    need = s + len(j)
        model.end_task()
        res['REV_curve'] = curve
        res['REV_final'] = float((model.predict(Xev[m23], cls4) == swap[yev[m23]]).mean())
        res['REV_examples_per_class_to_80'] = None if need is None else need / 2
        res['REV_other_drop'] = before_01 - acc_on(model, Xev, yev, [0, 1], cls4)
        print(f"  REV final={res['REV_final']:.3f} ex/class to 80%={res['REV_examples_per_class_to_80']} "
              f"other drop={res['REV_other_drop']:.4f}", flush=True)
        # NOISE: first halves with 20% within-task flipped labels, second halves clean; vs fully clean run
        out = {}
        for noisy in (True, False):
            model = make_model(a.model, a.seed, overrides, dim=Xtr.shape[1])
            rng = np.random.RandomState(a.seed + 300)
            for t in TASKS[:2]:
                i1, i2 = halves[t]
                yl = ytr.copy()
                if noisy:
                    flip = i1[rng.rand(len(i1)) < 0.2]
                    yl[flip] = np.where(yl[flip] == t[0], t[1], t[0])
                feed(model, Xtr, yl, i1)
                feed(model, Xtr, ytr, i2)
                model.end_task()
            out[noisy] = acc_on(model, Xev, yev, cls4, cls4)
        res['NOISE_acc'], res['NOISE_clean_acc'] = out[True], out[False]
        res['NOISE_gap'] = out[False] - out[True]
        print(f"  NOISE acc={out[True]:.4f} clean={out[False]:.4f} gap={res['NOISE_gap']:.4f}", flush=True)

    res['wall_clock_s'] = time.time() - T0
    os.makedirs(a.out, exist_ok=True)
    name = a.model + (f'_{a.tag}' if a.tag else '') + f'_{a.data}_{a.seed}_{a.split}' + (f'_cap{a.cap}' if a.cap else ('_quick' if a.quick else ''))
    path = os.path.join(a.out, name + '.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'wrote {path}  ({res["wall_clock_s"]:.0f}s)')


if __name__ == '__main__':
    main()
