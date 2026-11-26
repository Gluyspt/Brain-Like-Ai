"""Build the results tables for ARCHITECTURE.md directly from results JSONs (no hand-copied numbers).
python make_report.py > results/report_tables.md
"""
import glob, json
import numpy as np


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def ms(D, k, scale=100, fmt='{:.1f}'):
    v = [d[k] for d in D if d.get(k) is not None]
    if not v:
        return '–'
    m = np.mean(v) * scale
    if len(v) == 1:
        return fmt.format(m)
    return (fmt + ' ± ' + fmt).format(m, np.std(v) * scale)


def main_table(tag):
    print(f'### Final results (test split, mean ± sd over seeds; tag `{tag}` for SPARC, `final` for baselines)\n')
    print('| model | data | n | ACC | FGT | FEW1 | FEW5 | FEW5 old drop | LBL5 | NOVELTY | REV ex/class→80 % | REV final | REV other drop | NOISE gap |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in ('mnist', 'fashion'):
        for m, t in (('sparc', tag), ('b1', 'final'), ('b2', 'final'), ('b3', 'final'), ('b4', 'final')):
            D = load(f'results/{m}_{t}_{d}_*_test.json')
            if not D:
                continue
            print(f"| {m.upper() if m != 'sparc' else 'SPARC'} | {d} | {len(D)} | {ms(D,'ACC')} | {ms(D,'FGT')} | "
                  f"{ms(D,'FEW1')} | {ms(D,'FEW5')} | {ms(D,'FEW5_old_drop')} | {ms(D,'LBL5')} | "
                  f"{ms(D,'NOVELTY',1,'{:.3f}')} | {ms(D,'REV_examples_per_class_to_80',1)} | {ms(D,'REV_final')} | "
                  f"{ms(D,'REV_other_drop')} | {ms(D,'NOISE_gap')} |")
    D = load(f'results/sparc_{tag}_mnist_*_test.json')
    if D:
        st = D[0]['stats']
        print(f"\nSPARC sparsity (MNIST seed 0): {D[0]['SPARSITY']}; Fashion: "
              f"{load(f'results/sparc_{tag}_fashion_*_test.json')[0]['SPARSITY']}. "
              f"params {st['params']:,}; stored numbers {st['stored_numbers']:,} ({st['hippocampus_budget']}); "
              f"raw images stored {st['raw_images_stored']}. Wall-clock per full bench run ≈ "
              f"{np.mean([x['wall_clock_s'] for x in D]):.0f} s.\n")


def ablation_table():
    keys = ['ACC', 'FGT', 'FEW5', 'FEW5_old_drop', 'LBL5', 'NOVELTY', 'REV_examples_per_class_to_80', 'NOISE_gap']
    print('### Ablations (validation split, seed 0; Δ vs full model, sign so that + = full model better)\n')
    print('| ablation | data | ' + ' | '.join(keys) + ' | costs ≥ 1 pt somewhere? |')
    print('|---|---|' + '---|' * (len(keys) + 1))
    better_high = {'ACC': 1, 'FGT': -1, 'FEW5': 1, 'FEW5_old_drop': -1, 'LBL5': 1, 'NOVELTY': 1,
                   'REV_examples_per_class_to_80': -1, 'NOISE_gap': -1}
    tags = sorted({f.split('_abl_')[1].split('_')[0] for f in glob.glob('results/sparc_abl_*_val.json')})
    for d in ('mnist', 'fashion'):
        full = load(f'results/sparc_abl_full_{d}_0_val.json')
        if not full:
            continue
        full = full[0]
        cells = [f"{full[k]:.3f}" if k == 'NOVELTY' else (f"{full[k]:.1f}" if k.startswith('REV') else f"{100*full[k]:.1f}")
                 for k in keys]
        print(f'| **full** | {d} | ' + ' | '.join(cells) + ' | |')
        for t in tags:
            if t == 'full':
                continue
            D = load(f'results/sparc_abl_{t}_{d}_0_val.json')
            if not D:
                continue
            a = D[0]
            cells, earns = [], False
            for k in keys:
                if a.get(k) is None or full.get(k) is None:
                    cells.append('–')
                    continue
                if k.startswith('REV'):
                    delta = a[k] - full[k]
                    cells.append(f'{a[k]:.1f} ({delta:+.0f})')
                    earns |= delta >= 10
                    continue
                sc = 1 if k == 'NOVELTY' else 100
                delta = (full[k] - a[k]) * sc * better_high[k]
                cells.append(f'{a[k]*sc:.{3 if k=="NOVELTY" else 1}f} ({delta:+.{3 if k=="NOVELTY" else 1}f})')
                earns |= delta >= (0.01 if k == 'NOVELTY' else 1.0)
            print(f'| {t} | {d} | ' + ' | '.join(cells) + f" | {'yes' if earns else 'no'} |")
    print()


if __name__ == '__main__':
    main_table('final4')
    ablation_table()
