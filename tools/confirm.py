"""Mean ± sd over val seeds 0–2 for a tag vs the It 12 baseline: python tools/confirm.py <tag> [<tag> ...]"""
import json, sys
import numpy as np

def get(tag, d, s):
    if tag == 'base':
        f = f'results/sparc_it12_r10_c5_{d}_0_val.json' if s == 0 else f'results/sparc_r2_3c_base_{d}_{s}_val.json'
    else:
        f = f'results/sparc_{tag}_{d}_{s}_val.json'
    return json.load(open(f))

for d in ('mnist', 'fashion'):
    for tag in ['base'] + sys.argv[1:]:
        D = [get(tag, d, s) for s in (0, 1, 2)]
        f = lambda k: f"{100*np.mean([x[k] for x in D]):.1f}±{100*np.std([x[k] for x in D]):.1f}"
        print(f"{d:8s} {tag:12s} ACC {f('ACC')}  FGT {f('FGT')}  FEW5 {f('FEW5')}  drop {f('FEW5_old_drop')}  NOV {np.mean([x['NOVELTY'] for x in D]):.3f}")
