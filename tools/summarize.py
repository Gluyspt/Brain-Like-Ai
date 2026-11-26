"""Print a compact table of results JSONs: python summarize.py results/*.json"""
import json, sys
keys = ['ACC', 'FGT', 'FEW1', 'FEW1_old_drop', 'FEW5', 'FEW5_old_drop', 'LBL5', 'NOVELTY',
        'REV_final', 'REV_examples_per_class_to_80', 'REV_other_drop', 'NOISE_gap']
print('run | ' + ' | '.join(keys) + ' | sparsity | wall')
for f in sys.argv[1:]:
    d = json.load(open(f))
    cells = []
    for k in keys:
        v = d.get(k)
        cells.append('-' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v)))
    sp = d.get('SPARSITY', {})
    print(f.split('/')[-1][:-5] + ' | ' + ' | '.join(cells) + ' | ' +
          ','.join(f'{k}:{v:.3f}' for k, v in sp.items()) + f" | {d.get('wall_clock_s', 0):.0f}")
