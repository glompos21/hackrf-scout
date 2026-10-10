import sys, json
sys.path.insert(0, '.')
import numpy as np, rules, truth_class, harness
from collections import Counter, defaultdict
T = rules.load_thr()
rows = [json.loads(l) for l in open('results/stress.jsonl')]
rows = [r for r in rows if 'error' not in r]
print('records', len(rows))
LAB = ['noise-only', 'carrier', 'narrow', 'medium', 'wide', 'ofdm', 'hopper', 'unknown']
for v in ['ideal', 'rolloff_known', 'rolloff_blind', 'rolloff_narrow', 'rolloff+iq+dcw', 'all+dc6']:
    print('\n=== variant %s ===' % v)
    print('%-14s %3s  ' % ('case', 'snr') + ' '.join('%10s' % l[:10] for l in LAB) + '   truth')
    for case in harness.CASES:
        for snr in ([0] if harness.CASES[case][0] == 'noise' else [5, 15]):
            rr = [r for r in rows if r['variant'] == v and r['case'] == case and r['snr'] == snr]
            if not rr: continue
            cnt = Counter(rules.classify(r['feat'], T)[0] for r in rr)
            lab, care = truth_class.truth_label(case, rr[0]['truth']['occupied_bw_hz'] or 0)
            print('%-14s %3d  ' % (case, snr) + ' '.join('%10d' % cnt.get(l, 0) for l in LAB) + '   %s%s  (n=%d, floor_ok %d)' % (lab, '' if care else '*', len(rr), sum(bool(r['feat'].get('floor_ok')) for r in rr)))
