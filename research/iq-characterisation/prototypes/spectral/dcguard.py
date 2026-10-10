"""Mitigation experiment for the DC pedestal (wandering DC): widen the DC guard.  Seeds 100-109, frozen thresholds."""
import sys, json
sys.path.insert(0, '.')
import numpy as np, harness, rules
from collections import Counter
from multiprocessing import Pool
T = rules.load_thr()
CASES = ['noise2', 'cw', 'nfm-voice', 'qpsk']
GUARDS = [('2 bins', None), ('2 kHz', 2e3), ('5 kHz', 5e3), ('10 kHz', 10e3), ('25 kHz', 25e3)]
FE = dict(dc_wander_lsb=0.5, iq_gain_db=0.5, iq_phase_deg=2.0, filter_hz=0.875 * 2e6)

def one(a):
    case, snr, seed, gname, ghz = a
    kind, rate, secs, kw = harness.CASES[case]
    df = rate / 32768.0
    nb = 2 if ghz is None else int(round(ghz / df))
    rec = harness.run_capture(case, snr, seed, 'clean', extra=dict(dc=6.0), usable_frac=0.66, frontend=FE, dc_guard_bins=nb, guard_obw=ghz is not None)
    lab = rules.classify(rec['feat'], T)[0]
    return dict(case=case, snr=snr, seed=seed, guard=gname, label=lab, obw=rec['feat'].get('obw99_hz'), struct_z=rec['feat'].get('struct_z'))

if __name__ == '__main__':
    tasks = [(c, s, sd, gn, gh) for c in CASES for s in ([0] if c.startswith('noise') else [5, 15]) for sd in range(100, 110) for gn, gh in GUARDS]
    with Pool(4) as p:
        res = p.map(one, tasks, chunksize=4)
    json.dump(res, open('results/dcguard.json', 'w'))
    for c in CASES:
        for s in ([0] if c.startswith('noise') else [5, 15]):
            line = '%-10s snr %2d ' % (c, s)
            for gn, gh in GUARDS:
                rr = [r for r in res if r['case'] == c and r['snr'] == s and r['guard'] == gn]
                cnt = Counter(r['label'] for r in rr)
                line += ' | %s: %s' % (gn, ','.join('%s %d' % (k[:6], v) for k, v in cnt.most_common(3)))
            print(line)
