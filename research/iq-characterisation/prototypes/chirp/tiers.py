"""Confidence tiers, centre error and cost per condition from harness JSONL (verdict recomputed offline)."""
import json, sys, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from chirp import analyze as an

def rows_of(path):
    out = []
    for l in open(path):
        d = json.loads(l)
        if 'error' in d: continue
        d['verdict'] = an.verdict({'chirp': d['chirp'], 'pulsed': d['pulsed']})
        out.append(d)
    return out

if __name__ == '__main__':
    pos_paths = sys.argv[1].split(',')
    neg_paths = sys.argv[2].split(',') if len(sys.argv) > 2 else []
    pos = [r for p in pos_paths for r in rows_of(p) if r['truth']['chirp']]
    neg = [r for p in neg_paths for r in rows_of(p) if not r['truth']['chirp']]
    print('POSITIVES (chirp truth) - tier by SNR: high / medium / low / none')
    tier = collections.defaultdict(collections.Counter)
    for r in pos:
        v = r['verdict']['chirp']
        tier[r['snr']][v['confidence'] if v else 'none'] += 1
    for s in sorted(tier):
        c = tier[s]; n = sum(c.values())
        print('snr %4g n=%3d  high %3d medium %3d low %3d none %3d' % (s, n, c['high'], c['medium'], c['low'], c['none']))
    if neg:
        c = collections.Counter(); 
        for r in neg:
            v = r['verdict']['chirp']; c[v['confidence'] if v else 'none'] += 1
        print('NEGATIVES tiers:', dict(c), 'n=%d' % len(neg))
    ce = collections.defaultdict(list)
    for r in pos:
        v = r['verdict']['chirp']
        if v and r['name'] != 'multi_lora_fsk':
            ce[r['snr']].append(abs(v['centre_hz'] - (r['imp']['err'] + r['imp']['cfo'])))
    print('centre-frequency error |Hz| (median, p90) by SNR:', {s: (int(np.median(a)), int(np.percentile(a, 90))) for s, a in sorted(ce.items())})
