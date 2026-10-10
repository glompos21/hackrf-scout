"""Stand-alone accuracy of each candidate baud estimator (no classifier gating), by variant and SNR.
usage: feature_study.py file.jsonl [conds]"""
import sys, json, collections
import numpy as np
import evaluate as E

def cands(f):
    c = {}
    if f.get('stage') != 'ok': return c
    el = f.get('env_lines') or []
    if el: c['env_top_line'] = el[0]['f']
    if f.get('env_fund'): c['env_fundamental'] = f['env_fund']['f0']
    dm = f.get('dm_lines') or []
    if dm: c['dm_top_line'] = max(dm, key=lambda x: x['ratio'])['f']
    dmf = f.get('dm_fund') or []
    if dmf: c['dm_fundamental'] = max(dmf, key=lambda x: x['z'])['f0']
    o = f.get('ook') or {}
    if 'T' in o: c['ook_runs'] = 1.0 / o['T']
    if 'T' in o and o.get('support', 0) >= 0.6: c['ook_runs_sup>=0.6'] = 1.0 / o['T']
    fr = f.get('fsk_runs') or {}
    if 'T' in fr: c['fsk_runs'] = 1.0 / fr['T']
    ac = (o.get('acf') or {})
    if ac.get('first_min_s'): c['ook_acf_first_min'] = 1.0 / ac['first_min_s']
    if ac.get('width_s'): c['ook_acf_width'] = 1.0 / ac['width_s']
    return c

if __name__ == '__main__':
    path = sys.argv[1]; conds = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    recs = E.load(path, conds=conds)
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in recs:
        sr = d['truth']['symbol_rate'] if d['truth']['kind'] in E.HAS_RATE else None
        if not sr: continue
        for name, val in cands(d['feat']).items():
            r = val / sr
            tab[(name, d['truth']['family'])][d['snr']].append(r)
    snrs = sorted({d['snr'] for d in recs})
    for (name, fam), by in sorted(tab.items()):
        line = '%-20s %-4s' % (name, fam)
        for s in snrs:
            v = np.array(by.get(s, []))
            if len(v) == 0: line += '   -   '; continue
            ok5 = (np.abs(v - 1) <= 0.05).mean(); harm = np.mean([any(abs(x - k) / k <= 0.06 for k in (0.25, 1/3., 0.5, 2, 3, 4)) for x in v])
            line += ' %3.0f/%-3.0f' % (100 * ok5, 100 * harm)
        print(line)
    print('columns = SNR', snrs, ' entries: %within5% / %harmonic-alias')
