"""Empirical confidence: P(correct | answer family, SNR bin), fitted on the tuning seeds with Laplace smoothing.

conf_label = P(label right)           for an answered family label
conf_baud  = P(label right and baud within 5 % of truth)   for an answered baud
The tables are lookup tables written to conf_table.json; they are only as representative as the synthetic bench."""
import json, math, os
import numpy as np
import classify as C
import evaluate as E

EDGES = [-1e9, 3.0, 8.0, 14.0, 1e9]
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf_table.json')


def snr_on(f):
    s = f.get('snr_psd_db')
    if s is None:
        return None
    duty = max(min(f.get('blk_duty', 1.0), 1.0), 0.02)
    return s - 10 * math.log10(duty)


def bin_of(s):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= s < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def fit(recs, classifier=None):
    cl = classifier or C.classify
    cnt = {}
    for d in recs:
        o = cl(d['feat'])
        if o['label'] not in E.FAMS:
            continue
        sc = E.score(d, o)
        key = '%s|%d' % (o['label'], bin_of(snr_on(d['feat'])))
        c = cnt.setdefault(key, dict(n=0, ok=0, nb=0, okb=0))
        c['n'] += 1
        c['ok'] += int(sc['ok'])
        if o['baud_hz'] is not None:
            c['nb'] += 1
            c['okb'] += int(sc['ok'] and sc.get('hit5', False))
    tab = {}
    for k, c in cnt.items():
        tab[k] = dict(n=c['n'], conf_label=(c['ok'] + 1.0) / (c['n'] + 2.0), nb=c['nb'],
                      conf_baud=(c['okb'] + 1.0) / (c['nb'] + 2.0) if c['nb'] else None)
    return tab


def save(tab):
    json.dump(tab, open(_path, 'w'), indent=1)


def load():
    return json.load(open(_path)) if os.path.exists(_path) else {}


def apply(out, f, tab=None):
    tab = load() if tab is None else tab
    if out['label'] in E.FAMS:
        key = '%s|%d' % (out['label'], bin_of(snr_on(f)))
        c = tab.get(key)
        if c is None:
            # nearest populated bin of the same family
            cands = [(abs(int(k.split('|')[1]) - bin_of(snr_on(f))), k) for k in tab if k.startswith(out['label'] + '|')]
            c = tab[min(cands)[1]] if cands else None
        if c:
            out['conf_label'] = c['conf_label']
            out['conf_baud'] = c['conf_baud'] if out['baud_hz'] is not None else None
    return out
