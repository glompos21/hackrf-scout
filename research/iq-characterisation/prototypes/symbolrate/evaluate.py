"""Scoring of classify() output against generator truth."""
from __future__ import annotations

import collections
import json
import math
import sys

import numpy as np

import classify as C

FAMS = ['carrier', 'am', 'fm', 'ook', 'fsk', 'psk']
HAS_RATE = {'ook', 'fsk2', 'gfsk', 'bpsk', 'qpsk'}      # generator kinds with truth['symbol_rate'] meaningful for us


def load(path, snrs=None, conds=None):
    recs = []
    for line in open(path):
        d = json.loads(line)
        if 'error' in d:
            continue
        if snrs is not None and d['snr'] not in snrs:
            continue
        if conds is not None and d['cond'] not in conds:
            continue
        recs.append(d)
    return recs


def truth_family(d):
    """(family, symbol_rate, note) the classifier should answer with.  For 'multi' captures the answer
    is judged against the component whose band the analysis actually looked at."""
    fam = d['truth']['family']
    if fam != 'multi':
        sr = d['truth']['symbol_rate'] if d['truth']['kind'] in HAS_RATE else None
        return fam, sr, None
    f = d['feat']
    if f.get('stage') != 'ok':
        return 'multi', None, None
    cen = f['band']['centroid'] + 0.0
    best = None
    for c in d.get('comps', []):
        # truth band centre is relative to the capture centre: center_hz
        dist = abs(c['center_hz'] - cen)
        if best is None or dist < best[0]:
            best = (dist, c)
    if best is None:
        return 'multi', None, None
    c = best[1]
    fam = {'cw': 'carrier', 'nfm': 'fm', 'wfm': 'fm', 'ook': 'ook', 'fsk2': 'fsk', 'am': 'am'}.get(c['kind'], 'other')
    sr = c['symbol_rate'] if c['kind'] in HAS_RATE else None
    return fam, sr, 'multi:%s' % c['kind']


def score(d, out):
    """Return a dict of booleans / numbers for one capture."""
    fam, sr, note = truth_family(d)
    lab = out['label']
    r = dict(truth=fam, label=lab)
    # label correctness
    if fam == 'none':
        r['ok'] = lab in ('none', 'unknown')
        r['false_alarm'] = lab in FAMS
        r['refused'] = lab in ('none', 'unknown')
        r['wrong'] = lab in FAMS
    elif fam in FAMS:
        r['ok'] = lab == fam
        r['refused'] = lab in ('none', 'unknown')
        r['wrong'] = (lab in FAMS) and lab != fam
        r['missed'] = lab == 'none'
    else:  # other / multi-other: anything but a refusal is wrong
        r['ok'] = lab in ('none', 'unknown')
        r['refused'] = lab in ('none', 'unknown')
        r['wrong'] = lab in FAMS
    # baud
    b = out.get('baud_hz')
    r['has_rate_truth'] = sr is not None and fam in FAMS
    r['baud_answered'] = b is not None
    if sr is not None and b is not None:
        e = abs(b - sr) / sr
        r['err'] = e
        r['hit1'] = e <= 0.01
        r['hit5'] = e <= 0.05
        ratio = b / sr
        r['harm'] = any(abs(ratio - k) / k <= 0.06 for k in (0.25, 1 / 3., 0.5, 2.0, 3.0, 4.0)) and e > 0.05
        bnd = out.get('bound_rel')
        r['in_bound'] = (e <= bnd) if bnd is not None else None
    r['false_baud'] = (sr is None) and (b is not None)
    return r


def summarize(recs, classifier=None, by=('variant', 'snr', 'cond'), verbose=True):
    classifier = classifier or C.classify
    rows = collections.OrderedDict()
    for d in recs:
        out = classifier(d['feat'])
        sc = score(d, out)
        key = tuple(d[k] for k in by)
        rows.setdefault(key, []).append((d, out, sc))
    return rows


def fmt_rows(rows):
    lines = []
    for key, items in rows.items():
        n = len(items)
        labs = collections.Counter(o['label'] for _, o, _ in items)
        ok = sum(1 for _, _, s in items if s['ok'])
        wrong = sum(1 for _, _, s in items if s.get('wrong'))
        refused = sum(1 for _, _, s in items if s.get('refused'))
        rate_items = [s for _, _, s in items if s['has_rate_truth']]
        ba = [s for s in rate_items if s['baud_answered']]
        txt = '%-22s n=%2d ok=%2d wrong=%2d refused=%2d |' % ('/'.join(str(k) for k in key), n, ok, wrong, refused)
        txt += ' ' + ','.join('%s:%d' % (k, v) for k, v in sorted(labs.items()))
        if rate_items:
            h1 = sum(1 for s in ba if s['hit1'])
            h5 = sum(1 for s in ba if s['hit5'])
            hm = sum(1 for s in ba if s['harm'])
            txt += ' | baud answered %d/%d, <=1%%: %d, <=5%%: %d, harmonic: %d' % (len(ba), len(rate_items), h1, h5, hm)
        fb = sum(1 for _, _, s in items if s['false_baud'])
        if fb:
            txt += ' | FALSE BAUD %d' % fb
        lines.append(txt)
    return lines


if __name__ == '__main__':
    path = sys.argv[1]
    conds = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    variants = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    recs = load(path, conds=conds)
    if variants:
        recs = [d for d in recs if d['variant'] in variants]
    rows = summarize(recs)
    for l in fmt_rows(rows):
        print(l)
