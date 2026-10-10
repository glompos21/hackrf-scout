"""Evaluate the frozen rules (thresholds.json) on a jsonl of features.  Used on seeds 100-119 for the numbers
in the report.  Nothing here changes a threshold.

    python3 eval_rules.py results/eval.jsonl [out.json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np

import harness
import rules
import summ
import truth_class

LABELS = ['noise-only', 'carrier', 'narrow', 'medium', 'wide', 'ofdm', 'hopper', 'unknown']
SN = [0, 5, 10, 15, 20, 30]


def kind_of(case):
    return harness.CASES[case][0]


def annotate(rows, T):
    for r in rows:
        nb = r['truth'].get('occupied_bw_hz')
        r['pred'], r['conf'], r['why'] = rules.classify(r['feat'], T)
        r['true'], r['care'] = truth_class.truth_label(r['case'], nb or 0.0)
        r['kind'] = kind_of(r['case'])
    return rows


def confusion(rows):
    m = defaultdict(Counter)
    for r in rows:
        if r['care']:
            m[r['true']][r['pred']] += 1
    return m


def print_conf(m, title):
    print('\n' + title)
    print('%-11s' % 'truth\\pred' + ''.join('%11s' % l[:10] for l in LABELS) + '      n   acc')
    for t in LABELS[:-1]:
        c = m.get(t)
        if not c:
            continue
        n = sum(c.values())
        print('%-11s' % t + ''.join('%11d' % c.get(l, 0) for l in LABELS) + '%7d  %4.0f%%' % (n, 100.0 * c.get(t, 0) / n))


def acc_table(rows, title, truth_labels=None):
    print('\n' + title + ' (accuracy %, n per cell in brackets)')
    print('%-15s %-9s' % ('case', 'truth') + ''.join('%12d' % s for s in SN))
    for case in harness.CASES:
        rr = [r for r in rows if r['case'] == case]
        if not rr:
            continue
        line = '%-15s %-9s' % (case, rr[0]['true'] + ('' if rr[0]['care'] else '*'))
        for s in SN:
            xs = [r for r in rr if (r['snr'] == s or kind_of(case) == 'noise')]
            if not xs:
                line += '%12s' % '-'
                continue
            ok = np.mean([r['pred'] == r['true'] for r in xs]) * 100
            line += '%8.0f%%(%2d)' % (ok, len(xs))
        print(line)


def main(path, out=None, thr=None):
    T = rules.load_thr(thr)
    rows = annotate(summ.load(path), T)
    res = {}
    for cond in ('clean', 'imp'):
        rc = [r for r in rows if r['cond'] == cond]
        for lo in (0, 5, 10):
            sub = [r for r in rc if r['snr'] >= lo or r['kind'] == 'noise']
            m = confusion(sub)
            print_conf(m, '== confusion, cond=%s, SNR >= %d dB (noise included)  [* = don\'t-care classes excluded]' % (cond, lo))
            res['confusion_%s_ge%d' % (cond, lo)] = {t: dict(c) for t, c in m.items()}
        acc_table(rc, '== per-case accuracy by SNR, cond=%s' % cond)

    # detection probability (anything but noise-only/unknown)
    print('\n== detection probability (pred not noise-only/unknown), by case')
    for cond in ('clean', 'imp'):
        print('cond=%s' % cond)
        print('%-15s' % 'case' + ''.join('%8d' % s for s in SN))
        for case in harness.CASES:
            if kind_of(case) == 'noise':
                continue
            line = '%-15s' % case
            for s in SN:
                xs = [r for r in rows if r['case'] == case and r['cond'] == cond and r['snr'] == s]
                line += '%8.2f' % np.mean([r['pred'] not in ('noise-only', 'unknown') for r in xs]) if xs else '%8s' % '-'
            print(line)
    nz = [r for r in rows if r['kind'] == 'noise']
    fp = sum(r['pred'] != 'noise-only' for r in nz)
    print('\nnoise-only captures: %d, called something else: %d  (%s)' % (len(nz), fp, Counter(r['pred'] for r in nz)))
    res['noise_fp'] = [fp, len(nz)]

    # OFDM / hopper detail
    for name, lab in (('OFDM', 'ofdm'), ('hopper', 'hopper')):
        print('\n== %s rule: true-positive rate by SNR and false positives' % name)
        for cond in ('clean', 'imp'):
            for case in [c for c in harness.CASES if kind_of(c) == lab]:
                line = '%-12s %-5s TPR:' % (case, cond)
                for s in SN:
                    xs = [r for r in rows if r['case'] == case and r['cond'] == cond and r['snr'] == s]
                    line += ' %4.2f' % np.mean([r['pred'] == lab for r in xs])
                print(line)
        fps = Counter((r['case'], r['cond']) for r in rows if r['pred'] == lab and r['true'] != lab)
        n_other = Counter((r['case'], r['cond']) for r in rows if r['true'] != lab)
        print('false positives (case,cond): %s' % (', '.join('%s/%s %d/%d' % (k[0], k[1], v, n_other[k]) for k, v in sorted(fps.items())) or 'none'))
        res['fp_' + lab] = {'%s|%s' % k: [v, n_other[k]] for k, v in fps.items()}
        res['fp_%s_total' % lab] = [int(sum(fps.values())), int(sum(n_other.values()))]

    # dwell estimate
    dw = [(r['snr'], r['feat'].get('hop_dwell_s')) for r in rows if r['kind'] == 'hopper' and r['cond'] == 'clean'
          and r['feat'].get('hop_dwell_kind') == 'period']
    print('\nhopper dwell (truth 625 us), clean: ')
    for s in SN:
        v = np.array([d for ss_, d in dw if ss_ == s and d])
        if len(v):
            print('  snr %2d: median %.1f us  p10..p90 %.1f..%.1f us (n=%d)' % (s, np.median(v) * 1e6, np.percentile(v, 10) * 1e6, np.percentile(v, 90) * 1e6, len(v)))

    # confidence reliability
    c = np.array([r['conf'] for r in rows if r['care']])
    ok = np.array([r['pred'] == r['true'] for r in rows if r['care']])
    print('\n== confidence reliability (care classes)')
    for a, b in ((0, 0.34), (0.34, 0.67), (0.67, 1.01)):
        m = (c >= a) & (c < b)
        if m.any():
            print('  conf in [%.2f,%.2f): n=%d accuracy=%.0f%%' % (a, b, m.sum(), 100 * ok[m].mean()))
    res['conf_rel'] = [[float(a), int(((c >= a) & (c < b)).sum()), float(ok[(c >= a) & (c < b)].mean())] for a, b in ((0, 0.34), (0.34, 0.67), (0.67, 1.01)) if ((c >= a) & (c < b)).any()]
    if out:
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
