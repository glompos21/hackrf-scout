"""Tables for the report from a JSONL feature file (tuning or evaluation seeds).
usage: report_eval.py file.jsonl [--cond cont|gated|all] [--section label,baud,confusion,noise,lowest,calib,bound]"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys

import numpy as np

import classify as C
import conf as CF
import evaluate as E

SNRS = [0, 5, 10, 15, 20, 30]


def classify_all(recs, conf_tab=None):
    items = []
    for d in recs:
        o = C.classify(d['feat'])
        if conf_tab is not None:
            CF.apply(o, d['feat'], conf_tab)
        items.append((d, o, E.score(d, o)))
    return items


def pct(a, b):
    return '  -' if b == 0 else '%3.0f' % (100.0 * a / b)


def table_label(items, variants=None):
    by = collections.defaultdict(list)
    for d, o, s in items:
        by[(d['variant'], d['cond'], d['snr'])].append((o, s))
    print('== label table: % correct / %wrong / %refused (n per cell in header) ==')
    print('%-16s %-5s' % ('variant', 'cond') + ''.join('   %4s dB      ' % s for s in SNRS))
    seen = []
    for (v, c, s) in by:
        if (v, c) not in seen:
            seen.append((v, c))
    for v, c in seen:
        if variants and v not in variants:
            continue
        row = '%-16s %-5s' % (v, c)
        for snr in SNRS:
            cell = by.get((v, c, float(snr))) or by.get((v, c, snr)) or []
            n = len(cell)
            if n == 0:
                row += '       -        '
                continue
            ok = sum(1 for _, s in cell if s['ok'])
            wr = sum(1 for _, s in cell if s.get('wrong'))
            rf = sum(1 for _, s in cell if s.get('refused'))
            row += '  %s/%s/%s (%d)' % (pct(ok, n), pct(wr, n), pct(rf, n), n)
        print(row)


def table_baud(items):
    by = collections.defaultdict(list)
    for d, o, s in items:
        if s['has_rate_truth'] and s['truth'] == s['label'] or (s['has_rate_truth']):
            by[(d['variant'], d['cond'], d['snr'])].append((o, s))
    print('== baud table: answered / within1% / within5% / in-bound / harmonic-alias (of all captures in cell); median |err| of answered ==')
    seen = []
    for k in by:
        if (k[0], k[1]) not in seen:
            seen.append((k[0], k[1]))
    for v, c in seen:
        row = '%-12s %-5s' % (v, c)
        for snr in SNRS:
            cell = by.get((v, c, float(snr))) or by.get((v, c, snr)) or []
            n = len(cell)
            if n == 0:
                row += '          -          '
                continue
            ans = [(o, s) for o, s in cell if s['baud_answered']]
            h1 = sum(1 for o, s in ans if s['hit1'])
            h5 = sum(1 for o, s in ans if s['hit5'])
            ib = sum(1 for o, s in ans if s.get('in_bound'))
            hm = sum(1 for o, s in ans if s['harm'])
            med = np.median([s['err'] for o, s in ans]) if ans else float('nan')
            row += ' %s/%s/%s/%s/%s %.0e' % (pct(len(ans), n), pct(h1, n), pct(h5, n), pct(ib, len(ans)), pct(hm, n), med if med == med else 0)
        print(row)


def confusion(items, snr_min=0, only_cond=None):
    labs = ['none', 'unknown', 'carrier', 'am', 'fm', 'ook', 'fsk', 'psk']
    by = collections.defaultdict(collections.Counter)
    for d, o, s in items:
        if d['snr'] < snr_min:
            continue
        by[d['variant']][o['label']] += 1
    print('== confusion (rows: variant, columns: output) for SNR >= %s dB ==' % snr_min)
    print('%-16s' % 'variant' + ''.join('%8s' % l for l in labs))
    for v, c in by.items():
        print('%-16s' % v + ''.join('%8d' % c.get(l, 0) for l in labs))


def noise_fpr(items):
    tot = collections.Counter()
    for d, o, s in items:
        if d['variant'] == 'noise':
            tot[o['label']] += 1
    print('== noise-only captures: output counts ==', dict(tot))


def lowest_snr(items, thresh=0.9):
    """per variant/cond: the lowest SNR such that label-correct rate >= thresh at that and all higher SNRs;
    same for baud within 5 %."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for d, o, s in items:
        by[(d['variant'], d['cond'])][d['snr']].append(s)
    print('== lowest grid SNR with label-correct >= %.0f%% (and baud<=5%% >= %.0f%% of all captures) at it and above ==' % (100 * thresh, 100 * thresh))
    for (v, c), sn in by.items():
        if v == 'noise':
            continue
        snrs = sorted(sn)
        low_l = low_b = None
        for s in reversed(snrs):
            cell = sn[s]
            okl = np.mean([x['ok'] for x in cell])
            if okl >= thresh:
                low_l = s
            else:
                break
        has_rate = any(x['has_rate_truth'] for cell in sn.values() for x in cell)
        if has_rate:
            for s in reversed(snrs):
                cell = sn[s]
                okb = np.mean([bool(x['baud_answered'] and x.get('hit5')) for x in cell])
                if okb >= thresh:
                    low_b = s
                else:
                    break
        print('%-16s %-5s label: %s dB   baud: %s' % (v, c, low_l, low_b if has_rate else 'n/a'))


def calibration(items):
    rows = []
    for d, o, s in items:
        if o['label'] in E.FAMS and o.get('conf_label') is not None:
            rows.append((o['conf_label'], s['ok'], o.get('conf_baud'), bool(s.get('ok') and s.get('hit5')) if o['baud_hz'] is not None else None))
    print('== calibration of conf_label (n=%d answers) ==' % len(rows))
    edges = [0.0, 0.5, 0.7, 0.85, 0.95, 0.99, 1.01]
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if a <= r[0] < b]
        if sel:
            print('  conf in [%.2f,%.2f): n=%4d mean predicted %.3f  observed %.3f' % (a, b, len(sel), np.mean([r[0] for r in sel]), np.mean([r[1] for r in sel])))
    rb = [r for r in rows if r[2] is not None]
    print('== calibration of conf_baud (n=%d answers with baud) ==' % len(rb))
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in rb if a <= r[2] < b]
        if sel:
            print('  conf in [%.2f,%.2f): n=%4d mean predicted %.3f  observed %.3f' % (a, b, len(sel), np.mean([r[2] for r in sel]), np.mean([r[3] for r in sel])))


def bound_cov(items):
    print('== error-bound coverage (fraction of answered baud with |err| <= bound_rel) and size ==')
    by = collections.defaultdict(list)
    for d, o, s in items:
        if o['baud_hz'] is not None and s.get('err') is not None and s['truth'] == o['label']:
            by[o['label']].append((s['err'], o['bound_rel']))
    for k, v in by.items():
        v = np.array(v)
        print('  %-4s n=%4d coverage %.3f  median bound %.2e  p90 bound %.2e  median err %.2e  p99 err %.2e' %
              (k, len(v), (v[:, 0] <= v[:, 1]).mean(), np.median(v[:, 1]), np.percentile(v[:, 1], 90), np.median(v[:, 0]), np.percentile(v[:, 0], 99)))


GROUPS = collections.OrderedDict([
    ('carrier', ['cw']), ('am', ['am_tone', 'am_voice']), ('fm (voice+broadcast)', ['nfm_voice', 'wfm']), ('fm (tone, corner)', ['nfm_tone']),
    ('ook', ['ook_pwm', 'ook_man', 'ook_nrz']), ('fsk (fsk2)', ['fsk2_rect', 'fsk2_gauss', 'fsk2_h1', 'fsk2_h2']), ('gfsk 1 Mbaud', ['gfsk']),
    ('psk (bpsk/qpsk)', ['bpsk', 'qpsk', 'qpsk_50k']), ('other: lora/ofdm/hopper/pulsed', ['lora', 'lora_sf9', 'ofdm', 'ofdm_nb', 'hopper', 'pulsed']),
    ('multi (2 signals)', ['multi_cw_nfm', 'multi_ook_fsk']), ('noise', ['noise'])])


def table_family(items):
    print('== by group and SNR: % correct / % wrong / % refused (n) ; cont and gated pooled separately ==')
    for cond in ('cont', 'gated'):
        print('-- %s --' % cond)
        print('%-34s' % 'group' + ''.join('   %4s dB       ' % s for s in SNRS))
        for g, vs in GROUPS.items():
            row = '%-34s' % g
            any_ = False
            for snr in SNRS:
                cell = [(o, s) for d, o, s in items if d['variant'] in vs and d['cond'] == cond and d['snr'] == snr]
                if g == 'noise' and cond == 'cont' and snr == 0:
                    cell = [(o, s) for d, o, s in items if d['variant'] in vs]
                n = len(cell)
                if not n:
                    row += '       -        '
                    continue
                any_ = True
                ok = sum(1 for _, s in cell if s['ok']); wr = sum(1 for _, s in cell if s.get('wrong')); rf = sum(1 for _, s in cell if s.get('refused'))
                row += '  %s/%s/%s(%3d)' % (pct(ok, n), pct(wr, n), pct(rf, n), n)
            if any_:
                print(row)


def table_group_baud(items):
    print('== baud by group and SNR: of ALL captures in the cell -> % with a baud answered / % within 1 % / % within 5 % / % wrong (answered but >5 % off) ==')
    for cond in ('cont', 'gated'):
        print('-- %s --' % cond)
        for g, vs in GROUPS.items():
            if g not in ('ook', 'fsk (fsk2)', 'gfsk 1 Mbaud', 'psk (bpsk/qpsk)'):
                continue
            row = '%-26s' % g
            for snr in SNRS:
                cell = [(o, s) for d, o, s in items if d['variant'] in vs and d['cond'] == cond and d['snr'] == snr]
                n = len(cell)
                if not n:
                    row += '        -        '
                    continue
                ans = sum(1 for _, s in cell if s['baud_answered']); h1 = sum(1 for _, s in cell if s['baud_answered'] and s.get('hit1'))
                h5 = sum(1 for _, s in cell if s['baud_answered'] and s.get('hit5')); bad = ans - h5
                row += '  %s/%s/%s/%s' % (pct(ans, n), pct(h1, n), pct(h5, n), pct(bad, n))
            print(row)
        if cond == 'cont':
            print('   columns = SNR', SNRS)


def false_baud(items):
    fb = [(d, o) for d, o, s in items if s['false_baud']]
    print('== baud answered for a capture that has no symbol rate (false baud): %d of %d captures ==' % (len(fb), len(items)))
    c = collections.Counter((d['variant'], d['snr']) for d, o in fb)
    print('   ', dict(c))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--cond', default='all')
    ap.add_argument('--sections', default='label,baud,confusion,noise,lowest,bound')
    ap.add_argument('--conf', default=None)
    ap.add_argument('--variants', default=None)
    ap.add_argument('--snr_min', type=float, default=0)
    a = ap.parse_args()
    recs = E.load(a.path, conds=None if a.cond == 'all' else a.cond.split(','))
    if a.variants:
        vs = a.variants.split(',')
        recs = [d for d in recs if d['variant'] in vs]
    else:
        recs = [d for d in recs if not d['variant'].startswith('x_')]
    tab = CF.load() if a.conf else None
    items = classify_all(recs, tab)
    for sec in a.sections.split(','):
        {'label': table_label, 'baud': table_baud, 'confusion': lambda it: confusion(it, a.snr_min), 'noise': noise_fpr,
         'lowest': lowest_snr, 'calib': calibration, 'bound': bound_cov, 'family': table_family, 'gbaud': table_group_baud, 'falsebaud': false_baud}[sec](items)
        print()
