"""Summarise harness JSONL output.   usage: python3 summarize.py FILE.jsonl [what] [mode]
what in: chirp pulse fp hints all
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 2)[0])


def load(path, mode=None):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if 'error' in d:
            continue
        if mode and d['mode'] != mode:
            continue
        rows.append(d)
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def table(rows, key, cols, fmt='%5.0f%%'):
    names = sorted({r['name'] for r in rows})
    print('%-22s' % '' + ''.join('%8s' % c for c in cols))
    for n in names:
        line = '%-22s' % n
        for c in cols:
            sel = [r for r in rows if r['name'] == n and r['snr'] == c]
            if not sel:
                line += '%8s' % '-'
                continue
            k = sum(1 for r in sel if key(r))
            line += ('%4d/%-3d' % (k, len(sel)))
        print(line)


def chirp_flag(r):
    return bool(r['chirp'].get('flag'))


def chirp_flag_confirmed(r):
    c = r['chirp']
    if not c.get('flag'):
        return False
    dc = c.get('dechirp')
    return bool(dc and dc['n_hits'] >= 3)


def pulse_flag(r):
    return bool(r['pulsed'].get('flag'))


def main():
    path = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else 'all'
    mode = sys.argv[3] if len(sys.argv) > 3 else None
    rows = load(path, mode)
    snrs = sorted({r['snr'] for r in rows})
    print('%d rows  snrs %s' % (len(rows), snrs))
    if what in ('chirp', 'all'):
        pos = [r for r in rows if r['truth']['chirp']]
        neg = [r for r in rows if not r['truth']['chirp']]
        print('\n== chirp flag, POSITIVES (detected / n)')
        table(pos, chirp_flag, snrs)
        print('\n== chirp flag, NEGATIVES (false positives / n)')
        table(neg, chirp_flag, snrs)
        print('\n== chirp flag + dechirp confirmation (>=3 windows), POSITIVES')
        table(pos, chirp_flag_confirmed, snrs)
        print('\n== chirp flag + dechirp confirmation, NEGATIVES')
        table(neg, chirp_flag_confirmed, snrs)
        n = len(neg)
        k = sum(chirp_flag(r) for r in neg)
        print('\nnegatives overall: %d/%d  95%% Wilson CI [%.4f, %.4f]' % (k, n, *wilson(k, n)))
    if what in ('hints', 'all'):
        pos = [r for r in rows if r['truth']['chirp'] and chirp_flag(r)]
        print('\n== hints among detected positives')
        by = defaultdict(list)
        for r in pos:
            b = r['chirp']['best']
            ok_sf = r['truth']['sf'] == b['sf_final']
            ok_bw = abs((r['truth']['bw'] or 0) - b['bw_final']) < 1
            by[r['name']].append((ok_sf, ok_bw, ok_sf and ok_bw))
        for n, v in sorted(by.items()):
            v = np.array(v)
            print('%-22s n=%4d  SF ok %5.1f%%  BW ok %5.1f%%  both %5.1f%%' % (n, len(v), *(100 * v.mean(0))))
    if what in ('pulse', 'all'):
        pos = [r for r in rows if r['truth']['pulse']]
        neg = [r for r in rows if not r['truth']['pulse']]
        print('\n== pulsed flag, POSITIVES')
        table(pos, pulse_flag, snrs)
        print('\n== pulsed flag, NEGATIVES')
        table(neg, pulse_flag, snrs)
        n = len(neg)
        k = sum(pulse_flag(r) for r in neg)
        print('\nnegatives overall: %d/%d  95%% Wilson CI [%.4f, %.4f]' % (k, n, *wilson(k, n)))
        print('\n== width / PRI error among flagged positives (median |rel err|, 90th pct)')
        by = defaultdict(list)
        for r in pos:
            if pulse_flag(r):
                p = r['pulsed']
                ew = abs(p['width_s'] - r['truth']['pw3db_s']) / r['truth']['pw3db_s'] if p.get('width_s') else np.nan
                ep = abs(p['info']['pri'] - r['truth']['pri_s']) / r['truth']['pri_s']
                by[(r['name'], r['snr'])].append((ew, ep, p.get('completeness', np.nan)))
        for k, v in sorted(by.items()):
            v = np.array(v, float)
            print('%-22s snr %3g n=%3d  width err med %5.1f%% p90 %5.1f%% | PRI err med %6.3f%% | completeness med %.2f' % (
                k[0], k[1], len(v), 100 * np.nanmedian(v[:, 0]), 100 * np.nanpercentile(v[:, 0], 90) if np.isfinite(v[:, 0]).any() else np.nan,
                100 * np.nanmedian(v[:, 1]), np.nanmedian(v[:, 2])))
        print('\n== intra-pulse chirp test (|t|>6 and same-sign share >0.8), by class')
        by = defaultdict(lambda: [0, 0])
        for r in pos:
            if pulse_flag(r):
                cp = r['pulsed'].get('chirp_pulse')
                by[(r['name'], r['snr'])][1] += 1
                if cp and abs(cp['t']) > 6 and cp['frac_same_sign'] > 0.8:
                    by[(r['name'], r['snr'])][0] += 1
        for k, v in sorted(by.items()):
            print('%-22s snr %3g  chirp-in-pulse %d/%d' % (k[0], k[1], v[0], v[1]))
    if what in ('fp', 'all'):
        print('\n== any false flag (chirp or pulsed) on classes that are neither, per class')
        neg = [r for r in rows if not r['truth']['chirp'] and not r['truth']['pulse']]
        table(neg, lambda r: chirp_flag(r) or pulse_flag(r), snrs)


if __name__ == '__main__':
    main()
