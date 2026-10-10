"""Post-hoc analyses on a JSONL: coding hint from run histogram, FSK deviation estimate, standard-rate plausibility,
ACF-based OOK width estimator.   usage: extras_analysis.py file.jsonl"""
import sys, json, collections
import numpy as np
import classify as C, evaluate as E

def main(path):
    recs = [d for d in E.load(path) if not d['variant'].startswith('x_')]
    # --- coding hint from the run histogram (hist = share of runs that are 1,2,3,>=4 unit intervals)
    print('== OOK run-length histogram by coding (share of runs of 1T, 2T, 3T, >=4T), SNR>=15 ==')
    by = collections.defaultdict(list)
    for d in recs:
        if d['variant'].startswith('ook') and d['snr'] >= 15:
            o = (d['feat'].get('ook') or {})
            if o.get('run_hist') and o.get('support', 0) >= 0.9:
                by[d['variant']].append(o['run_hist'])
    for v, h in by.items():
        h = np.array(h)
        print('  %-8s n=%3d mean %s  sd %s' % (v, len(h), np.round(h.mean(0), 3), np.round(h.std(0), 3)))
    # --- FSK deviation estimate
    print('== FSK deviation estimate (hints.dev_hz = mean |discriminator|) vs truth dev_hz; h = 2 dev/baud ==')
    rows = collections.defaultdict(list)
    for d in recs:
        if d['truth']['family'] != 'fsk':
            continue
        o = C.classify(d['feat'])
        if o['label'] != 'fsk' or 'dev_hz' not in o['hints']:
            continue
        dv = d['truth']['parameters'].get('dev_hz')
        if not dv:
            continue
        rows[(d['variant'], d['snr'])].append(o['hints']['dev_hz'] / dv)
    for v in sorted({k[0] for k in rows}):
        line = '  %-10s' % v
        for snr in (5, 10, 15, 20, 30):
            r = rows.get((v, snr)) or rows.get((v, float(snr)))
            line += ' %s' % ('%5.2f(%2d)' % (np.median(r), len(r)) if r else '   -   ')
        print(line, ' (ratio est/true median, n)')
    # --- standard-rate plausibility
    print('== plausibility against the standard-rate list ==')
    n = nstd = nlist = nlist_ok = nsnap_break = nsnap = 0
    for d in recs:
        sr = d['truth']['symbol_rate'] if d['truth']['kind'] in E.HAS_RATE else None
        o = C.classify(d['feat'])
        if sr is None or o['baud_hz'] is None or o['label'] != d['truth']['family']:
            continue
        n += 1
        in_std = any(abs(r - sr) / sr < 1e-3 for r in C.STD_RATES)
        nstd += in_std
        if o['plausible']:
            nlist += 1
            nlist_ok += any(abs(r - sr) / sr < 1e-3 for r in o['plausible'])
        # 3 % snap window
        near = [r for r in C.STD_RATES if abs(r - o['baud_hz']) / o['baud_hz'] <= 0.03]
        if near:
            nsnap += 1
            best = min(near, key=lambda r: abs(r - o['baud_hz']))
            if abs(best - sr) / sr > abs(o['baud_hz'] - sr) / sr + 1e-9:
                nsnap_break += 1
    print('  answered & correct-family: %d ; truth rate is on the standard list: %d (%.0f%%)' % (n, nstd, 100.0 * nstd / max(n, 1)))
    print('  estimate within its own bound of a listed rate: %d, of which the truth: %d' % (nlist, nlist_ok))
    print('  3%% snap window would move %d estimates; %d of those moves make the error WORSE' % (nsnap, nsnap_break))
    # --- OOK ACF-based estimators
    print('== OOK envelope-ACF widths vs chip time (SNR>=15) ==')
    by = collections.defaultdict(list)
    for d in recs:
        if d['variant'].startswith('ook') and d['snr'] >= 15:
            ac = (d['feat'].get('ook') or {}).get('acf') or {}
            T = 1.0 / d['truth']['symbol_rate']
            if ac.get('first_min_s'):
                by[(d['variant'], 'first_min/T')].append(ac['first_min_s'] / T)
            if ac.get('width_s'):
                by[(d['variant'], 'width(0.1)/T')].append(ac['width_s'] / T)
            if ac.get('first_peak_s'):
                by[(d['variant'], 'first_peak/T')].append(ac['first_peak_s'] / T)
    for k, v in sorted(by.items()):
        print('  %-8s %-14s median %.2f  p10 %.2f  p90 %.2f  (n=%d)' % (k[0], k[1], np.median(v), np.percentile(v, 10), np.percentile(v, 90), len(v)))

if __name__ == '__main__':
    main(sys.argv[1])
