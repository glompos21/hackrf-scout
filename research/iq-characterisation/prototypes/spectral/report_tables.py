import sys, json
sys.path.insert(0, '.')
import numpy as np, summ, harness
rows = summ.load(sys.argv[1])
refs = json.load(open('results/refs.json'))
SN = [0, 5, 10, 15, 20, 30]
def nominal(case):
    return refs[case]['hull'] if case.startswith('hopper') else refs[case]['nominal']
def cell(rr, key, fn=lambda x: x):
    a = summ.col(rr, key)
    fin = np.isfinite(a)
    return (np.nanmedian(fn(a)) if fin.any() else np.nan), fin.mean(), a

out = {}
for cond in ('clean', 'imp'):
    print('\n=== OBW99 error vs the 30 dB clean value of the same estimator ("self-reference") and vs nominal B; cond=%s' % cond)
    print('%-15s %9s %9s' % ('case', 'nominal', 'ref30') + ''.join('%16d' % s for s in SN) + '   (median ratio to ref30 [p10..p90]; nominal ratio at 30 dB)')
    for case in harness.CASES:
        if case.startswith('noise'): continue
        ref_rows = [r for r in rows if r['case'] == case and r['cond'] == 'clean' and r['snr'] == 30]
        ref = np.nanmedian(summ.col(ref_rows, 'obw99_hz'))
        line = '%-15s %9.0f %9.0f' % (case, nominal(case), ref)
        for s in SN:
            rr = [r for r in rows if r['case'] == case and r['cond'] == cond and r['snr'] == s]
            a = summ.col(rr, 'obw99_hz') / ref
            if np.isfinite(a).any():
                line += '  %5.2f[%4.2f-%4.2f]' % (np.nanmedian(a), np.nanpercentile(a, 10), np.nanpercentile(a, 90))
            else:
                line += '%16s' % 'n/a'
        print(line + '   nom30: %.2f' % (ref / nominal(case)))
        out.setdefault(cond, {})[case] = dict(nominal=nominal(case), ref30=float(ref))

print('\n=== -20 dB / -10 dB / -3 dB width availability (share of captures where the level is above floor+tolerance) and median ratio to the 30 dB value, clean')
for key in ('bw_3db_hz', 'bw_10db_hz', 'bw_20db_hz'):
    print('-- %s' % key)
    for case in harness.CASES:
        if case.startswith('noise'): continue
        ref_rows = [r for r in rows if r['case'] == case and r['cond'] == 'clean' and r['snr'] == 30]
        ref = np.nanmedian(summ.col(ref_rows, key)) if len(ref_rows) else np.nan
        line = '%-15s ref30 %9s' % (case, '%.0f' % ref if np.isfinite(ref) else 'n/a')
        for s in SN:
            rr = [r for r in rows if r['case'] == case and r['cond'] == 'clean' and r['snr'] == s]
            a = summ.col(rr, key)
            fin = np.isfinite(a)
            line += '  %3.0f%%/%s' % (100 * fin.mean(), '%4.2f' % (np.nanmedian(a) / ref) if fin.any() and np.isfinite(ref) else ' n/a')
        print(line)
    if key != 'bw_20db_hz': print('(only -20 dB shown in full below)') if False else None

print('\n=== median n_lines by case and SNR (clean) / (imp)')
for cond in ('clean', 'imp'):
    print('cond', cond)
    for case in harness.CASES:
        line = '%-15s' % case
        for s in SN:
            rr = [r for r in rows if r['case'] == case and r['cond'] == cond and (r['snr'] == s or case.startswith('noise'))]
            line += '%8.0f' % np.nanmedian(summ.col(rr, 'n_lines'))
        print(line)

print('\n=== hopper: dwell (truth 625 us) and channel count, by SNR, clean / imp')
for cond in ('clean', 'imp'):
    for case in ('hopper2', 'hopper10'):
        for s in SN:
            rr = [r for r in rows if r['case'] == case and r['cond'] == cond and r['snr'] == s and r['feat'].get('hop_dwell_kind') == 'period']
            d = summ.col(rr, 'hop_dwell_s') * 1e6
            n = summ.col([r for r in rows if r['case'] == case and r['cond'] == cond and r['snr'] == s], 'hop_n_ch')
            print('%-9s %-5s snr %2d: dwell median %6.1f us [p10 %6.1f p90 %6.1f] (n=%d)  n_ch median %.0f min %.0f' % (case, cond, s, np.nanmedian(d) if len(d) else np.nan, np.nanpercentile(d, 10) if len(d) else np.nan, np.nanpercentile(d, 90) if len(d) else np.nan, len(d), np.nanmedian(n), np.nanmin(n)))
