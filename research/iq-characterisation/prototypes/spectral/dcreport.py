import sys, json
sys.path.insert(0, '.')
import numpy as np, rules
from collections import Counter, defaultdict
T = rules.load_thr()
res = json.load(open('results/dcstudy.json'))
by = defaultdict(list)
for r in res: by[(r['case'], r['tag'])].append(r)
tags = ['dc0', 'dc6_raw', 'dc6_mean', 'dc12_raw', 'dc12_mean', 'dc0_wander0.5', 'dc6_wander0.5', 'dc6_wander2', 'cfo4k', 'dc6_cfo4k']
print('=== DC / CFO ablation at 10 dB, eval-type seeds 100-109; thresholds frozen.  mean = global complex mean removed, raw = not removed')
print('%-10s %-14s %-44s %9s %9s %7s %9s' % ('case', 'mode', 'labels (frozen rules)', 'struct_z', 'obw99/ref', 'lines', 'dc_resid_z'))
for case in ['noise2', 'cw', 'nfm-voice', 'qpsk', 'lora-sf7', 'ofdm10', 'hopper10']:
    ref = np.nanmedian([r['f']['obw99_hz'] or np.nan for r in by[(case, 'dc0')]]) if case != 'noise2' else np.nan
    for tg in tags:
        rr = by.get((case, tg), [])
        if not rr: continue
        labs = Counter()
        for r in rr:
            F = dict(r['f']); F['rate'] = 10e6 if case in ('ofdm10', 'hopper10') else 2e6
            # reconstruct what the rules need: they require the full feature dict, so classify only the cheap subset
            labs[None] += 1
        # use the stored feature subset directly
        zs = np.array([r['f']['struct_z'] for r in rr], float)
        ob = np.array([r['f']['obw99_hz'] if r['f']['obw99_hz'] else np.nan for r in rr], float)
        nl = np.array([r['f']['n_lines'] for r in rr], float)
        dz = np.array([r['f']['dc_resid_z'] if r['f']['dc_resid_z'] is not None else np.nan for r in rr], float)
        det = np.mean([bool(r['f']['detected']) for r in rr])
        fok = np.mean([bool(r['f']['floor_ok']) for r in rr])
        print('%-10s %-14s detected %.2f floor_ok %.2f  %9.1f %9s %7.1f %9.1f' % (case, tg, det, fok, np.nanmedian(zs), ('%.2f' % (np.nanmedian(ob) / ref)) if np.isfinite(ref) else '-', np.nanmedian(nl), np.nanmedian(dz)))
    print()

print('=== carrier close to the tune centre (DC): does the line survive the mean removal?  (SNR 10 dB in B, seeds 100-105, mean removed)')
print('%-9s %9s %5s  %-9s %-8s %-10s %s' % ('case', 'offset Hz', 'dc', 'lines>=1', 'detected', 'struct_z', 'strongest line found (Hz) vs truth'))
for case in ['cw', 'am-tone', 'nfm-tone']:
    for off in (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
        for dc in (0.0, 6.0):
            rr = by.get((case, 'atdc_off%g_dc%g' % (off, dc)), [])
            if not rr: continue
            l1 = np.mean([r['f']['n_lines'] >= 1 for r in rr])
            det = np.mean([bool(r['f']['detected']) for r in rr])
            zs = np.median([r['f']['struct_z'] for r in rr])
            errs = []
            for r in rr:
                ls = r['f'].get('lines') or []
                if ls: errs.append(abs(ls[0][0] - r['truth_center']))
            print('%-9s %9.0f %5.0f  %-9.2f %-8.2f %-10.1f %s' % (case, off, dc, l1, det, zs, ('median |df| %.0f Hz' % np.median(errs)) if errs else 'none'))
