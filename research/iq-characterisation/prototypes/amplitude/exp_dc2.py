"""E4b: a carrier near the tune centre vs the capture-mean (DC) removal, with the final pipeline.  dc=6+6j, CW 15 dB nominal,
1 s at 2 Msps, prior = (0, 100 kHz) i.e. the sweep says 'something at the centre'.  Seeds 100-104.
Reports the coarse (488 Hz bins, DC guard +-3 bins) line test and the fine-PSD carrier test, and the shape statistic."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A
SEEDS = [100, 101, 102, 103, 104]
print('%-9s | %-26s | %-22s | %-10s' % ('cw offset', 'coarse line (over thr dB)', 'fine carrier (over thr dB)', 'carrier_frac'))
for off in (0, 20, 60, 150, 400, 1000, 3000):
    co, fi, fr, ms = [], [], [], []
    for seed in SEEDS:
        r = gen.make('cw', 2e6, 1.0, 15, seed=seed, offset_hz=off, dc=6, return_ci8=False)
        R = A.analyze(r['iq'], 2e6, prior=(0.0, 100e3))
        co.append(R['line']['line_over_thr_db']); fi.append(R['carrier']['carrier_over_thr_db']); fr.append(R['carrier']['carrier_frac'])
    print('%-9d | %6.1f (min %6.1f)            | %6.1f (min %6.1f)        | %.2f' % (off, np.mean(co), np.min(co), np.mean(fi), np.min(fi), np.nanmean(fr)))
