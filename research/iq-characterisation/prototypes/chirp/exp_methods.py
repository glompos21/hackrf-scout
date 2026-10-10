"""Method comparison at the ORACLE channel (true centre, BW = 125 kHz, true SF hypothesis):
lag-line (L=64, L=128), dechirp (peak/rest per window), IF-slope (second phase difference), spectrogram ridge.
Threshold of every method = (1 - PFA) quantile of its capture-level maximum over NREP pure-noise channels of the same
length (white Gaussian at the critical rate), PFA = 1e-2 per test.  Detection = max statistic of the signal capture
above that threshold.  Seeds 100..109 (evaluation seeds), noise reps use their own rng.
usage: python3 exp_methods.py OUT.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import gen  # noqa: E402
from chirp import frontend as fe, css, alt  # noqa: E402

PFA = 1e-2
NREP = 300
SNRS = (-12, -9, -6, -3, 0, 3, 6, 10, 15)
SFS = (7, 9, 12)
BW = 125e3


def stats(y, sf):
    out = {}
    z = css.lag_z(y)
    for L in (64, 128):
        lb = css.lag_blocks(y, sf, L=L, z=z)
        out['lag%d' % L] = float(max(lb['up'].max(), lb['dn'].max())) if lb else np.nan
    dc = css.dechirp_blocks(y, sf)
    out['dechirp'] = float(max(dc['up'].max(), dc['dn'].max())) if dc else np.nan
    # a good-faith normalisation of the dechirp statistic to a p-value so that SFs are comparable is not needed here
    s = alt.if_slope_stat(y, sf)
    out['if_slope'] = float(s['matched'].max()) if s else np.nan
    r = alt.ridge_stat(y, sf)
    out['ridge'] = float(r['count'].max()) if r else np.nan
    return out


def chan_oracle(r, fc):
    x = r['iq'] - r['iq'].mean()
    nin = fe.common_nin(2e6, [BW])
    ch = fe.Channeliser(2e6, fc, BW, nin)
    nb = len(x) // nin
    return np.concatenate([ch.block(x[a * nin:(a + 1) * nin]) for a in range(nb)])


if __name__ == '__main__':
    out = sys.argv[1]
    n = int(5.0 * BW)
    res = {'null': {}, 'det': {}}
    t0 = time.time()
    for sf in SFS:
        rng = np.random.default_rng(1000 + sf)
        reps = []
        for i in range(NREP):
            y = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
            reps.append(stats(y, sf))
        thr = {k: float(np.quantile([r[k] for r in reps], 1 - PFA)) for k in reps[0]}
        res['null'][str(sf)] = thr
        print('SF%d null thresholds (PFA %.0e, %d reps): %s  [%.0fs]' % (sf, PFA, NREP, {k: round(v, 2) for k, v in thr.items()}, time.time() - t0), flush=True)
        for snr in SNRS:
            det = {k: 0 for k in thr}
            vals = {k: [] for k in thr}
            nn = 10
            for seed in range(100, 100 + nn):
                rng2 = np.random.default_rng(seed)
                r = gen.make('lora', 2e6, 5.0, float(snr), seed=seed, sf=sf, on_fraction=float(rng2.uniform(0.1, 0.3)),
                             offset_hz=float(rng2.uniform(-60e3, 60e3)), cfo_hz=float(rng2.uniform(-5e3, 5e3)),
                             dc=complex(rng2.uniform(-8, 8), rng2.uniform(-8, 8)), return_ci8=False)
                y = chan_oracle(r, r['truth']['center_hz'])
                s = stats(y, sf)
                for k in thr:
                    vals[k].append(s[k])
                    if s[k] > thr[k]:
                        det[k] += 1
            res['det']['%d_%d' % (sf, snr)] = {k: det[k] / nn for k in det}
            print('SF%d snr %3d  detection: ' % (sf, snr) + '  '.join('%s %3.0f%%' % (k, 100 * v / nn * nn / nn) for k, v in det.items()), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
