"""E6: how a wrong noise-floor estimate biases the noise-compensated kurtosis m4_sig (the only statistic that needs N).
Oracle band + oracle mask; the noise power used in the compensation is scaled by a factor (dB error).  Seeds 100-104."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A, bench
SEEDS = [100, 101, 102, 103, 104]
ERR_DB = [-2, -1, -0.5, 0, 0.5, 1, 2]
CASES = [('nfm 5dB', 'nfm', 5, {}), ('nfm 10dB', 'nfm', 10, {}), ('nfm 20dB', 'nfm', 20, {}), ('qpsk 10dB', 'qpsk', 10, {}), ('ofdmN 10dB', 'ofdm', 10, dict(bw_hz=500e3))]
print('%-11s %-8s ' % ('case', 'snr_ch') + ' '.join('%+5.1fdB' % e for e in ERR_DB) + '   (m4_sig when the assumed noise floor is off by x dB)')
for label, kind, snr, kw in CASES:
    acc = {e: [] for e in ERR_DB}; sn = []
    for seed in SEEDS:
        r = gen.make(kind, 2e6, 1.0, snr, seed=seed, offset_hz=20e3, dc=6, cfo_hz=2000, return_ci8=False, **kw)
        T = r['truth']; lo, hi = bench.hull(T['bands_hz'])
        R = A.analyze(r['iq'], 2e6, oracle_band=(0.5 * (lo + hi), (hi - lo) * 1.1), oracle_on=lambda ts, TT, d0: np.ones(TT, bool), keep_trace=True)
        tk = R['trace']['tk']; N = R['trace']['noise']; g = R['channel']['g']
        allm = A._moments(tk, np.ones(len(tk['s2']), bool), g)
        for e in ERR_DB:
            st = A.stats_from_sums(allm, N * 10 ** (e / 10.0))
            acc[e].append(st.get('m4_sig', float('nan')))
        sn.append(R['all']['snr_ch_db'])
    print('%-11s %-8.1f ' % (label, np.mean(sn)) + ' '.join('%7.3f' % np.nanmean(acc[e]) for e in ERR_DB))
