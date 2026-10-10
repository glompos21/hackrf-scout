"""E5: channel bandwidth mis-set.  The statistic is computed in a channel that is x times the true band (oracle centre).
Shows how kurtosis drifts toward the Gaussian value 2 and SNR_ch drops when the channel is too wide, and what is lost when too
narrow.  Seeds 100-104, 2 Msps, continuous signals, DC and CFO present."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A, bench
SEEDS = [100, 101, 102, 103, 104]
CASES = [('nfm15', 'nfm', 15, {}), ('fsk2g15', 'fsk2', 15, dict(shape='gauss')), ('qpsk15', 'qpsk', 15, {}), ('ofdmN15', 'ofdm', 15, dict(bw_hz=500e3)), ('am_tone15', 'am', 15, dict(audio='tone'))]
FAC = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
if __name__ == '__main__':
    print('%-10s' % 'case' + ''.join('| x%-5g m4/sig/snr  ' % f for f in FAC))
    for label, kind, snr, kw in CASES:
        cells = []
        for fac in FAC:
            m4, ms, sn = [], [], []
            for seed in SEEDS:
                r = gen.make(kind, 2e6, 0.5, snr, seed=seed, offset_hz=20e3, dc=6, cfo_hz=2000, return_ci8=False, **kw)
                T = r['truth']; lo, hi = bench.hull(T['bands_hz']); fc = 0.5 * (lo + hi); B = hi - lo
                bw = B * fac
                if bw > 1.7e6: continue
                R = A.analyze(r['iq'], 2e6, oracle_band=(fc, bw), oracle_on=lambda ts, TT, d0: np.ones(TT, bool))
                m4.append(R['all']['m4']); ms.append(R['all'].get('m4_sig', float('nan'))); sn.append(R['all'].get('snr_ch_db', float('nan')))
            cells.append('| %.2f/%.2f/%5.1f   ' % (np.mean(m4), np.nanmean(ms), np.nanmean(sn)) if m4 else '| n/a               ')
        print('%-10s' % label + ''.join(cells), flush=True)
