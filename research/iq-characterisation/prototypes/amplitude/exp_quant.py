"""E1: how the 8-bit quantisation (low rms_lsb) and clipping (high rms_lsb) bias the amplitude statistics.
Oracle channel + oracle on-mask, so only the statistic itself is on trial.  Seeds 100-104 (not used for tuning)."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A, bench

CASES = [  # label, kind, rate, sec, snr, kw, oracle channel bw factor, band centre fixed
    ('noise', 'noise', 2e6, 0.5, 0, {}, 100e3),
    ('cw15', 'cw', 2e6, 0.5, 15, {}, None),
    ('qpsk15', 'qpsk', 2e6, 0.5, 15, {}, None),
    ('ofdmN15', 'ofdm', 2e6, 0.5, 15, dict(bw_hz=500e3), None),
    ('nfm15', 'nfm', 2e6, 0.5, 15, {}, None),
]
RMS = [0.6, 1.0, 1.5, 2.5, 5, 10, 20, 40, 60, 90]
SEEDS = [100, 101, 102, 103, 104]

def run(case, rms, seed):
    label, kind, rate, sec, snr, kw, bwf = case
    r = gen.make(kind, rate, sec, snr, seed=seed, offset_hz=20e3, dc=0.0, cfo_hz=0.0, rms_lsb=rms, return_ci8=False, **kw)
    T = r['truth']
    if kind == 'noise':
        ob = (20e3, 100e3)
    else:
        lo, hi = bench.hull(T['bands_hz']); ob = (0.5 * (lo + hi), (hi - lo) * 1.1)
    R = A.analyze(r['iq'], rate, oracle_band=ob, oracle_on=lambda ts, TT, d0: np.ones(TT, bool))
    return T, R

if __name__ == '__main__':
    print('%-6s %-7s | %-9s %-7s %-7s | channel (whole capture = continuous signals): m4 (sd) / m4_sig' % ('rms', 'clip%', 'raw_rfsK', 'raw_m4', 'raw_iqK'))
    for rms in RMS:
        out = []
        raw = []
        clip = []
        for case in CASES:
            m4, ms, sn = [], [], []
            for seed in SEEDS:
                T, R = run(case, rms, seed)
                m4.append(R['all']['m4']); ms.append(R['all'].get('m4_sig', float('nan'))); sn.append(R['all'].get('snr_ch_db', float('nan')))
                if case[0] == 'noise':
                    raw.append((R['raw']['rfs_kurt'], R['raw']['m4'], R['raw']['iq_kurt']))
                    clip.append(T['clip_fraction'])
            out.append('%s m4 %.3f(%.3f) sig %.2f snr %.1f' % (case[0], np.mean(m4), np.std(m4), np.nanmean(ms), np.nanmean(sn)))
        rr = np.mean(raw, 0)
        print('%-6.1f %-7.3f | %-9.3f %-7.3f %-7.3f | %s' % (rms, 100 * np.mean(clip), rr[0], rr[1], rr[2], ' | '.join(out)), flush=True)
