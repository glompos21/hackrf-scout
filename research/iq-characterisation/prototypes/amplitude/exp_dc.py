"""E4: DC offset / LO leakage.  With and without capture-mean subtraction: false 'carrier' on noise, bias of the channel
kurtosis, and how close to 0 Hz a real carrier can sit before it is removed with the DC.  Seeds 100-104."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A, bench

SEEDS = [100, 101, 102, 103, 104]
def run(kind, snr, dc, sub, seed, offset=20e3, prior=(0.0, 100e3), **kw):
    r = gen.make(kind, 2e6, 0.5, snr, seed=seed, offset_hz=offset, dc=dc, return_ci8=False, **kw)
    cfg = dict(A.CFG); cfg['subtract_dc'] = sub
    R = A.analyze(r['iq'], 2e6, prior=prior, cfg=cfg)
    return r['truth'], R

if __name__ == '__main__':
    print('--- noise only, prior channel = 100 kHz around DC')
    print('%-5s %-5s | %-8s %-8s %-9s %-8s | %-10s' % ('dc', 'sub', 'clusters', 'line_z', 'ch_m4', 'ch_snr', 'raw_rfsK'))
    for dc in (0, 3, 6, 12, 25):
        for sub in (True, False):
            cl, lz, m4, sn, rk = [], [], [], [], []
            for seed in SEEDS:
                T, R = run('noise', 0, dc, sub, seed)
                cl.append(R['n_clusters']); lz.append(R['line']['line_z']); m4.append(R['all']['m4']); sn.append(R['all']['snr_ch_db']); rk.append(R['raw']['rfs_kurt'])
            print('%-5s %-5s | %-8.1f %-8.1f %-9.3f %-8.1f | %-10.3f' % (dc, sub, np.mean(cl), np.mean(lz), np.mean(m4), np.mean(sn), np.mean(rk)))
    print('--- carrier (cw, 15 dB nominal) at small offset from DC, dc=6: is it still found?')
    print('%-9s | %-6s %-9s %-9s %-8s' % ('offset_Hz', 'sub', 'line_z', 'line_f', 'clusters'))
    for off in (0, 150, 400, 1000, 2000, 5000):
        for sub in (True,):
            lz, lf, cl = [], [], []
            for seed in SEEDS:
                T, R = run('cw', 15, 6, sub, seed, offset=off)
                lz.append(R['line']['line_z']); lf.append(R['line']['line_f']); cl.append(R['n_clusters'])
            print('%-9d | %-6s %-9.1f %-9.0f %-8.1f' % (off, sub, np.mean(lz), np.mean(lf), np.mean(cl)))
