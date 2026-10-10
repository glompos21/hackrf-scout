import sys
import numpy as np
sys.path.insert(0, '..')
import gen, frontend as fe
from common import *
from explore1 import prep

kinds = sys.argv[1].split(',')
snrs = [float(s) for s in sys.argv[2].split(',')]
extra = {}
for kind in kinds:
    rate = 2e6
    if kind == 'gfsk':
        rate = 4e6
    for snr in snrs:
        r = gen.make(kind, rate, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False)
        tr = r['truth']
        rec, fs, n_ch, info, b, fb = prep(r, rate)
        p = (np.abs(rec) ** 2).astype(np.float64)
        K = 24
        Lp = int(np.ceil(K * fs / info['neb']))
        a = movmean(p, Lp)
        thr = n_ch * (1 + 5 / np.sqrt(K))
        m = a > thr
        act = m.mean()
        if act > 0:
            Pa = p[m].mean()
            snr_ch = (Pa - n_ch) / n_ch
            V = p[m].var() / Pa ** 2
            s = max(snr_ch, 0)
            Vc = (1 + 2 * s) / (1 + s) ** 2
        else:
            snr_ch = V = Vc = float('nan')
        # discriminator
        z = rec[1:] * np.conj(rec[:-1])
        Ls = max(1, int(round(0.25 * fs / (info['neb'] / 1.0) * 2)))
        zs = movsum_complex(z, Ls)
        sf = np.angle(zs) * fs / (2 * np.pi)
        mm = m[1:]
        sfm = sf[mm] if mm.any() else sf
        sfm = sfm - np.median(sfm)
        sd = sfm.std()
        # robust kurtosis of the smoothed frequency
        ku = ((sfm - sfm.mean()) ** 4).mean() / max(sfm.var(), 1e-9) ** 2
        # peak sharpness
        print('%-5s snr %4.0f | bandw %.0f D %d fs %.0f | act %.2f snr_ch %.1f dB | V %.3f Vc %.3f exc %.3f | f-sd %.0f Hz ku %.2f' % (
            kind, snr, b['hi'] - b['lo'], info['D'], fs, act, 10 * np.log10(max(snr_ch, 1e-3)) if snr_ch == snr_ch else -99, V, Vc, V - Vc, sd, ku))
