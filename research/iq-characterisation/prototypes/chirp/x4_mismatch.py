import sys, time
sys.path.insert(0, '..')
import numpy as np, gen, frontend as fe
from x2_lag import lag_line

snr = float(sys.argv[1]); sf = int(sys.argv[2]); wch_mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
r = gen.make('lora', 2e6, 2.0, snr, seed=3, on_fraction=0.3, sf=sf, dc=5, cfo_hz=3000, offset_hz=40000)
tr = r['truth']
x = r['iq'] - r['iq'].mean()
BW = 125e3
on = gen.on_mask(tr, 'bursts')
for dmult in [0, 0.1, 0.25, 0.4, 0.5, 0.75]:
    fc = tr['center_hz'] + dmult * BW
    Wch = BW * wch_mult
    nin = 1 << 16
    ch = fe.Channeliser(2e6, fc, Wch, nin)
    y = np.concatenate([ch.block(x[a:a + nin]) for a in range(0, len(x) - nin + 1, nin)])
    M = 1 << sf
    # oversample factor os = Wch/BW: line at 1/(M*os^2)...  generalise: chirp rate mu=BW^2/M ; line = mu/Fs'^2 cycles/sample ; block len N
    os_ = wch_mult
    # emulate lag_line with scaled M: effective M_eff = M*os^2 samples per 1/line
    Meff = int(round(M * os_ * os_))
    rat, N = lag_line(y, int(np.log2(Meff)) if abs(np.log2(Meff) - round(np.log2(Meff))) < 1e-9 else 0, 8)
    D = int(2e6 / Wch)
    w_on = np.array([on[i * N * D:(i + 1) * N * D].mean() for i in range(len(rat))])
    print('dmult %.2f  (dF=%6.0f Hz)  in-burst median %.1f  p10 %.1f | null p99 %.1f max %.1f' % (dmult, dmult * BW, np.median(rat[w_on > .99]), np.percentile(rat[w_on > .99], 10), np.percentile(rat[w_on == 0], 99), rat[w_on == 0].max()))
