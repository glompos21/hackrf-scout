import sys, time
sys.path.insert(0, '..')
import numpy as np, gen
from x1_dechirp import channelise

def lag_line(y, sf, nsym=8, lag=1):
    M = 1 << sf
    N = nsym * M
    nb = (len(y) - lag) // N
    z = y[lag:lag + nb * N] * np.conj(y[:nb * N])
    z = z.reshape(nb, N)
    w = np.hanning(N).astype(np.float32)
    F = np.abs(np.fft.fft(z * w, axis=1)) ** 2
    # up chirp: +nsym bin ; down: -nsym bin
    k = nsym
    pk = F[:, k - 1:k + 2].max(1)
    # background: bins 6..N/2 excluding line neighbourhood; use median ratio
    bgidx = np.r_[k + 4:N // 2]
    bg = np.median(F[:, bgidx], axis=1) / np.log(2)   # median of exp(mean) = mean*ln2
    return pk / bg, N

if __name__ == '__main__':
    snr = float(sys.argv[1]); sf = int(sys.argv[2]); nsym = int(sys.argv[3])
    r = gen.make('lora', 2e6, 2.0, snr, seed=3, on_fraction=0.3, sf=sf, dc=5, cfo_hz=3000, offset_hz=40000)
    tr = r['truth']
    x = r['iq'] - r['iq'].mean()
    y = channelise(x, 2e6, tr['center_hz'], 125e3, 4096)
    rat, N = lag_line(y, sf, nsym)
    on = gen.on_mask(tr, 'bursts')
    D = 16
    w_on = np.array([on[i * N * D:(i + 1) * N * D].mean() for i in range(len(rat))])
    print('SF%d snr %g nsym %d: blocks %d (full-on %d)' % (sf, snr, nsym, len(rat), (w_on > .99).sum()))
    if (w_on > .99).sum():
        print('  in-burst median %.1f p10 %.1f' % (np.median(rat[w_on > .99]), np.percentile(rat[w_on > .99], 10)))
    print('  outside median %.1f p99 %.1f max %.1f' % (np.median(rat[w_on == 0]), np.percentile(rat[w_on == 0], 99), rat[w_on == 0].max()))
