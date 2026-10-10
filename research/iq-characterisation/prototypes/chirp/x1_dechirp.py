import sys, time
sys.path.insert(0, '..')
import numpy as np, gen

def channelise(x, rate, fc, bw, nblk):
    """frequency-domain brick-wall selection of [fc-bw/2, fc+bw/2) and critical decimation to rate'=bw.
    x: complex64 (len multiple of nblk*D). returns y (complex64) at bw."""
    D = int(round(rate / bw))
    nin = nblk * D
    nb = len(x) // nin
    out = []
    kc = int(round(fc / rate * nin))
    idx = (np.arange(nblk) - nblk // 2 + kc) % nin
    for b in range(nb):
        X = np.fft.fft(x[b * nin:(b + 1) * nin])
        Y = X[idx]
        Y = np.fft.ifftshift(np.fft.fftshift(Y))  # selected bins are ordered from -bw/2 .. bw/2 around fc
        y = np.fft.ifft(np.fft.ifftshift(Y)) * (nblk / nin) ** 0.5 * 1.0
        out.append(y.astype(np.complex64))
    return np.concatenate(out)

def upchirp(M, up=True):
    n = np.arange(M)
    ph = np.pi * n * n / M - np.pi * n  # f = -bw/2 + bw*n/M  (cycles/sample from -.5 to .5)
    c = np.exp(1j * ph)
    return c if up else np.conj(c)

def dechirp_ratios(y, sf, hop_div=4):
    M = 1 << sf
    ref = np.conj(upchirp(M))
    hop = M // hop_div
    nw = (len(y) - M) // hop + 1
    idx = np.arange(nw)[:, None] * hop + np.arange(M)[None, :]
    seg = y[idx] * ref[None, :]
    F = np.abs(np.fft.fft(seg, axis=1)) ** 2
    pk = F.max(1)
    rest = (F.sum(1) - pk) / (M - 1)
    return pk / rest, hop

if __name__ == '__main__':
    snr = float(sys.argv[1]); sf = int(sys.argv[2])
    r = gen.make('lora', 2e6, 2.0, snr, seed=3, on_fraction=0.3, sf=sf, dc=5, cfo_hz=3000, offset_hz=40000)
    tr = r['truth']
    x = r['iq'] - r['iq'].mean()
    t0 = time.time()
    y = channelise(x, 2e6, tr['center_hz'], 125e3, 4096)
    print('chan', time.time() - t0, len(y))
    t0 = time.time()
    rat, hop = dechirp_ratios(y, sf)
    print('dechirp', time.time() - t0, len(rat))
    on = gen.on_mask(tr, 'bursts')
    # window on-fraction
    D = 16
    w_on = np.array([on[(i * hop) * D:(i * hop + (1 << sf)) * D].mean() for i in range(len(rat))])
    print('ratio in-burst (w_on>0.99): median %.1f  p10 %.1f' % (np.median(rat[w_on > 0.99]), np.percentile(rat[w_on > 0.99], 10)))
    print('ratio outside (w_on==0):    median %.1f  p99 %.1f  max %.1f' % (np.median(rat[w_on == 0]), np.percentile(rat[w_on == 0], 99), rat[w_on == 0].max()))
    print('M=%d  expected coherent M*snr=%.0f' % (1 << sf, (1 << sf) * 10 ** (snr / 10)))
