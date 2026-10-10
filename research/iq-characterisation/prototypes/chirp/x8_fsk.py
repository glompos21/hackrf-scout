import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe, css
r = gen.make('fsk2', 2e6, 5.0, 20, seed=100, dc=6, cfo_hz=3000, offset_hz=40e3)
src = fe.ArraySource(r['iq'], 2e6)
p1 = fe.pass1(src)
mean = np.complex64(p1['mean'])
chans = an.candidate_channels(2e6, [125e3, 250e3], 0.0, 200e3, 0.5)
nin = fe.common_nin(2e6, [125e3, 250e3])
cho = [fe.Channeliser(2e6, fc, bw, nin) for bw, fc in chans]
ys = fe.channelise_stream(src, mean, cho)
best = []
for (bw, fc), y in zip(chans, ys):
    z = css.lag_z(y)
    for sf in range(5, 13):
        lb = css.lag_blocks(y, sf, 64, z=z)
        if lb is None: continue
        best.append((max(lb['up'].max(), lb['dn'].max()), bw, fc, sf, lb['up'].max(), lb['dn'].max(), int(np.argmax(np.maximum(lb['up'], lb['dn'])))))
best.sort(reverse=True)
for b in best[:8]:
    print('R=%6.0f bw %6.0f fc %7.0f sf %d  up %6.0f dn %6.0f block %d' % b)
# look at the spectrum of z (sf7, bw125, fc 62500) in the best block
bw, fc = 125e3, 62500.0
y = ys[chans.index((bw, fc))]
z = css.lag_z(y)
for sf in (7,):
    M = 1 << sf; q = M // 8
    m = len(z) // q
    zd = z[:m * q].reshape(m, q).sum(1)
    blk = zd[1000:1064]
    blk = blk - blk.mean()
    F = np.abs(np.fft.fft(blk * np.hanning(64))) ** 2
    print(np.round(F / np.median(F), 1))
