import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe, css
r = gen.make('lora', 2e6, 5.0, 10, seed=100, on_fraction=0.2, dc=6, cfo_hz=3000, offset_hz=40e3, sf=9)
src = fe.ArraySource(r['iq'], 2e6)
p1 = fe.pass1(src)
mean = np.complex64(p1['mean'])
chans = an.candidate_channels(2e6, [125e3, 250e3], 0.0, 200e3, 0.5)
nin = fe.common_nin(2e6, [125e3, 250e3])
cho = [fe.Channeliser(2e6, fc, bw, nin) for bw, fc in chans]
ys = fe.channelise_stream(src, mean, cho)
print(r['truth']['center_hz'])
for (bw, fc), y in zip(chans, ys):
    z = css.lag_z(y)
    row = []
    for sf in (7, 8, 9, 10, 11, 12):
        lb = css.lag_blocks(y, sf, 64, z=z)
        row.append('sf%d %5.0f/%5.0f' % (sf, lb['up'].max(), lb['dn'].max()))
    print('bw %6.0f fc %7.0f  ' % (bw, fc) + '  '.join(row))
