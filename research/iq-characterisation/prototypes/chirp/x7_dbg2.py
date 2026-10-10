import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe, css
for sf, snr in ((9, 10), (7, 10), (9, 30)):
    r = gen.make('lora', 2e6, 5.0, snr, seed=100, on_fraction=0.2, dc=6, cfo_hz=3000, offset_hz=40e3, sf=sf)
    src = fe.ArraySource(r['iq'], 2e6)
    p1 = fe.pass1(src)
    mean = np.complex64(p1['mean'])
    print('--- SF%d snr %d truth centre %g' % (sf, snr, r['truth']['center_hz']))
    for bw in (125e3, 250e3, 500e3):
        nin = fe.common_nin(2e6, [bw])
        ch = fe.Channeliser(2e6, 43000, bw, nin)
        y = fe.channelise_stream(src, mean, [ch])[0]
        z = css.lag_z(y)
        row = []
        for s in range(5, 13):
            lb = css.lag_blocks(y, s, 64, z=z)
            row.append('sf%d %4.0f/%4.0f' % (s, lb['up'].max(), lb['dn'].max()))
        print('bw %6.0f ' % bw + ' '.join(row))
