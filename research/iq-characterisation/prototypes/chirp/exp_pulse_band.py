"""Oracle experiment: how much does a band-limited (matched) envelope help pulse detection?
Channelise at the TRUE signal centre with the nominal bandwidth (decimate by 4 -> 500 kHz at 2 Msps), run the same
detector with the same false-alarm setting.  Compare with the full-band detector (what analyze.py does)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import gen
from chirp import analyze as an, frontend as fe, cases, pulse as pl

def run(name, snr, seed, D=4):
    spec = cases.SPECS[name]
    gk, imp, rng = cases.case_params(name, snr, seed)
    r = gen.make(spec['kind'], spec['rate'], 5.0, float(snr), seed=seed, return_ci8=False, **gk)
    tr = r['truth']
    rate = spec['rate']
    src = fe.ArraySource(r['iq'], rate)
    p1 = fe.pass1(src); fl, _, _ = fe.find_bands(p1)
    mean = np.complex64(p1['mean'])
    out = {}
    # full band
    d = pl.PulseDetector(fl * rate)
    for c in src.chunks(1 << 20):
        cc = c - mean; d.feed((cc.real ** 2 + cc.imag ** 2).astype(np.float32))
    ev = d.finish()
    out['full'] = len(ev)
    # oracle band-limited
    bw = rate / D
    nin = 65536
    ch = fe.Channeliser(rate, tr['center_hz'], bw, nin)
    d2 = pl.PulseDetector(fl * rate / D)
    for blk in fe.rechunk(src, nin):
        y = ch.block(blk - mean)
        d2.feed((y.real ** 2 + y.imag ** 2).astype(np.float32))
    out['band'] = len(d2.finish())
    out['truth'] = len(tr['on_intervals'])
    return out

if __name__ == '__main__':
    name = sys.argv[1]
    for snr in (0, 2, 4, 6, 8, 10, 15):
        res = [run(name, snr, s) for s in range(100, 104)]
        print('%s snr %2d: truth %s | full-band events %s | band-limited (oracle, D=4) events %s' % (name, snr, [x['truth'] for x in res], [x['full'] for x in res], [x['band'] for x in res]), flush=True)
