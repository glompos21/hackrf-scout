"""Realism stress: HackRF-like baseband filter roll-off (coloured noise), I/Q imbalance, extra slow DC wander,
int8 re-quantisation.  usage: realism.py out.jsonl variants snrs seeds"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import scen, frontend as fe, features, classify as C, evaluate as E, bench

def hackrf_like(iq, rate, bb_hz, imb_db, imb_deg, seed, order=4):
    n = len(iq)
    rng = np.random.default_rng(seed)
    X = np.fft.fft(iq.astype(np.complex64))
    f = np.fft.fftfreq(n, 1.0 / rate)
    H = 1.0 / np.sqrt(1.0 + (np.abs(f) / (bb_hz / 2.0)) ** (2 * order))
    y = np.fft.ifft(X * H.astype(np.float32))
    del X
    g = 10 ** (imb_db / 20.0); ph = np.deg2rad(imb_deg)
    I = y.real; Q = g * (y.imag * np.cos(ph) + y.real * np.sin(ph))
    # slow DC wander: +-1 LSB random walk-ish
    t = np.arange(n) / rate
    wander = 0.8 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))
    I = np.clip(np.round(I + wander), -128, 127); Q = np.clip(np.round(Q - wander), -128, 127)
    return (I + 1j * Q).astype(np.complex64)

def job(a):
    variant, snr, seed, cond = a
    try:
        res, imp = scen.make(variant, snr, seed, cond)
        tr = res['truth']; rate = tr['rate']
        iq = hackrf_like(res['iq'], rate, max(1.75e6, 0.0), 0.5, 3.0, seed)
        src = fe.ArraySource(iq, rate)
        feat = features.analyze(src, rate)
        return dict(variant=variant, snr=snr, seed=seed, cond=cond,
                    imp=dict(offset_hz=imp['offset_hz'], cfo_hz=imp['cfo_hz'], dc=[imp['dc'].real, imp['dc'].imag], on_fraction=imp['on_fraction']),
                    truth=dict(kind=tr['kind'], family=scen.V[variant][4], symbol_rate=tr['symbol_rate'], bw=tr['occupied_bw_hz'], on_fraction=tr['on_fraction'], center_hz=tr['center_hz']),
                    comps=[], feat=feat)
    except Exception:
        import traceback
        return dict(variant=variant, snr=snr, seed=seed, cond=cond, error=traceback.format_exc())

if __name__ == '__main__':
    out = sys.argv[1]; variants = sys.argv[2].split(','); snrs = [float(s) for s in sys.argv[3].split(',')]
    seeds = [int(s) for s in sys.argv[4].split(',')]
    jobs = [(v, s, sd, 'cont') for v in variants for s in (snrs if v != 'noise' else [0]) for sd in seeds]
    with Pool(4) as pool, open(out, 'w') as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r, default=bench.default) + '\n'); fh.flush()
