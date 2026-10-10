"""Signal at/near the tune centre (where the HackRF DC spike sits).  usage: offsets.py out.jsonl"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import scen, frontend as fe, features, classify as C, bench, gen

OFFS = [0.0, 300.0, 1500.0, 5000.0, 20000.0]
VARS = ['cw', 'nfm_voice', 'am_voice', 'ook_pwm', 'fsk2_rect', 'qpsk']

def job(a):
    v, off, snr, seed = a
    try:
        kind, rate, secs, kw, fam = scen.V[v]
        res = gen.make(kind, rate, secs, float(snr), seed=seed, offset_hz=off, cfo_hz=0.0, dc=complex(4.0, 3.0), return_ci8=False, **kw)
        tr = res['truth']
        feat = features.analyze(fe.ArraySource(res['iq'], rate), rate)
        return dict(variant=v, snr=snr, seed=seed, cond='cont', offset=off, imp=dict(offset_hz=off), truth=dict(kind=tr['kind'], family=fam, symbol_rate=tr['symbol_rate'], center_hz=tr['center_hz']), comps=[], feat=feat)
    except Exception:
        import traceback
        return dict(variant=v, snr=snr, seed=seed, cond='cont', offset=off, error=traceback.format_exc())

if __name__ == '__main__':
    out = sys.argv[1]
    jobs = [(v, off, 15.0, sd) for v in VARS for off in OFFS for sd in range(200, 204)]
    with Pool(4) as pool, open(out, 'w') as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r, default=bench.default) + '\n'); fh.flush()
