"""Many noise-only captures: how often does the pipeline output anything but 'none'/'unknown'?
usage: noise_fpr.py out.json n_seeds [rolloff]"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import scen, frontend as fe, features, classify as C, gen, realism

def job(a):
    seed, rate, secs, roll = a
    imp = scen.impair('noise', 0, seed, 'cont')
    res = gen.make('noise', rate, secs, 0.0, seed=seed, offset_hz=imp['offset_hz'], cfo_hz=imp['cfo_hz'], dc=imp['dc'], return_ci8=False)
    iq = res['iq']
    if roll:
        iq = realism.hackrf_like(iq, rate, 1.75e6 if rate <= 2.5e6 else max(1.75e6, 0.8 * rate * 0.3), 0.5, 3.0, seed)
    f = features.analyze(fe.ArraySource(iq, rate), rate)
    o = C.classify(f)
    return dict(seed=seed, rate=rate, roll=roll, stage=f.get('stage'), n_bands=f.get('n_bands'), label=o['label'],
                snr=f.get('snr_psd_db'), bw=f.get('band_bw'), reasons=o['reasons'][:1])

if __name__ == '__main__':
    out = sys.argv[1]; n = int(sys.argv[2]); roll = len(sys.argv) > 3 and sys.argv[3] == 'rolloff'
    jobs = [(1000 + i, 2e6, 5.0, roll) for i in range(n)] + [(2000 + i, 10e6, 1.0, roll) for i in range(n // 3)]
    res = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
    json.dump(res, open(out, 'w'))
    for rate in (2e6, 10e6):
        rr = [r for r in res if r['rate'] == rate]
        c = collections.Counter(r['label'] for r in rr)
        c2 = collections.Counter(r['stage'] for r in rr)
        print('rate %g: n=%d labels %s stages %s' % (rate, len(rr), dict(c), dict(c2)))
