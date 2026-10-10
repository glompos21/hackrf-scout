"""Run feature extraction over a grid and append JSON lines.   usage: bench.py out.jsonl seeds(tune|eval|a,b,c) [variants] [conds]"""
import sys, os, json, time, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scen, frontend as fe, features

def job(a):
    variant, snr, seed, cond = a
    try:
        t = time.time()
        res, imp = scen.make(variant, snr, seed, cond)
        tg = time.time() - t
        tr = res['truth']
        src = fe.ArraySource(res['iq'], tr['rate'])
        tim = {}
        feat = features.analyze(src, tr['rate'], timings=tim)
        out = dict(variant=variant, snr=snr, seed=seed, cond=cond, imp=dict(offset_hz=imp['offset_hz'], cfo_hz=imp['cfo_hz'], dc=[imp['dc'].real, imp['dc'].imag], on_fraction=imp['on_fraction']),
                   truth=dict(kind=tr['kind'], family=scen.V[variant][4], symbol_rate=tr['symbol_rate'], bw=tr['occupied_bw_hz'], on_fraction=tr['on_fraction'],
                              center_hz=tr['center_hz'], bands=tr['bands_hz'], parameters={k: v for k, v in tr['parameters'].items() if isinstance(v, (int, float, str, bool))}),
                   comps=[dict(kind=c['kind'], family=scen.gen.FAMILY[c['kind']], center_hz=c['center_hz'], symbol_rate=c['symbol_rate'], bands=c['bands_hz'], on_fraction=c['on_fraction'], snr_db=c['snr_db'], bw=c['occupied_bw_hz']) for c in tr.get('components', [])],
                   feat=feat, time=dict(gen=tg, **tim))
        return out
    except Exception as e:
        return dict(variant=variant, snr=snr, seed=seed, cond=cond, error=traceback.format_exc())

def default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

if __name__ == '__main__':
    out = sys.argv[1]
    seeds = {'tune': scen.TUNE_SEEDS, 'eval': scen.EVAL_SEEDS}.get(sys.argv[2]) or [int(s) for s in sys.argv[2].split(',')]
    variants = (sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] not in ('all', 'extras') else ([k for k in scen.V if k.startswith('x_')] if len(sys.argv) > 3 and sys.argv[3] == 'extras' else None)) or [k for k in scen.V if not k.startswith('x_')]
    conds = sys.argv[4].split(',') if len(sys.argv) > 4 else ['cont', 'gated']
    snrs = [float(s) for s in sys.argv[5].split(',')] if len(sys.argv) > 5 else scen.SNRS
    jobs = []
    for v in variants:
        for cond in conds:
            if cond == 'gated' and v not in scen.GATED_VARIANTS:
                continue
            for snr in (snrs if v != 'noise' else [0]):
                for seed in seeds:
                    jobs.append((v, snr, seed, cond))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                d = json.loads(line); done.add((d['variant'], d['snr'], d['seed'], d['cond']))
            except Exception: pass
    jobs = [j for j in jobs if (j[0], float(j[1]), j[2], j[3]) not in done and (j[0], j[1], j[2], j[3]) not in done]
    print('jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(4) as pool, open(out, 'a') as fh:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            fh.write(json.dumps(r, default=default) + '\n'); fh.flush()
            if (i + 1) % 20 == 0:
                print(i + 1, '/', len(jobs), '%.0fs' % (time.time() - t0), flush=True)
