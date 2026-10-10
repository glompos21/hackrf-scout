"""Run benchmark cases through chirp.analyze.characterise in parallel and append JSON lines.

usage: python3 harness.py OUT.jsonl SEEDS(a:b) SNRS(csv|std|low) NAMES(csv|all|css|pulsed|neg) MODE(meta|blind) [procs]
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import gen  # noqa: E402
from chirp import analyze as an, frontend as fe, cases  # noqa: E402


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items() if k not in ('profile',)}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


def run_case(args):
    name, snr, seed, mode, cfgpath = args
    t0 = time.time()
    try:
        spec = cases.SPECS[name]
        gk, imp, rng = cases.case_params(name, snr, seed)
        secs = spec.get('secs', 5.0)
        r = gen.make(spec['kind'], spec['rate'], secs, float(snr), seed=seed, return_ci8=False, **gk)
        tr = r['truth']
        meta = cases.sweep_meta(tr, rng, spec.get('target'), spec) if mode == 'meta' else None
        src = fe.ArraySource(r['iq'], spec['rate'])
        cfg = an.load_cfg(cfgpath) if cfgpath else an.load_cfg()
        t1 = time.time()
        res = an.characterise(src, meta=meta, cfg=cfg)
        out = dict(name=name, snr=snr, seed=seed, mode=mode, rate=spec['rate'], imp=jclean(imp), meta=meta,
                   truth=jclean(cases.truth_labels(name, tr)), clip=tr['clip_fraction'],
                   chirp=jclean(res['chirp']), pulsed=jclean(res['pulsed']), label=res['label'], verdict=jclean(res['verdict']),
                   timing=res['timing'], t_total=time.time() - t0, t_gen=t1 - t0)
        return out
    except Exception:
        return dict(name=name, snr=snr, seed=seed, mode=mode, error=traceback.format_exc())


def expand(names):
    if names == 'all':
        return list(cases.SPECS)
    if names == 'css':
        return [n for n, s in cases.SPECS.items() if s.get('group') in ('lora',)]
    if names == 'pulsed':
        return [n for n, s in cases.SPECS.items() if s.get('group') == 'pulsed']
    if names == 'neg':
        return [n for n, s in cases.SPECS.items() if s.get('group') not in ('lora', 'pulsed', 'mix')]
    if names == 'mix':
        return [n for n, s in cases.SPECS.items() if s.get('group') == 'mix']
    return names.split(',')


if __name__ == '__main__':
    import multiprocessing as mp
    out = sys.argv[1]
    a, b = [int(v) for v in sys.argv[2].split(':')]
    snrs = []
    for tok in sys.argv[3].split(','):
        if tok == 'std':
            snrs += list(cases.SNRS)
        elif tok == 'low':
            snrs += list(cases.SNRS_LOW)
        else:
            snrs.append(float(tok))
    names = expand(sys.argv[4])
    mode = sys.argv[5]
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    cfgpath = sys.argv[7] if len(sys.argv) > 7 else None
    jobs = [(n, s, sd, mode, cfgpath) for n in names for s in snrs for sd in range(a, b)]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                d = json.loads(line)
                done.add((d['name'], d['snr'], d['seed'], d['mode']))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print('%d jobs (%d already done)' % (len(jobs), len(done)), flush=True)
    t0 = time.time()
    with mp.Pool(procs) as pool, open(out, 'a') as fh:
        for i, d in enumerate(pool.imap_unordered(run_case, jobs, chunksize=1)):
            fh.write(json.dumps(d) + '\n')
            fh.flush()
            if 'error' in d:
                print('ERROR', d['name'], d['snr'], d['seed'], d['error'][-400:], flush=True)
            if (i + 1) % 50 == 0:
                print('%d/%d  %.0fs' % (i + 1, len(jobs), time.time() - t0), flush=True)
    print('done %.0fs' % (time.time() - t0))
