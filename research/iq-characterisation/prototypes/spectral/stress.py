"""Front-end stress: break the 'white floor, clean edges, constant DC' assumptions on purpose.
Seeds 100-109, thresholds frozen.  Output results/stress.jsonl then eval via stress_report.py."""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
import harness
from multiprocessing import Pool

CASES = ['noise2', 'noise10', 'cw', 'nfm-voice', 'qpsk', 'lora-sf7', 'gfsk4', 'ofdm2', 'ofdm10', 'hopper10', 'multi-cw+nfm']
SNRS = [5, 15]
VARIANTS = {
    # name: (filter as fraction of rate or None, usable_frac rule, extra front-end kwargs, dc)
    'ideal':          dict(fr=None,  usable='0.9',     fe={}, dc=0.0),
    'rolloff_known':  dict(fr=0.875, usable='matched', fe={}, dc=0.0),
    'rolloff_blind':  dict(fr=0.875, usable='0.9',     fe={}, dc=0.0),
    'rolloff_narrow': dict(fr=0.6,   usable='matched', fe={}, dc=0.0),
    'rolloff+iq+dcw': dict(fr=0.875, usable='matched', fe=dict(dc_wander_lsb=0.5, iq_gain_db=0.5, iq_phase_deg=2.0), dc=0.0),
    'all+dc6':        dict(fr=0.875, usable='matched', fe=dict(dc_wander_lsb=0.5, iq_gain_db=0.5, iq_phase_deg=2.0), dc=6.0),
}

def one(a):
    case, snr, seed, cond, vname = a
    v = VARIANTS[vname]
    kind, rate, secs, kw = harness.CASES[case]
    fe = dict(v['fe'])
    if v['fr']:
        fe['filter_hz'] = v['fr'] * rate
    usable = 0.9 if v['usable'] == '0.9' else min(0.9, 0.75 * v['fr'])
    extra = dict(dc=v['dc']) if v['dc'] else {}
    try:
        rec = harness.run_capture(case, snr, seed, cond, extra=extra, usable_frac=usable, frontend=fe or None)
    except Exception as e:
        return dict(error=repr(e), case=case, snr=snr, seed=seed, cond=cond, variant=vname)
    rec['variant'] = vname
    return rec

if __name__ == '__main__':
    tasks = []
    for vname in VARIANTS:
        for case in CASES:
            kind = harness.CASES[case][0]
            for seed in range(100, 110):
                for snr in ([0] if kind == 'noise' else SNRS):
                    tasks.append((case, snr, seed, 'clean', vname))
    t = time.time()
    with open('results/stress.jsonl', 'w') as f, Pool(4) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, tasks, chunksize=2)):
            f.write(json.dumps(rec, default=harness._js) + '\n')
            if (i + 1) % 100 == 0: print(i + 1, len(tasks), time.time() - t, flush=True)
