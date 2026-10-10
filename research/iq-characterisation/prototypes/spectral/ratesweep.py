"""Bandwidth accuracy vs sample rate: same signal, rates 2/4/10/20 Msps, 10 dB and 0 dB, seeds 100-104."""
import sys, json
sys.path.insert(0, '.')
import numpy as np, harness
from multiprocessing import Pool
KINDS = ['nfm-voice', 'wfm', 'bpsk', 'lora-sf7', 'gfsk4', 'fsk2-rect']
RATES = [2e6, 4e6, 10e6, 20e6]

def one(a):
    case, rate, snr, seed = a
    secs = min(1.0, 8e6 / rate)
    try:
        rec = harness.run_capture(case, snr, seed, 'clean', rate=rate, secs=secs)
    except Exception as e:
        return dict(error=repr(e), case=case, rate=rate, snr=snr, seed=seed)
    rec['rate_run'] = rate
    return rec

if __name__ == '__main__':
    tasks = [(c, r, s, sd) for c in KINDS for r in RATES for s in (0, 10, 20) for sd in range(100, 105)
             if not (c == 'gfsk4' and r < 4e6)]
    with Pool(4) as pool:
        res = pool.map(one, tasks, chunksize=2)
    with open('results/ratesweep.jsonl', 'w') as f:
        for r in res: f.write(json.dumps(r, default=harness._js) + '\n')
    print('done', len(res), sum('error' in r for r in res))
    for e in [r for r in res if 'error' in r][:5]: print(e)
