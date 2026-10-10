"""What happens when the capture holds only part of a LoRa packet?
Packets are cut to n_sym symbols and padded with matching noise to 5 s:
  'end'   : capture ENDS mid-packet  (first n_sym symbols of a packet: preamble, sync, SFD, start of payload)
  'start' : capture STARTS mid-packet (last n_sym symbols: payload only, no preamble)
usage: python3 partial.py OUT.jsonl CFG.json
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import gen
from chirp import analyze as an, frontend as fe, cases

def one(args):
    sf, snr, nsym, where, seed, cfgpath = args
    rng = np.random.default_rng([seed, sf, nsym, int(snr * 10) + 1000])
    err = float(rng.uniform(-60e3, 60e3)); cfo = float(rng.uniform(-5e3, 5e3)); dc = complex(rng.uniform(-8, 8), rng.uniform(-8, 8))
    T = (1 << sf) / 125e3
    rate = 2e6
    secs = max(1.5, 45 * T)
    r = gen.make('lora', rate, secs, float(snr), seed=seed, sf=sf, on_fraction=1.0, offset_hz=err, cfo_hz=cfo, dc=dc, return_ci8=False)
    tr = r['truth']
    cand = [(a, b) for a, b in tr['bursts'] if (b - a) >= (nsym + 1) * T]
    if not cand:
        return None
    a, b = cand[0]
    n = int(round(nsym * T * rate))
    if where == 'end':
        a0 = int(round(a * rate)); seg = r['iq'][a0:a0 + n]
    else:
        b0 = int(round(b * rate)); seg = r['iq'][b0 - n:b0]
    npad = int(5.0 * rate) - len(seg)
    nz = gen.make('noise', rate, npad / rate, 0.0, seed=seed + 777, rms_lsb=tr['noise_rms_lsb'], dc=dc, return_ci8=False)['iq']
    nz = nz[:npad]
    x = np.concatenate([seg, nz]) if where == 'start' else np.concatenate([nz, seg])
    meta = cases.sweep_meta(tr, rng)
    res = an.characterise(fe.ArraySource(x.astype(np.complex64), rate), meta=meta, cfg=an.load_cfg(cfgpath), want_pulse=False)
    c = res['chirp']
    b_ = c.get('best')
    return dict(sf=sf, snr=snr, nsym=nsym, where=where, seed=seed, flag=bool(c['flag']), sf_ok=bool(b_ and b_['sf'] == sf and abs(b_['bw'] - 125e3) < 1),
                R=b_['R'] if b_ else None, max_score=c.get('max_score'), pre=(c.get('dechirp') or {}).get('preamble_run'))

if __name__ == '__main__':
    import multiprocessing as mp
    out, cfgpath = sys.argv[1], sys.argv[2]
    jobs = [(sf, snr, ns, w, seed, cfgpath) for sf in (7, 9) for snr in (5, 10, 20) for ns in (2, 3, 4, 6, 8, 12, 16, 24) for w in ('end', 'start') for seed in range(100, 112)]
    t0 = time.time()
    with mp.Pool(4) as pool, open(out, 'w') as fh:
        for i, d in enumerate(pool.imap_unordered(one, jobs)):
            if d: fh.write(json.dumps(d) + '\n'); fh.flush()
            if (i + 1) % 100 == 0: print(i + 1, len(jobs), '%.0fs' % (time.time() - t0), flush=True)
