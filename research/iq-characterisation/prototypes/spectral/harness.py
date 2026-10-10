"""Capture matrix + parallel feature extraction for the spectral family.

    python3 harness.py tune  out.jsonl            # seeds 0-9   (thresholds are chosen on these)
    python3 harness.py eval  out.jsonl            # seeds 100-119 (reported numbers)

One JSON line per capture: case, snr, seed, cond, truth subset, features, timings.  Re-runnable: lines
already present in the output file (same key) are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gen  # noqa: E402
import sp_feat  # noqa: E402
import sp_stream  # noqa: E402

SNRS = (0, 5, 10, 15, 20, 30)

# name: (kind, rate, seconds, kw)   -- same variants as the bench self-test
CASES = OrderedDict([
    ('noise2', ('noise', 2e6, 1.0, {})),
    ('noise10', ('noise', 10e6, 0.5, {})),
    ('cw', ('cw', 2e6, 1.0, {})),
    ('am-tone', ('am', 2e6, 1.0, {})),
    ('am-voice', ('am', 2e6, 1.0, {'audio': 'voice'})),
    ('nfm-voice', ('nfm', 2e6, 1.0, {})),
    ('nfm-tone', ('nfm', 2e6, 1.0, {'audio': 'tone'})),
    ('wfm', ('wfm', 2e6, 1.0, {})),
    ('ook-pwm', ('ook', 2e6, 1.0, {})),
    ('ook-manch', ('ook', 2e6, 1.0, {'coding': 'manchester', 'baud': 4000.0})),
    ('fsk2-rect', ('fsk2', 2e6, 1.0, {'shape': 'rect'})),
    ('fsk2-gauss', ('fsk2', 2e6, 1.0, {'shape': 'gauss'})),
    ('gfsk4', ('gfsk', 4e6, 1.0, {})),
    ('bpsk', ('bpsk', 2e6, 1.0, {})),
    ('qpsk', ('qpsk', 2e6, 1.0, {})),
    ('lora-sf7', ('lora', 2e6, 1.0, {'sf': 7})),
    ('lora-sf9', ('lora', 2e6, 1.2, {'sf': 9})),
    ('ofdm2', ('ofdm', 2e6, 1.0, {})),
    ('ofdm10', ('ofdm', 10e6, 0.5, {})),
    ('ofdm-narrow', ('ofdm', 2e6, 1.0, {'bw_hz': 600e3, 'mod': '16qam'})),
    ('hopper2', ('hopper', 2e6, 1.0, {})),
    ('hopper10', ('hopper', 10e6, 0.5, {})),
    ('pulsed', ('pulsed', 2e6, 1.0, {})),
    ('pulsed-fast', ('pulsed', 10e6, 0.3, {'pw_s': 2e-6, 'pri_s': 100e-6})),
    ('pulsed-lfm', ('pulsed', 2e6, 1.0, {'mod': 'lfm', 'pw_s': 50e-6, 'pri_s': 2e-3})),
    ('multi-cw+nfm', ('multi', 2e6, 1.0, {'kinds': ('cw', 'nfm'), 'rel_offsets_hz': (-350e3, 250e3)})),
    ('multi-ook+qpsk', ('multi', 2e6, 1.0, {'kinds': ('ook', 'qpsk'), 'rel_offsets_hz': (-400e3, 300e3)})),
])

# truth class used for the "which cue separates what" tables
CLASS = {
    'noise2': 'noise', 'noise10': 'noise', 'cw': 'carrier', 'am-tone': 'narrow', 'am-voice': 'narrow',
    'nfm-voice': 'narrow', 'nfm-tone': 'narrow', 'wfm': 'medium', 'ook-pwm': 'narrow', 'ook-manch': 'narrow',
    'fsk2-rect': 'narrow', 'fsk2-gauss': 'narrow', 'gfsk4': 'wide', 'bpsk': 'medium', 'qpsk': 'medium',
    'lora-sf7': 'medium', 'lora-sf9': 'medium', 'ofdm2': 'ofdm', 'ofdm10': 'ofdm', 'ofdm-narrow': 'ofdm',
    'hopper2': 'hopper', 'hopper10': 'hopper', 'pulsed': 'medium', 'pulsed-fast': 'wide', 'pulsed-lfm': 'medium',
    'multi-cw+nfm': 'multi', 'multi-ook+qpsk': 'multi',
}

CONDS = ('clean', 'imp')          # clean: offset only;  imp: + DC 6+6j, CFO +-4 kHz, 10-30 % on-fraction


def cond_params(case, cond, seed, snr_i):
    kind, rate, secs, kw = CASES[case]
    rng = np.random.default_rng([int(seed), list(CASES).index(case), int(snr_i), CONDS.index(cond) if cond in CONDS else 9])
    span = 30e3 if rate <= 4e6 else 100e3
    p = dict(offset_hz=float(rng.uniform(-span, span)))
    if cond == 'imp':
        p.update(dc=float(rng.choice([-1, 1]) * 6.0), cfo_hz=float(rng.uniform(-4e3, 4e3)),
                 on_fraction=float(rng.uniform(0.1, 0.3)))
    return p


def run_capture(case, snr, seed, cond, extra=None, remove_dc=True, usable_frac=0.9, frontend=None, full=True,
                rate=None, secs=None, nfft=None, dc_guard_bins=2, guard_obw=False):
    """Generate one capture and extract the features.  Returns the record dict."""
    kind, rate0, secs0, kw = CASES[case]
    rate = rate or rate0
    secs = secs or secs0
    snr_i = SNRS.index(snr) if snr in SNRS else 0
    p = cond_params(case, cond, seed, snr_i)
    if rate != rate0:
        p['offset_hz'] = p['offset_hz'] * rate / rate0 if rate > rate0 else p['offset_hz']
    if extra:
        p.update(extra)
    if kind == 'noise':
        p.pop('on_fraction', None)
    t0 = time.time()
    r = gen.make(kind, rate, secs, float(snr), seed=int(seed), return_ci8=False, **p, **kw)
    tg = time.time() - t0
    iq = r['iq']
    if frontend is not None:
        import realism
        iq = realism.apply(iq, rate, seed=seed, **frontend)
    t0 = time.time()
    sp = sp_stream.analyse_stream(lambda: sp_stream.iter_array(iq), rate, remove_dc=remove_dc, nfft=nfft)
    t1 = time.time() - t0
    t0 = time.time()
    F = sp_feat.features(sp, usable_frac=usable_frac, dc_guard_bins=dc_guard_bins, guard_obw=guard_obw)
    t2 = time.time() - t0
    tr = r['truth']
    tt = dict(kind=tr['kind'], rate=tr['rate'], seconds=tr['seconds'], occupied_bw_hz=tr['occupied_bw_hz'],
              bands_hz=tr['bands_hz'], symbol_rate=tr['symbol_rate'], on_fraction=tr['on_fraction'],
              gate_fraction=tr.get('gate_fraction'), clip_fraction=tr['clip_fraction'], center_hz=tr['center_hz'],
              offset_hz=tr['offset_hz'], cfo_hz=tr['cfo_hz'], dc=[tr['dc_lsb'].real, tr['dc_lsb'].imag],
              noise_psd=tr['noise_psd_lsb2_per_hz'], on_power=tr.get('on_power_lsb2'))
    if kind == 'hopper':
        tt['hop_dwell_s'] = 625e-6
        tt['n_hop_ch'] = 5
    rec = dict(case=case, snr=snr, seed=int(seed), cond=cond, truth=tt, feat=F, t_gen=tg, t_pass=t1, t_feat=t2,
               remove_dc=remove_dc, usable_frac=usable_frac)
    return rec


def _task(a):
    try:
        return run_capture(*a)
    except Exception as e:                       # keep going, record the failure
        return dict(case=a[0], snr=a[1], seed=a[2], cond=a[3], error=repr(e))


def key_of(rec):
    return '%s|%s|%s|%s' % (rec['case'], rec['snr'], rec['seed'], rec['cond'])


def build_tasks(seeds, cases=None, snrs=SNRS, conds=CONDS):
    tasks = []
    for case in (cases or CASES):
        kind = CASES[case][0]
        for cond in conds:
            for seed in seeds:
                if kind == 'noise':
                    tasks.append((case, 0, seed, cond))
                else:
                    for snr in snrs:
                        tasks.append((case, snr, seed, cond))
    return tasks


def run_tasks(tasks, out, procs=4):
    import multiprocessing as mp
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'error' not in d:
                        done.add(key_of(d))
                except Exception:
                    pass
    todo = [t for t in tasks if '%s|%s|%s|%s' % (t[0], t[1], t[2], t[3]) not in done]
    print('tasks %d, already done %d, to run %d' % (len(tasks), len(done), len(todo)), flush=True)
    t0 = time.time()
    with open(out, 'a') as f, mp.Pool(procs) as pool:
        for i, rec in enumerate(pool.imap_unordered(_task, todo, chunksize=2)):
            f.write(json.dumps(rec, default=_js) + '\n')
            f.flush()
            if 'error' in rec:
                print('ERR', rec, flush=True)
            if (i + 1) % 100 == 0:
                print('%d/%d  %.0fs' % (i + 1, len(todo), time.time() - t0), flush=True)


def _js(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


if __name__ == '__main__':
    mode, out = sys.argv[1], sys.argv[2]
    seeds = range(0, 10) if mode == 'tune' else range(100, 120)
    cases = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    run_tasks(build_tasks(seeds, cases), out)
