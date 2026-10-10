"""Evaluation harness for the amplitude-statistics family.

    python3 bench.py tune  out.jsonl [--snrs 0,10,20,30] [--seeds 0-9]
    python3 bench.py eval  out.jsonl
Each row = one synthetic capture analysed twice: blind (PSD band, HMM mask) and with oracle band + oracle on-mask
(upper bound for the feature itself, separates 'feature fails' from 'segmentation/band finder fails').
Seeds 0-9 are for choosing thresholds, 100-119 for the numbers that are reported.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402

import gen                              # noqa: E402
import ampfeat as A                     # noqa: E402

# name, kind, rate, seconds, kwargs
SCEN = [
    ('noise2', 'noise', 2e6, 1.0, {}),
    ('noise10', 'noise', 10e6, 0.3, {}),
    ('cw', 'cw', 2e6, 1.0, {}),
    ('am_tone', 'am', 2e6, 1.0, dict(audio='tone')),
    ('am_voice', 'am', 2e6, 1.0, dict(audio='voice')),
    ('nfm_voice', 'nfm', 2e6, 1.0, {}),
    ('nfm_tone', 'nfm', 2e6, 1.0, dict(audio='tone')),
    ('wfm', 'wfm', 2e6, 1.0, {}),
    ('ook_pwm', 'ook', 2e6, 1.0, {}),
    ('ook_man', 'ook', 2e6, 1.0, dict(coding='manchester', baud=2400.0)),
    ('fsk2_rect', 'fsk2', 2e6, 1.0, dict(shape='rect')),
    ('fsk2_gauss', 'fsk2', 2e6, 1.0, dict(shape='gauss')),
    ('gfsk', 'gfsk', 4e6, 0.5, {}),
    ('bpsk', 'bpsk', 2e6, 1.0, {}),
    ('qpsk', 'qpsk', 2e6, 1.0, {}),
    ('lora7', 'lora', 2e6, 1.0, {}),
    ('lora9', 'lora', 2e6, 2.0, dict(sf=9)),
    ('ofdm_wide', 'ofdm', 10e6, 0.3, {}),
    ('ofdm_narrow', 'ofdm', 2e6, 0.5, dict(bw_hz=500e3)),
    ('hopper', 'hopper', 10e6, 0.3, {}),
    ('pulsed', 'pulsed', 2e6, 1.0, {}),
    ('pulsed_fast', 'pulsed', 10e6, 0.3, dict(pw_s=2e-6)),
    ('pulsed_lfm', 'pulsed', 2e6, 1.0, dict(mod='lfm', pw_s=20e-6)),
    ('multi_cw_nfm', 'multi', 2e6, 1.0, dict(kinds=('cw', 'nfm'))),
    ('multi_ook_nfm', 'multi', 2e6, 1.0, dict(kinds=('ook', 'nfm'))),
]
SCEN_D = {s[0]: s for s in SCEN}
NOISE_BW0 = {2e6: [100e3, 200e3, 400e3, 1.2e6], 10e6: [200e3, 1.2e6, 5.3e6, 8.1e6]}


def target_label(truth, name):
    """Ground-truth *envelope-level* class from truth only.  Used for scoring the decision layer."""
    if truth['kind'] == 'noise':
        return 'noise'
    return None


def pick_params(name, snr, onmode, seed):
    rng = np.random.default_rng([seed, 7, sum(map(ord, name))])
    onf = 1.0 if onmode == 'native' else float(rng.choice([0.1, 0.2, 0.3]))
    dc = complex(rng.uniform(-8, 8), rng.uniform(-8, 8))
    cfo = float(rng.uniform(-5e3, 5e3))
    off = float(rng.uniform(-60e3, 60e3))
    return rng, onf, dc, cfo, off


def truth_mask(truth, tick_s, T, d0):
    """Fraction-based tick mask of truth on_intervals (tick i covers [(i+d0) tick_s, (i+d0+1) tick_s))."""
    m = np.zeros(T, np.float64)
    for a, b in truth['on_intervals']:
        i0 = (a / tick_s) - d0
        i1 = (b / tick_s) - d0
        if i1 <= 0 or i0 >= T:
            continue
        j0, j1 = int(max(0, math.floor(i0))), int(min(T, math.ceil(i1)))
        if j1 - j0 <= 0:
            continue
        m[j0:j1] += 1.0
        m[j0] -= max(0.0, i0 - j0) if j0 >= 0 else 0.0
        if j1 - 1 >= 0 and j1 - 1 < T:
            m[j1 - 1] -= max(0.0, j1 - i1)
    return m >= 0.5


def event_scores(on, tm, tick_s):
    s_d, e_d = A.runs_of(on)
    s_t, e_t = A.runs_of(tm)
    out = dict(n_truth=int(len(s_t)), n_det=int(len(s_d)))
    out['iou'] = float((on & tm).sum() / max(1, (on | tm).sum()))
    out['duty_det'] = float(on.mean())
    out['duty_tm'] = float(tm.mean())
    if len(s_t) and len(s_d):
        def ov(a0, a1, bs, be):
            return bool(np.any((bs < a1) & (be > a0)))
        out['recall'] = float(np.mean([ov(a, b, s_d, e_d) for a, b in zip(s_t, e_t)]))
        out['precision'] = float(np.mean([ov(a, b, s_t, e_t) for a, b in zip(s_d, e_d)]))
        errs = []
        for a, b in zip(s_t, e_t):
            k = np.flatnonzero((s_d < b) & (e_d > a))
            if len(k):
                errs.append(abs(int(s_d[k[0]]) - int(a)))
        out['edge_err_s'] = float(np.median(errs) * tick_s) if errs else float('nan')
    else:
        out['recall'] = 0.0 if len(s_t) else float('nan')
        out['precision'] = float('nan') if not len(s_d) else 0.0
        out['edge_err_s'] = float('nan')
    return out


def _flat(prefix, d, keys=None):
    out = {}
    if not d:
        return out
    for k, v in d.items():
        if keys is not None and k not in keys:
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[prefix + k] = float(v)
    return out


def flatten(R, truth):
    row = {}
    row.update(_flat('raw_', R['raw']))
    row['has_channel'] = R.get('channel') is not None
    row.update(line_z=R['line']['line_z'], line_frac=R['line']['line_frac'], line_f=R['line']['line_f'], line_over_thr_db=R['line']['line_over_thr_db'],
               n_clusters=R['n_clusters'], band_snr_db=R['band_snr_db'], band_bw=R['band_bw'], band_fc=R['band_fc'],
               floor_ok=bool(R['floor_ok']), floor=R['floor'], dc_est_re=R['dc'].real, dc_est_im=R['dc'].imag,
               band_excess_frac=R['band_excess_frac'])
    row['floor_ratio_truth'] = R['floor'] / truth['noise_psd_lsb2_per_hz']
    if R.get('channel') is None:
        return row
    c = R['channel']
    row.update(ch_src=c['src'], ch_bw=c['bw'], ch_bw_raw=c['bw_raw'], ch_fc=c['fc'], ch_rate_d=c['rate_d'], ch_g=c['g'],
               tick_s=c['tick_s'], ch_T=c['T'], k_tick=c['k_tick'], ch_noise=c['noise'])
    row.update(_flat('all_', R['all']))
    row.update(_flat('on_', R.get('on')))
    row.update(_flat('onse_', R.get('on_se')))
    row.update(_flat('off_', R.get('off')))
    row.update(_flat('det_', R['det']))
    row.update(_flat('run_', R['runs']))
    row.update(_flat('otsu_', R['otsu']))
    row.update(_flat('car_', R['carrier']))
    row.update(duty=R['duty'], noise_ratio_off=R['noise_ratio_off'], p_over_noise=R['p_mean_over_noise'],
               n_on_ticks=R['n_on_ticks'], chip_hz=R['chip'][0], chip_q=R['chip'][1])
    return row


def hull(bands):
    return min(b[0] for b in bands), max(b[1] for b in bands)


def one(args):
    name, snr, onmode, seed = args
    _, kind, rate, sec, kw = SCEN_D[name]
    rng, onf, dc, cfo, off = pick_params(name, snr, onmode, seed)
    if kind == 'noise':
        onf = 1.0
    t0 = time.time()
    try:
        r = gen.make(kind, rate, sec, snr, seed=seed, offset_hz=off, on_fraction=onf, dc=dc, cfo_hz=cfo,
                     return_ci8=False, **kw)
    except ValueError as e:
        return dict(scen=name, kind=kind, snr=snr, onmode=onmode, seed=seed, error=str(e))
    T = r['truth']
    row = dict(scen=name, kind=kind, family=T['family'], rate=rate, seconds=sec, snr=snr, onmode=onmode, seed=seed,
               on_fraction_req=onf, dc_re=dc.real, dc_im=dc.imag, cfo=cfo, offset=off, clip=T['clip_fraction'],
               t_duty=T['on_fraction'], t_n_on=len(T['on_intervals']), t_n_burst=len(T['bursts']),
               t_B=T['occupied_bw_hz'] or 0.0, t_symrate=T['symbol_rate'] or 0.0,
               t_gate=T.get('gate_fraction', 0.0), t_noise_psd=T['noise_psd_lsb2_per_hz'])
    if T['on_intervals']:
        ln = np.array([b - a for a, b in T['on_intervals']])
        row['t_run_med_s'] = float(np.median(ln))
        if len(ln) > 1:
            st = np.array([a for a, b in T['on_intervals']])
            row['t_pri_med_s'] = float(np.median(np.diff(st)))
    # prior for the fallback when the PSD finds nothing: what the sweep knows (centre ~ right, bw ~ right)
    if kind == 'noise':
        bws = NOISE_BW0[rate]
        prior = (float(off + cfo + rng.uniform(-50e3, 50e3)), float(rng.choice(bws)))
        obnd = None
    else:
        bands = T['bands_hull'] if False else (T['bands_hz'] if kind != 'multi' else T['components'][0]['bands_hz'])
        lo, hi = hull(bands)
        fc, B = 0.5 * (lo + hi), hi - lo
        # what the sweep knows: bins of 100 kHz -> bandwidth is a whole number of bins (>= 1), centre +- half a bin
        prior = (float(fc + rng.uniform(-50e3, 50e3)), float(max(1, math.ceil(B / 100e3)) * 100e3))
        obnd = (fc, B * 1.1)
    row.update(prior_fc=prior[0], prior_bw=prior[1])
    R = A.analyze(r['iq'], rate, prior=(None if os.environ.get('NOPRIOR') else prior), keep_trace=True)
    row.update(flatten(R, T))
    if R.get('channel') is not None:
        tr = R['trace']
        tm = truth_mask(T, tr['tick_s'], len(tr['p']), tr['d0'])
        row.update({'ev_' + k: v for k, v in event_scores(tr['on'], tm, tr['tick_s']).items()})
    # oracle band + oracle on-mask: what the features say when segmentation/band finding are perfect
    if obnd is not None:
        Ro = A.analyze(r['iq'], rate, oracle_band=obnd,
                       oracle_on=lambda ts, TT, d0: truth_mask(T, ts, TT, d0))
        if Ro.get('channel') is not None:
            for k, v in _flat('o_all_', Ro['all']).items():
                row[k] = v
            for k, v in _flat('o_on_', Ro.get('on')).items():
                row[k] = v
            for k, v in _flat('o_onse_', Ro.get('on_se')).items():
                row[k] = v
            row['o_duty'] = Ro['duty']
            row['o_ch_bw'] = Ro['channel']['bw']
    row['secs_proc'] = time.time() - t0
    return row


def build_tasks(snrs, seeds, onmodes=('native', 'gated'), names=None):
    tasks = []
    for name, kind, rate, sec, kw in SCEN:
        if names and name not in names:
            continue
        for seed in seeds:
            if kind == 'noise':
                tasks.append((name, 0, 'native', seed))
                continue
            for snr in snrs:
                for om in onmodes:
                    tasks.append((name, snr, om, seed))
    return tasks


def parse_seeds(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv):
    import multiprocessing as mp
    mode, outp = argv[0], argv[1]
    snrs = [0, 5, 10, 15, 20, 30]
    seeds = parse_seeds('0-9') if mode == 'tune' else parse_seeds('100-119')
    names = None
    onmodes = ('native', 'gated')
    for i, a in enumerate(argv):
        if a == '--snrs':
            snrs = [int(x) for x in argv[i + 1].split(',')]
        if a == '--seeds':
            seeds = parse_seeds(argv[i + 1])
        if a == '--names':
            names = argv[i + 1].split(',')
        if a == '--onmodes':
            onmodes = tuple(argv[i + 1].split(','))
    tasks = build_tasks(snrs, seeds, onmodes, names)
    print('tasks', len(tasks), flush=True)
    t0 = time.time()
    with mp.Pool(4) as pool, open(outp, 'w') as f:
        for i, row in enumerate(pool.imap_unordered(one, tasks, chunksize=2)):
            f.write(json.dumps(row) + '\n')
            if i % 100 == 0:
                print(i, '/', len(tasks), '%.0fs' % (time.time() - t0), flush=True)
                f.flush()
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main(sys.argv[1:])
