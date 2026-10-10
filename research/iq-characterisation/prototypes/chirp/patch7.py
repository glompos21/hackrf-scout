a = open('analyze.py').read()
i = a.index('def chirp_stage(')
j = a.index('def pulse_stage(')
new = '''def chirp_stage(outs, chans, rate, floor, cfg, p1, bands, src, nin, bws):
    L = cfg['L']
    ys = [np.concatenate(o) if o else np.zeros(0, np.complex64) for o in outs]
    lag = []
    n_trials = 0
    for ci, ((bw, fc), y) in enumerate(zip(chans, ys)):
        z = css.lag_z(y) if len(y) > 2 else None
        if z is None:
            continue
        for sf in cfg.get('sfs', SFS):
            lb = css.lag_blocks(y, sf, L=L, z=z)
            if lb is None:
                continue
            n_trials += 2 * len(lb['up'])
            lag.append((ci, bw, fc, sf, lb))
    if not lag:
        return dict(flag=False, why='capture too short for any candidate')
    p_block = cfg['p_capture'] / max(n_trials, 1)
    T = lag_null_threshold(L, p_block)
    passing = []
    mx = 0.0
    for ci, bw, fc, sf, lb in lag:
        up, dn = lb['up'], lb['dn']
        mx = max(mx, float(up.max()), float(dn.max()))
        for name, main, opp in (('up', up, dn), ('down', dn, up)):
            ok = (main > T) & (main > cfg['asym'] * opp)
            if ok.any():
                k = int(np.argmax(np.where(ok, main, 0)))
                passing.append(dict(ci=ci, bw=bw, fc=fc, sf=sf, dir=name, R=float(main[k]), score=float(main[k] / T),
                                    n_hits=int(ok.sum()), n_blocks=int(len(main)), block=k, start=int(lb['start'][k]),
                                    span=int(lb['span']), hit_idx=np.flatnonzero(ok)))
    out = dict(T=T, n_trials=n_trials, p_block=p_block, max_R=mx, max_score=mx / T, n_pass=len(passing))
    if not passing:
        out['flag'] = False
        return out
    passing.sort(key=lambda d: -d['R'])
    best = passing[0]
    out['flag'] = True
    out['best'] = {k: v for k, v in best.items() if k not in ('hit_idx', 'ci')}
    out['alts'] = [dict(bw=d['bw'], sf=d['sf'], R=d['R']) for d in passing[1:6]]
    out['sym_time_s'] = (1 << best['sf']) / best['bw']
    out['rate_hz_per_s'] = best['bw'] ** 2 / (1 << best['sf'])
    # dechirp confirmation on the winning channel / SF
    if cfg.get('dechirp', True):
        y = ys[best['ci']]
        M = 1 << best['sf']
        dc = css.dechirp_blocks(y, best['sf'])
        if dc is not None:
            nw = len(dc['up'])
            thr = css.dechirp_threshold(M, cfg['p_dechirp'] / max(nw, 1))
            R = dc['up'] if best['dir'] == 'up' else dc['dn']
            hits = int((R > thr).sum())
            # symbol-aligned preamble evidence (every 4th window)
            run = css.preamble_run(dc if best['dir'] == 'up' else dict(up=dc['dn'], bin_up=dc['bin_dn']), thr, M)
            out['dechirp'] = dict(thr=float(thr), n_windows=int(nw), n_hits=hits, max_R=float(R.max()),
                                  hit_frac=hits / nw, preamble_run=int(run))
    return out


'''
a = a[:i] + new + a[j:]
a = a.replace("    asym=3.0,                # main line must exceed the mirror line by this factor", "    asym=3.0,                # main line must exceed the mirror line by this factor\n    p_dechirp=1e-2,          # capture-level false-alarm budget of the dechirp hit count\n    dechirp=True,")
open('analyze.py', 'w').write(a)
