a = open('analyze.py').read()
i = a.index('def refine_sf(')
j = a.index('def pulse_stage(')
new = '''def refine_sf(src, mean, rate, cand, bw, centre, cfg, T):
    """Matched re-analysis of the segments where the chirp test fired: channel centred on the gated-PSD centre with the
    bandwidth hint, every SF; the strongest line decides the SF (a mismatched channel shows lines at shifted
    positions, so the SF read from it can be off by 1-2).  Then the dechirp test on the same matched segments with the
    chosen SF.  Returns dict(sf, R, second, by_sf, dechirp)."""
    D = int(round(rate / bw))
    tile = D * 4096
    lb0 = cand['lb']
    hit = cand['hit_idx']
    Dc = int(round(rate / cand['bw']))
    spans = []
    order = np.argsort(-cand['main'][hit])[:cfg.get('refine_blocks', 24)]
    for h in hit[order]:
        a = int(lb0['start'][h]) * Dc
        spans.append((max(0, a - lb0['span'] * Dc // 2), a + lb0['span'] * Dc + lb0['span'] * Dc // 2))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    best = {}
    ch = fe.Channeliser(rate, centre, bw, tile)
    mean_c = np.complex64(mean)
    used = 0
    ys = []
    for a, b in merged:
        if used > cfg.get('refine_max', 8_000_000):
            break
        ntile = max(1, int(math.ceil((b - a) / tile)))
        x = src.read(a, ntile * tile)
        if len(x) < tile:
            continue
        x = x[:len(x) // tile * tile] - mean_c
        y = np.concatenate([ch.block(x[t * tile:(t + 1) * tile]) for t in range(len(x) // tile)])
        used += len(x)
        ys.append(y)
        z = css.lag_z(y)
        for sf in SFS:
            lb = css.lag_blocks(y, sf, L=cfg['L'], z=z)
            if lb is None:
                continue
            main = lb['up'] if cand['dir'] == 'up' else lb['dn']
            opp = lb['dn'] if cand['dir'] == 'up' else lb['up']
            ok = main > cfg['asym'] * opp
            v = float(np.max(np.where(ok, main, 0))) if ok.any() else 0.0
            best[sf] = max(best.get(sf, 0.0), v)
    if not best:
        return None
    srt = sorted(best.items(), key=lambda kv: -kv[1])
    out = dict(sf=int(srt[0][0]), R=float(srt[0][1]), second=float(srt[1][1]) if len(srt) > 1 else 0.0,
               by_sf={int(k): float(v) for k, v in best.items()}, n_seg=len(ys))
    if cfg.get('dechirp', True) and out['R'] > 0:
        M = 1 << out['sf']
        nwin = 0
        hits = 0
        run = 0
        mxr = 0.0
        thr = None
        for y in ys:
            dc = css.dechirp_blocks(y, out['sf'])
            if dc is None:
                continue
            nw = len(dc['up'])
            nwin += nw
            thr = css.dechirp_threshold(M, cfg['p_dechirp'] / max(nw * len(ys), 1))
            R = dc['up'] if cand['dir'] == 'up' else dc['dn']
            hits += int((R > thr).sum())
            mxr = max(mxr, float(R.max()))
            run = max(run, css.preamble_run(dc if cand['dir'] == 'up' else dict(up=dc['dn'], bin_up=dc['bin_dn']), thr, M))
        if nwin:
            out['dechirp'] = dict(thr=float(thr), n_windows=int(nwin), n_hits=int(hits), max_R=mxr,
                                  hit_frac=hits / nwin, preamble_run=int(run))
    return out


def chirp_stage(outs, chans, rate, floor, cfg, p1, bands, src, nin, bws):
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
                hit = np.flatnonzero(ok)
                passing.append(dict(ci=ci, bw=bw, fc=fc, sf=sf, dir=name, R=float(main[k]), score=float(main[k] / T),
                                    n_hits=int(len(hit)), n_blocks=int(len(main)), block=k, start=int(lb['start'][k]),
                                    span=int(lb['span']), hit_idx=hit, main=main, lb=lb,
                                    adjacent=bool(len(hit) >= 2 and np.any(np.diff(hit) <= 2))))
    out = dict(T=T, n_trials=n_trials, p_block=p_block, max_R=mx, max_score=mx / T, n_pass=len(passing))
    out['flag_lag'] = bool(passing)
    if not passing:
        out['flag'] = False
        return out
    passing.sort(key=lambda d: -d['R'])
    rejected = []
    final = None
    for cand in passing[:6]:
        info = dict(bw=cand['bw'], sf=cand['sf'], R=cand['R'], n_hits=cand['n_hits'])
        if cfg.get('min_adjacent', True) and not cand['adjacent']:
            info['reject'] = 'isolated block'
            rejected.append(info)
            continue
        band = None
        if cfg.get('band_check', True):
            D = int(round(rate / cand['bw']))
            order = np.argsort(-cand['main'][cand['hit_idx']])[:cfg.get('gate_blocks', 24)]
            segs = [(int(cand['lb']['start'][cand['hit_idx'][o]]) * D, int(cand['span']) * D) for o in order]
            g = gated.gated_psd(src, p1['mean'], segs, rate)
            if g is not None:
                band = gated.occupied(g[0], g[1], g[2], floor, cand['fc'])
        info['band'] = band
        cand['band'] = band
        if band is not None and band['measurable']:
            bw_final = _nearest_bw(band['bw'])
            if bw_final is None:
                info['reject'] = 'occupied bandwidth %.0f kHz is not a LoRa bandwidth' % (band['bw'] / 1e3)
                rejected.append(info)
                continue
            cand['bw_final'] = bw_final
            cand['centre'] = band['centre']
            cand['band_confirmed'] = True
            rf = refine_sf(src, p1['mean'], rate, cand, bw_final, band['centre'], cfg, T)
            cand['refine'] = rf
            cand['sf_final'] = rf['sf'] if (rf and rf['R'] > T) else (cand['sf'] if bw_final == cand['bw'] else None)
            if cand['sf_final'] is None:
                info['reject'] = 'no matched line at BW %.0f kHz' % (bw_final / 1e3)
                rejected.append(info)
                continue
        else:
            cand['bw_final'] = cand['bw']
            cand['sf_final'] = cand['sf']
            cand['centre'] = cand['fc']
            cand['band_confirmed'] = False
            if cfg.get('dechirp', True):
                cand['refine'] = refine_sf(src, p1['mean'], rate, cand, cand['bw'], cand['fc'], cfg, T)
        final = cand
        break
    out['rejected'] = rejected
    if final is None:
        out['flag'] = False
        return out
    out['flag'] = True
    out['best'] = {k: v for k, v in final.items() if k not in ('hit_idx', 'ci', 'main', 'lb')}
    out['alts'] = [dict(bw=d['bw'], sf=d['sf'], R=d['R']) for d in passing[1:6]]
    out['sym_time_s'] = (1 << final['sf_final']) / final['bw_final']
    out['dechirp'] = (final.get('refine') or {}).get('dechirp')
    return out


'''
a = a[:i] + new + a[j:]
open('analyze.py', 'w').write(a)
