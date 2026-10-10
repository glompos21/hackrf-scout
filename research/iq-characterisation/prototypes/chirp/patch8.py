a = open('analyze.py').read()
i = a.index('def chirp_stage(')
j = a.index('def pulse_stage(')
new = '''def _nearest_bw(bw3, bws=LORA_BWS, tol=0.3):
    best = min(bws, key=lambda b: abs(math.log(b / bw3)))
    return best if abs(math.log(best / bw3)) <= math.log(1 + tol) else None


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
                band = gated.occupied(g[0], g[1], g[2], floor, cand['fc'], cand['bw'])
        info['band'] = band
        if band is not None and band['measurable']:
            r = band['bw3'] / cand['bw']
            bw_final = _nearest_bw(band['bw3'])
            if bw_final is None or r < cfg['bw_ratio_min'] * 0.5 and False:
                pass
            ok_bw = (bw_final is not None)
            if not ok_bw:
                info['reject'] = 'occupied bandwidth %.0f kHz is not a LoRa bandwidth' % (band['bw3'] / 1e3)
                rejected.append(info)
                continue
            mu = cand['bw'] ** 2 / (1 << cand['sf'])
            sf_f = math.log2(bw_final ** 2 / mu)
            if abs(sf_f - round(sf_f)) > 0.2 or not (5 <= round(sf_f) <= 12):
                info['reject'] = 'chirp rate %.3g Hz/s has no LoRa SF at BW %.0f kHz' % (mu, bw_final / 1e3)
                rejected.append(info)
                continue
            cand['bw_final'] = bw_final
            cand['sf_final'] = int(round(sf_f))
            cand['centre'] = band['centre']
            cand['band_confirmed'] = True
        else:
            cand['bw_final'] = cand['bw']
            cand['sf_final'] = cand['sf']
            cand['centre'] = cand['fc']
            cand['band_confirmed'] = False
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
    out['rate_hz_per_s'] = final['bw'] ** 2 / (1 << final['sf'])
    # dechirp confirmation on the winning channel / SF (symbol-level tone)
    if cfg.get('dechirp', True):
        y = ys[final['ci']]
        M = 1 << final['sf']
        dc = css.dechirp_blocks(y, final['sf'])
        if dc is not None:
            nw = len(dc['up'])
            thr = css.dechirp_threshold(M, cfg['p_dechirp'] / max(nw, 1))
            R = dc['up'] if final['dir'] == 'up' else dc['dn']
            hits = int((R > thr).sum())
            run = css.preamble_run(dc if final['dir'] == 'up' else dict(up=dc['dn'], bin_up=dc['bin_dn']), thr, M)
            out['dechirp'] = dict(thr=float(thr), n_windows=int(nw), n_hits=hits, max_R=float(R.max()),
                                  hit_frac=hits / nw, preamble_run=int(run))
    return out


'''
a = a[:i] + new + a[j:]
a = a.replace("from . import pulse as pl", "from . import pulse as pl\nfrom . import gated")
a = a.replace("    dechirp=True,", "    dechirp=True,\n    band_check=True,\n    min_adjacent=True,\n    bw_ratio_min=0.55,")
open('analyze.py', 'w').write(a)
