"""Top level: characterise a capture as chirp-like (CSS / LoRa-like) and / or pulsed (radar-like).

    res = characterise(src, meta=None, cfg=None)

src is a frontend.ArraySource / Ci8FileSource.  meta (optional, what hackrf-scout stores with the capture):
    est_bw_hz     bandwidth estimated by the sweep (100 kHz resolution)
    filter_hz     baseband filter used for the capture
Two streaming passes over the file (pass 1: mean + PSD; pass 2: channeliser bank + power envelope), then small
arrays only.  Output: dict with 'chirp' and 'pulsed' sub-dicts (flag, confidence, hints) and 'label'.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import frontend as fe
from . import css
from . import pulse as pl
from . import gated

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_BWS = (125e3, 250e3, 500e3)
SFS = (5, 6, 7, 8, 9, 10, 11, 12)

DEFAULT_CFG = dict(
    p_capture=1e-3,          # target capture-level false-alarm probability of the chirp test (all tests together)
    L=64,                    # lag-line block (decimated samples) = 8 symbols
    asym=3.0,                # main line must exceed the mirror line by this factor
    p_dechirp=1e-2,          # capture-level false-alarm budget of the dechirp hit count
    dechirp=True,
    band_check=True,
    min_adjacent=True,
    bw_ratio_min=0.55,
    grid_step=0.5,           # centre grid step in units of the candidate BW
    pfa_pulse=1e-8,
    pulse=dict(n_min=8, cv_max=0.35, multiple_min=0.7, duty_max=0.25, jitter_max=0.2, n_cluster_min=5),
    use_psd_bw=True,
)


def load_cfg(path=None):
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    path = path or os.path.join(HERE, 'thresholds.json')
    if os.path.exists(path):
        over = json.load(open(path))
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ----------------------------------------------------------------------------------------------
# null table for the lag-line statistic
# ----------------------------------------------------------------------------------------------
_NULL = {}


def lag_null_threshold(L, p, k=1):
    """Threshold on min(R) over k consecutive blocks (same direction) with null exceedance probability p, from the
    Monte-Carlo table results/null_lagk_<L>.json by log-log interpolation / extrapolation (power-law tail)."""
    if L not in _NULL:
        _NULL[L] = json.load(open(os.path.join(HERE, 'results', 'null_lagk_%d.json' % L)))
    tab = _NULL[L][str(k)]
    ps = np.array(sorted([float(q) for q in tab if q != 'n'], reverse=True))
    ts = np.array([tab['%g' % q] for q in ps])
    keep = ps * tab['n'] >= 30                      # levels with enough Monte-Carlo exceedances
    ps, ts = ps[keep], ts[keep]
    lp, lt = np.log(ps), np.log(ts)
    if p >= ps[0]:
        return float(ts[0])
    if p <= ps[-1]:
        m = min(3, len(ps) - 1)
        slope = (lt[-1] - lt[-1 - m]) / (lp[-1] - lp[-1 - m])
        return float(math.exp(lt[-1] + slope * (math.log(p) - lp[-1])))
    return float(math.exp(np.interp(math.log(p), lp[::-1], lt[::-1])))


def kmin(x, k):
    """Running minimum over k consecutive elements (length n-k+1)."""
    if k <= 1:
        return x
    out = x[:len(x) - k + 1].copy()
    for j in range(1, k):
        out = np.minimum(out, x[j:len(x) - k + 1 + j])
    return out


# ----------------------------------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------------------------------
def candidate_channels(rate, bws, centre0, search_hz, step_rel):
    """List of (bw, fc) channels: for every candidate bandwidth a grid of centres covering
    [centre0 - search, centre0 + search], step = step_rel * bw, limited so the channel fits inside +-rate/2."""
    out = []
    for bw in bws:
        if bw * 2 > rate:
            continue
        step = step_rel * bw
        n = int(math.ceil(search_hz / step))
        for k in range(-n, n + 1):
            fc = centre0 + k * step
            if abs(fc) + bw / 2 <= rate / 2 * 0.98:
                out.append((bw, fc))
    return out


def characterise(src, meta=None, cfg=None, timing=None, want_css=True, want_pulse=True, bws=LORA_BWS, sfs=SFS):
    cfg = cfg or load_cfg()
    meta = meta or {}
    tm = {} if timing is None else timing
    rate = src.rate
    t0 = time.time()
    p1 = fe.pass1(src)
    floor, bands, binfo = fe.find_bands(p1)
    tm['pass1'] = time.time() - t0
    noise_full = floor * rate
    res = dict(rate=rate, n=p1['n'], floor_psd=floor, bands=bands[:3], mean=p1['mean'])

    # ---- search window for the centre ----------------------------------------------------------
    est_bw = meta.get('est_bw_hz')
    if est_bw:
        search = 0.5 * max(est_bw, 100e3) + 100e3
    else:
        search = min(rate / 2 - 0.5 * min(bws), 800e3)

    bw_ok = [b for b in bws if b * 2 <= rate]
    if est_bw:
        bw_ok = [b for b in bw_ok if 0.4 * est_bw <= b <= est_bw + 150e3]
    chans = candidate_channels(rate, bw_ok, 0.0, search, cfg['grid_step']) if want_css and bw_ok else []
    if want_css and not chans:
        want_css = False
    nin = fe.common_nin(rate, sorted({c[0] for c in chans}) or [rate], target=1 << 16) if want_css else (1 << 16)
    chobjs = [fe.Channeliser(rate, fc, bw, nin) for bw, fc in chans]
    outs = [[] for _ in chobjs]
    t1 = time.time()
    mean = np.complex64(p1['mean'])
    pdet = pl.PulseDetector(noise_full, cfg['pfa_pulse']) if want_pulse else None
    if want_pulse or want_css:
        for blk in fe.rechunk(src, nin):
            b = blk - mean
            if want_css:
                X = np.fft.fft(b)
                for i, c in enumerate(chobjs):
                    outs[i].append(c.apply(X))
            if want_pulse:
                pdet.feed((b.real * b.real + b.imag * b.imag).astype(np.float32))
    tm['pass2_channelise'] = time.time() - t1

    # ---- chirp ---------------------------------------------------------------------------------
    if want_css:
        t2 = time.time()
        res['chirp'] = chirp_stage(outs, chans, rate, floor, cfg, p1, bands, src, nin, bws)
        tm['chirp'] = time.time() - t2
    else:
        res['chirp'] = dict(flag=False, why='not run')

    # ---- pulses --------------------------------------------------------------------------------
    if want_pulse:
        t3 = time.time()
        res['pulsed'] = pulse_stage(pdet.finish(), rate, cfg, src=src, mean=p1['mean'], noise=noise_full)
        tm['pulse'] = time.time() - t3
    else:
        res['pulsed'] = dict(flag=False, why='not run')
    tm['total'] = time.time() - t0
    res['timing'] = tm
    res['label'] = label_of(res)
    return res


def _nearest_bw(bw3, bws=LORA_BWS, tol=0.3):
    best = min(bws, key=lambda b: abs(math.log(b / bw3)))
    return best if abs(math.log(best / bw3)) <= math.log(1 + tol) else None


def refine_sf(src, mean, rate, cand, bw, centre, cfg, T):
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
    kadj = cfg.get('k_adj', 2)
    p_block = cfg['p_capture'] / max(n_trials, 1)
    T = lag_null_threshold(L, p_block, kadj)
    passing = []
    mx = 0.0
    for ci, bw, fc, sf, lb in lag:
        up, dn = lb['up'], lb['dn']
        for name, main, opp in (('up', up, dn), ('down', dn, up)):
            mm = np.where(main > cfg['asym'] * opp, main, 0.0)          # mirror-line (asymmetry) test per block
            mk = kmin(mm, kadj)
            if len(mk) == 0:
                continue
            mx = max(mx, float(mk.max()))
            ok = mk > T
            if ok.any():
                k = int(np.argmax(mk))
                hit = np.flatnonzero(ok)
                # blocks covered by the hit k-windows (for the gated spectrum): all blocks i..i+kadj-1
                cov = np.unique(np.concatenate([hit + j for j in range(kadj)]))
                passing.append(dict(ci=ci, bw=bw, fc=fc, sf=sf, dir=name, R=float(mk[k]), score=float(mk[k] / T),
                                    n_hits=int(len(hit)), n_blocks=int(len(main)), block=k, start=int(lb['start'][k]),
                                    span=int(lb['span']), hit_idx=cov, main=main, lb=lb,
                                    adjacent=True))
    out = dict(T=T, n_trials=n_trials, p_block=p_block, max_R=mx, max_score=mx / T, n_pass=len(passing))
    # diagnostic: the best score (max over tests of min-R-over-k-adjacent-blocks / threshold_k) for k = 1, 2, 3
    sc = {}
    for kk in (1, 2, 3):
        Tk = lag_null_threshold(L, p_block, kk)
        best_k = 0.0
        for ci, bw, fc, sf, lb in lag:
            for main, opp in ((lb['up'], lb['dn']), (lb['dn'], lb['up'])):
                mk = kmin(np.where(main > cfg['asym'] * opp, main, 0.0), kk)
                if len(mk):
                    best_k = max(best_k, float(mk.max()))
        sc[kk] = best_k / Tk
    out['k_scores'] = sc
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


def pulse_stage(events, rate, cfg, src=None, mean=0j, noise=None):
    out = dict(n_events=len(events))
    if len(events) < 3:
        out['flag'] = False
        out['why'] = 'fewer than 3 events'
        return out
    t0s = np.array([e['t0'] for e in events]) / rate
    pri = pl.estimate_pri(t0s, rate)
    out['pri'] = pri
    if pri.get('pri'):
        out['completeness'] = pl.completeness(t0s, pri['pri'])
    width_s = None
    regular = pri.get('pri') and pri['frac_multiple'] >= 0.5 and pri.get('n_cluster', 0) >= 5
    if regular and src is not None:
        st = pl.stack_profile(src, mean, events, noise)
        if st is not None:
            width_s = st['width_samples'] / rate
            out['width_s'] = width_s
            out['stack_snr_db'] = 10 * math.log10(max(st['snr'], 1e-9))
            out['stack_n'] = st['n']
    flag, info = pl.decide_pulsed(events, rate, pri, width_s, cfg['pulse'])
    out.update(flag=bool(flag), info=info)
    if flag and src is not None:
        out['chirp_pulse'] = pl.intrapulse_slope(src, mean, events, rate)
    return out


def verdict(res):
    """Turn the raw stage outputs into labels with a confidence tier and 'unknown' when nothing passes.
    chirp tiers   high   : lag-line + occupied band consistent with a LoRa bandwidth + dechirp tone in many windows
                          (preamble run >= 6 or >= 20 hit windows)
                  medium : lag-line + band consistent + >= 3 dechirp windows
                  low    : lag-line only (band not measurable, or no dechirp support)
    pulsed tiers  high   : >= 50 in-train spacings and >= 90 % of the pulses on the PRI grid were found
                  medium : >= 15 spacings          low : otherwise"""
    c, p = res.get('chirp', {}), res.get('pulsed', {})
    out = dict(chirp=None, pulsed=None, chirp_pulse=None)
    if c.get('flag'):
        b = c['best']
        dc = c.get('dechirp') or {}
        conf = 'low'
        if b.get('band_confirmed') and dc:
            if dc.get('preamble_run', 0) >= 6 or dc.get('n_hits', 0) >= 20:
                conf = 'high'
            elif dc.get('n_hits', 0) >= 3:
                conf = 'medium'
        out['chirp'] = dict(confidence=conf, bw_hz=b['bw_final'], sf=b['sf_final'], centre_hz=b['centre'],
                            sym_time_s=c.get('sym_time_s'), direction=b['dir'], band_confirmed=bool(b.get('band_confirmed')))
    if p.get('flag'):
        info = p['info']
        comp = p.get('completeness', 0.0)
        ncl = info.get('n_cluster', 0)
        conf = 'high' if (ncl >= 50 and comp >= 0.9) else ('medium' if ncl >= 15 else 'low')
        out['pulsed'] = dict(confidence=conf, pri_s=info['pri'], width_s=p.get('width_s') if comp >= 0.9 else None,
                             width_valid=bool(comp >= 0.9), duty=info['duty'], n_spacings=ncl, jitter=info['jitter'],
                             completeness=comp)
        cp = p.get('chirp_pulse')
        if cp and abs(cp['t']) > 6 and cp['frac_same_sign'] > 0.8:
            out['chirp_pulse'] = dict(slope_hz_per_s=cp['slope_hz_per_s'], swept_hz=cp['swept_hz'], t=cp['t'])
    return out


def label_of(res):
    v = verdict(res)
    res['verdict'] = v
    parts = []
    if v['chirp']:
        parts.append('chirp-like(%s)' % v['chirp']['confidence'])
    if v['pulsed']:
        parts.append('pulsed(%s)' % v['pulsed']['confidence'])
        if v['chirp_pulse']:
            parts.append('chirp-in-pulse')
    return '+'.join(parts) if parts else 'unknown'
