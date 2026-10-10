"""Spectrum-domain features computed from the streaming result of sp_stream.analyse_stream.

numpy only.  Everything returns plain floats / ints / lists / None (JSON serialisable), None meaning
"not measurable on this capture" (floor not visible, SNR too low for that definition, ...).

Statistical model (verified on white noise, see t0.py): a Welch bin averaged over K_eff independent
segments is Gamma(K_eff)/K_eff distributed; a rebin of g adjacent Hann bins has shape
K_eff*g^2/(g+0.889(g-1)+0.0556(g-2)).  Wilson-Hilferty turns this into a normal score z.
"""
from __future__ import annotations

import math

import numpy as np

FLOOR_TOL = 1.12                       # floor ripple tolerance (+0.5 dB): sloped signal skirts / analog ripple
SCALES = (1, 4, 16, 64, 256)          # detection scales in fine bins (matched-filter scan)


# ----------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------
def shape_factor(g):
    if g <= 1:
        return 1.0
    return g * g / (g + 0.889 * (g - 1) + 0.0556 * max(g - 2, 0))


def gamma_q(a, z):
    """Quantile (at normal score z) of Gamma(a)/a by the Wilson-Hilferty transform."""
    v = 1.0 / (9.0 * a)
    return max(1.0 - v + z * math.sqrt(v), 0.0) ** 3


def zscore(R, a):
    v = 1.0 / (9.0 * a)
    return (np.cbrt(np.maximum(R, 1e-12)) - (1.0 - v)) / math.sqrt(v)


def rebin(x, g):
    m = (len(x) // g) * g
    return x[:m].reshape(-1, g).mean(1)


def _runs(mask):
    d = np.diff(np.concatenate(([0], np.asarray(mask, np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ----------------------------------------------------------------------------------------------
# noise floor
# ----------------------------------------------------------------------------------------------
def estimate_floor(Pu, guard, K, target_bins=512):
    """Robust noise PSD from a lower-tail iterative clip.  Returns (n0, frac_floor, g) or None."""
    n = len(Pu)
    g = max(1, n // target_bins)
    Pc = rebin(Pu, g)
    ok = rebin(guard.astype(np.float64), g) == 0
    if ok.sum() < 16:
        return None
    a = K * shape_factor(g)
    n0 = np.quantile(Pc[ok], 0.05) / gamma_q(a, -1.645)
    sig = 1.0 / math.sqrt(a)
    corr = 1.0 - sig * 0.0175 / 0.9938                  # mean of the Gamma truncated at +2.5 sigma
    sel = ok
    for _ in range(10):
        thr = n0 * max(gamma_q(a, 2.5), FLOOR_TOL)
        sel = ok & (Pc < thr)
        if sel.sum() < 4:
            break
        new = Pc[sel].mean() / corr
        if abs(new - n0) < 1e-4 * n0:
            n0 = new
            break
        n0 = new
    return float(n0), float(sel.sum() / ok.sum()), int(g), int(sel.sum())


# ----------------------------------------------------------------------------------------------
# lobes / lines
# ----------------------------------------------------------------------------------------------
def count_lobes(db, min_prom):
    """Topographic prominence peak count (interior local maxima only)."""
    n = len(db)
    out = []
    for p in range(1, n - 1):
        if not (db[p] >= db[p - 1] and db[p] > db[p + 1]):
            continue
        j = p - 1
        while j >= 0 and db[j] <= db[p]:
            j -= 1
        lmin = db[max(j, 0):p + 1].min()
        k = p + 1
        while k < n and db[k] <= db[p]:
            k += 1
        rmin = db[p:min(k, n - 1) + 1].min()
        # a side with no higher peak uses the lowest point of that whole side
        prom = db[p] - max(lmin, rmin)
        if prom >= min_prom:
            out.append((p, float(prom)))
    return out


def find_lines(Pu, fu, guard, K, z_line=5.0):
    """Narrow spectral lines: local maxima standing out from a running-median baseline and from their
    own neighbours (>= 3 bins away).  The detection threshold uses the LOCAL scatter of P/baseline
    (robust MAD per 256-bin block, never better than the Welch model): gated / bursty signals scatter far
    more than K_eff independent averages would, and would otherwise sprout false lines.
    Returns list of (freq_hz, level_db_over_baseline, index)."""
    n = len(Pu)
    W = 32
    if n < 4 * W:
        return []
    pad = np.pad(Pu, W, mode='edge')
    base = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * W + 1), axis=1)
    base = base / (1.0 - 1.0 / (3.0 * K))                       # median -> mean of Gamma(K)/K
    R = Pu / np.maximum(base, 1e-30)
    nb = 256
    nblk = int(math.ceil(n / nb))
    Rp = np.pad(R, (0, nblk * nb - n), mode='edge').reshape(nblk, nb)
    med = np.median(Rp, axis=1)
    sd = 1.4826 * np.median(np.abs(Rp - med[:, None]), axis=1)
    k_loc = np.clip(1.0 / np.maximum(sd, 1e-6) ** 2, 2.0, K)
    thr = np.repeat(np.array([gamma_q(k, z_line) for k in k_loc]), nb)[:n]
    cand = np.flatnonzero((R > thr) & (R > FLOOR_TOL) & (Pu >= np.roll(Pu, 1)) & (Pu > np.roll(Pu, -1)) & ~guard)
    lines = []
    for i in cand:
        if i < 7 or i > n - 8:
            continue
        nbm = 0.5 * (Pu[i - 6:i - 2].mean() + Pu[i + 3:i + 7].mean())
        if (Pu[i] - base[i]) * 0.35 > (nbm - base[i]):
            lines.append((float(fu[i]), float(10 * math.log10(R[i])), int(i)))
    lines.sort(key=lambda t: t[2])
    merged = []
    for L in lines:
        if merged and L[2] - merged[-1][2] < 4:
            if L[1] > merged[-1][1]:
                merged[-1] = L
        else:
            merged.append(L)
    return merged


# ----------------------------------------------------------------------------------------------
# OFDM cyclic-prefix test from the PSD (Wiener-Khinchin: the capture-averaged autocorrelation IS the
# inverse FFT of the averaged periodogram, so the CP peak at lag N costs one IFFT, no extra pass)
# ----------------------------------------------------------------------------------------------
_RHO_CACHE = {}


def _rho_w(win):
    key = len(win)
    if key not in _RHO_CACHE:
        w = win.astype(np.float64)
        F = np.fft.fft(w, 2 * len(w))
        r = np.fft.ifft(np.abs(F) ** 2).real[:len(w)]
        _RHO_CACHE[key] = r / r[0]
    return _RHO_CACHE[key]


def cp_test(sp, band, tau_hi=None):
    P, f, rate = sp['P'], sp['f'], sp['rate']
    nfft = len(P)
    lo, hi = band
    B = hi - lo
    dl = 0.05 * B                                        # raised-cosine edges: a brick-wall mask rings in the autocorrelation
    t_lo = 0.5 * (1 - np.cos(np.pi * np.clip((f - (lo - dl)) / (2 * dl), 0, 1)))
    t_hi = 0.5 * (1 - np.cos(np.pi * np.clip(((hi + dl) - f) / (2 * dl), 0, 1)))
    Pm = P * t_lo * t_hi
    r = np.fft.ifft(np.fft.ifftshift(Pm)) * rate
    r0 = float(r[0].real)
    if r0 <= 0 or B <= 0:
        return None
    rho = _rho_w(sp['win'])
    tau_pk = tau_hi or nfft // 8                 # peak search range (Hann lag window >= 0.90)
    tau_hi = nfft // 4                           # comb range (Hann lag window >= 0.66)
    tau_lo = int(max(40, math.ceil(6.0 * rate / B)))
    if tau_lo >= tau_pk - 200:
        return None
    lags = np.arange(tau_lo, tau_hi)
    A = np.abs(r[lags]) / np.maximum(rho[lags], 0.5) / r0
    Wm = 64
    pad = np.pad(A, Wm, mode='edge')
    base = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * Wm + 1), axis=1)
    p = A - base
    mad = 1.4826 * np.median(np.abs(p - np.median(p)))
    sig_th = 1.0 / math.sqrt(max(sp['n'] * B / rate, 1.0))
    sigma = max(mad, sig_th)
    i = int(np.argmax(p[:tau_pk - tau_lo]))
    lag = int(lags[i])
    z = float(p[i] / sigma)
    # comb test: is there a comparable peak at 2*lag ?
    comb = None
    if 2 * lag + 4 < tau_hi:
        j0 = 2 * lag - 3 - tau_lo
        j1 = 2 * lag + 4 - tau_lo
        comb = float(max(p[j0:j1].max(), 0.0) / max(p[i], 1e-12))
    npk = int(np.sum(p > max(0.3 * p[i], 6 * sigma)))
    return dict(cp_z=z, cp_A=float(p[i]), cp_lag=lag, cp_lag_B=float(lag * B / rate), cp_comb=comb,
                cp_sigma=float(sigma), cp_npk=npk, cp_band_hz=float(B))


# ----------------------------------------------------------------------------------------------
# time-frequency (hopping) test
# ----------------------------------------------------------------------------------------------
def hop_test(sp, n0, usable_frac, z_act=4.5, W=None, min_seg=2):
    S, fg, dt, frame, G = sp['S'], sp['tf_f'], sp['tf_dt'], sp['tf_frame'], sp['tf_ng']
    rate = sp['rate']
    nt = S.shape[0]
    if nt < 50:
        return None
    use = np.abs(fg) <= usable_frac * rate / 2
    Su, fu = S[:, use].astype(np.float64), fg[use]
    Gu = Su.shape[1]
    gw = rate / G
    W = W or max(3, int(round(0.10 * G)))
    if Gu < W + 3:
        return None
    cs = np.concatenate((np.zeros((nt, 1)), np.cumsum(Su, axis=1)), axis=1)
    sm = (cs[:, W:] - cs[:, :-W]) / W
    R = sm / n0
    best = R.max(1)
    arg = R.argmax(1)
    a_f = shape_factor(W * (frame // G))
    thr = gamma_q(a_f, z_act)
    act = best > thr
    n_act = int(act.sum())
    out = dict(hop_dt=float(dt), hop_act_frac=float(n_act / nt), hop_n_act=n_act, hop_thr=float(thr), hop_win_hz=float(W * gw))
    if n_act < 8:
        out.update(hop_n_ch=0, hop_n_seg=0, hop_n_hops=0, hop_dwell_s=None, hop_conc=None, hop_dwell_kind=None)
        return out
    # centroid of the excess power in a window of +-W/2 around the strongest window
    wgt = np.maximum(Su - n0, 0.0)
    c0 = np.concatenate((np.zeros((nt, 1)), np.cumsum(wgt, axis=1)), axis=1)
    c1 = np.concatenate((np.zeros((nt, 1)), np.cumsum(wgt * fu[None, :], axis=1)), axis=1)
    ta = np.flatnonzero(act)
    lo = np.maximum(arg[ta] - W // 2, 0)
    hi = np.minimum(arg[ta] + W + W // 2, Gu)
    num = c1[ta, hi] - c1[ta, lo]
    den = c0[ta, hi] - c0[ta, lo]
    cen = np.where(den > 0, num / np.maximum(den, 1e-30), fu[np.minimum(arg[ta] + W // 2, Gu - 1)])
    # concurrency: a second, disjoint window that is also active AND comparable (>= 25 % of the excess of the
    # strongest window; skirts of a single strong channel are 15-30 dB down and must not count)
    idx = np.arange(sm.shape[1])[None, :]
    Ra = R[ta]
    excl = np.abs(idx - arg[ta][:, None]) <= W
    r2 = np.where(excl, 0.0, Ra).max(1)
    conc_frac = float(np.mean((r2 > thr) & ((r2 - 1.0) >= 0.25 * (best[ta] - 1.0))))
    # segmentation into stable-channel dwells
    tol = 0.5 * W * gw
    segs = []
    cur = None
    for t, c in zip(ta, cen):
        if cur is not None and t - cur[1] <= 2 and abs(c - cur[2] / cur[3]) <= tol:
            cur[1] = t
            cur[2] += c
            cur[3] += 1
        else:
            if cur is not None:
                segs.append(cur)
            cur = [t, t, c, 1]
    if cur is not None:
        segs.append(cur)
    segs = [s for s in segs if s[3] >= min_seg]
    nseg = len(segs)
    out.update(hop_conc=float(conc_frac), hop_n_seg=nseg)
    if nseg < 2:
        out.update(hop_n_ch=min(nseg, 1), hop_n_hops=0, hop_dwell_s=None, hop_dwell_kind=None)
        return out
    cents = np.array([s[2] / s[3] for s in segs])
    order = np.argsort(cents)
    lab = np.zeros(nseg, int)
    k = 0
    for ii in range(1, nseg):
        if cents[order[ii]] - cents[order[ii - 1]] > tol:
            k += 1
        lab[order[ii]] = k
    counts = np.bincount(lab)
    n_ch = int((counts >= 2).sum())
    hops = 0
    d = []
    for i in range(nseg - 1):
        if lab[i] != lab[i + 1]:
            hops += 1
            if segs[i + 1][0] - segs[i][1] <= 3:
                d.append(segs[i + 1][0] - segs[i][0])
    dwell = None
    kind = None
    if len(d) >= 3:
        d = np.array(d, float)
        med = np.median(d)
        good = d[np.abs(d - med) <= 0.4 * med]
        dwell = float(good.mean() * dt) if len(good) else float(med * dt)
        kind = 'period'
    elif nseg:
        dwell = float(np.median([s[3] for s in segs]) * dt)
        kind = 'seglen'
    out.update(hop_n_ch=n_ch, hop_n_hops=hops, hop_dwell_s=dwell, hop_dwell_kind=kind,
               hop_ch_hz=[float(cents[order[lab[order] == q][0]]) for q in range(k + 1) if counts[q] >= 2][:12])
    return out



def occupied_bw(Pu, fu, n0, K, df, lo0, hi0, z_b=3.0, iters=4, gmask=None):
    """99 % occupied bandwidth (0.5 % power in each tail) on the noise-subtracted PSD.

    Start from the span of the multi-scale detection mask (over-wide for narrow signals because a coarse
    detection window smears a line), then iterate: rebin to ~64 bins across the current span estimate, keep
    only bins significant at z_b (so noise-only bins add nothing), take the 0.5 % / 99.5 % power quantiles,
    and shrink/grow the span.  Returns None if no power survives."""
    B0 = max(hi0 - lo0, 4 * df)
    centre = 0.5 * (lo0 + hi0)
    res = None
    for _ in range(iters):
        g = max(1, int(B0 / df / 64))
        sl = (fu >= centre - 1.5 * B0) & (fu <= centre + 1.5 * B0)
        if sl.sum() < 3 * g:
            break
        Pr, fr = rebin(Pu[sl], g), rebin(fu[sl], g)
        a = K * shape_factor(g)
        z = zscore(Pr / n0, a)
        keep = z > z_b
        if gmask is not None:                      # optional DC guard also applies to the bandwidth sums
            keep &= ~(rebin(gmask[sl].astype(np.float64), g) > 0)
        # an isolated single bin that is only just significant is a noise excursion, not signal: it would
        # decide the 0.5 % tail quantile of a narrow low-SNR signal
        st_, en_ = _runs(keep)
        for a_, b_ in zip(st_, en_):
            if b_ - a_ == 1 and z[a_] < 5.0:
                keep[a_] = False
        w = np.where(keep, np.maximum(Pr - n0, 0.0), 0.0)
        tot = w.sum()
        if tot <= 0:
            break
        cum = np.cumsum(w) / tot
        dfc = df * g

        def q(p):
            j = min(int(np.searchsorted(cum, p)), len(cum) - 1)
            prev = cum[j - 1] if j > 0 else 0.0
            frac = (p - prev) / max(cum[j] - prev, 1e-12)
            return float(fr[j] - dfc / 2 + frac * dfc)
        flo, fhi = q(0.005), q(0.995)
        obw = max(fhi - flo, dfc)
        mk = keep | np.roll(keep, 1) | np.roll(keep, -1)
        res = dict(flo=flo, fhi=fhi, obw=obw, Pr=Pr, fr=fr, dfc=dfc, mk=mk, g=g,
                   feat=dict(obw99_hz=float(obw), obw_lo_hz=flo, obw_hi_hz=fhi, obw_centre_hz=0.5 * (flo + fhi),
                             obw_bins=float(obw / dfc), obw_res_hz=float(dfc),
                             sig_power_over_n0df=float(tot * dfc / (n0 * df)),
                             snr_est_db=float(10 * math.log10(max(tot * dfc / (n0 * obw), 1e-12)))))
        centre = 0.5 * (flo + fhi)
        B0 = max(1.3 * obw, 6 * g * df)
    return res

# ----------------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------------
def features(sp, usable_frac=0.9, dc_guard_bins=2, z_det=5.5, hop=True, guard_obw=False):
    f, P, rate, K = sp['f'], sp['P'], sp['rate'], sp['k_eff']
    df = float(f[1] - f[0])
    feat = dict(rate=rate, n=sp['n'], k_eff=K, df=df, dc_re=sp['dc'].real, dc_im=sp['dc'].imag,
                dc_rel_db=float(10 * math.log10(max(abs(sp['dc']) ** 2, 1e-12) / sp['p_raw'])))
    use = np.abs(f) <= usable_frac * rate / 2
    idx = np.flatnonzero(use)
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    Pu, fu = P[i0:i1], f[i0:i1]
    n = len(Pu)
    guard = np.abs(fu) <= dc_guard_bins * df
    fl = estimate_floor(Pu, guard, K)
    if fl is None:
        feat['floor_ok'] = False
        return feat
    n0, frac_floor, g_fl, nsel = fl
    feat.update(n0=n0, floor_frac=frac_floor, floor_g=g_fl, floor_nsel=nsel,
                floor_ok=bool(frac_floor >= 0.05 and nsel >= 12))

    # ---- multi-scale excess scan (also the noise-only / structure test) ----
    mask = np.zeros(n, bool)
    zs = {}
    for g in SCALES:
        if n // g < 16:
            continue
        m = (n // g) * g
        Pg = Pu[:m].reshape(-1, g).mean(1)
        a = K * shape_factor(g)
        Rg = Pg / n0
        z = zscore(Rg, a)
        z = np.where(Rg > FLOOR_TOL, z, np.minimum(z, 0.0))      # an excess below the floor-ripple tolerance is not evidence
        gg = rebin(guard.astype(np.float64), g) > 0
        z[gg] = -50.0
        zs[g] = float(z.max())
        hit = z > z_det
        if g >= 4 and hit.any():
            hit = hit | np.roll(hit, 1) | np.roll(hit, -1)
        mask[:m] |= np.repeat(hit, g)
    feat['struct_z'] = max(zs.values()) if zs else None
    feat['struct_z_by_scale'] = zs
    feat['struct_scale'] = max(zs, key=zs.get) if zs else None
    # DC residual after mean removal: worst z in +-3 bins around DC
    dcm = np.abs(f) <= 3 * df
    if dcm.any():
        feat['dc_resid_z'] = float(zscore(P[dcm] / n0, K).max())
    feat['occ_frac'] = float(mask.sum() / n)
    # ---- clusters (gap-merged) ----
    if not mask.any():
        feat['detected'] = False
    else:
        gapmin = max(4, int(0.02 * n))
        st, en = _runs(mask)
        cl = [[st[0], en[0]]]
        for s_, e_ in zip(st[1:], en[1:]):
            if s_ - cl[-1][1] <= gapmin:
                cl[-1][1] = e_
            else:
                cl.append([s_, e_])
        feat['detected'] = True
        feat['n_clusters'] = len(cl)
        lo_i, hi_i = cl[0][0], cl[-1][1]
        ob = occupied_bw(Pu, fu, n0, K, df, fu[lo_i], fu[min(hi_i, n) - 1], gmask=guard if guard_obw else None)
        if ob is not None:
            feat.update(ob['feat'])
            flo, fhi, obw = ob['flo'], ob['fhi'], ob['obw']
            Pr, fr, dfc, mk = ob['Pr'], ob['fr'], ob['dfc'], ob['mk']
            Ps = np.convolve(Pr - n0, np.ones(3) / 3.0, 'same')
            peak = Ps[mk].max() if mk.any() else Ps.max()
            ar = K * shape_factor(3 * ob['g'])
            floor_ex = n0 * max(gamma_q(ar, 4.0) - 1.0, FLOOR_TOL - 1.0)
            feat['peak_snr_db'] = float(10 * math.log10(max(peak, 1e-30) / n0))
            for x in (3, 10, 20):
                lev = peak * 10 ** (-x / 10.0)
                if lev > floor_ex and peak > 0:
                    ii = np.flatnonzero(Ps >= lev)
                    feat['bw_%ddb_hz' % x] = float((ii[-1] - ii[0] + 1) * dfc)
                else:
                    feat['bw_%ddb_hz' % x] = None
            w3, w10, w20 = feat.get('bw_3db_hz'), feat.get('bw_10db_hz'), feat.get('bw_20db_hz')
            feat['edge_ratio'] = float(w10 / w3) if (w3 and w10) else None
            feat['edge_ratio20'] = float(w20 / w3) if (w3 and w20) else None
            # flatness / peak-to-mean over the 99 % span, ~32 bins across it
            g2 = max(1, int(round(obw / df / 32)))
            sel = (fu >= flo) & (fu <= fhi)
            Pq = rebin(Pu[sel], g2) if sel.sum() >= g2 else np.zeros(0)
            if len(Pq) >= 8:
                am = Pq.mean()
                gm = math.exp(np.mean(np.log(np.maximum(Pq, 1e-30))))
                feat['flat_db'] = float(10 * math.log10(gm / am))
                feat['ptm_db'] = float(10 * math.log10(Pq.max() / am))
                m0 = int(0.2 * len(Pq))
                inner = Pq[m0:len(Pq) - m0]
                feat['ripple_db'] = float(np.std(10 * np.log10(np.maximum(inner, 1e-30))))
                sm3 = np.convolve(Pq, np.ones(3) / 3.0, 'valid')
                dbq = 10 * np.log10(np.maximum(sm3, 1e-30))
                feat['n_lobes'] = len(count_lobes(dbq, 3.0))
                feat['n_lobes6'] = len(count_lobes(dbq, 6.0))
            else:
                feat.update(flat_db=None, ptm_db=None, ripple_db=None, n_lobes=None, n_lobes6=None)
    # ---- lines ----
    lines = find_lines(Pu, fu, guard, K)
    feat['n_lines'] = len(lines)
    feat['lines'] = [(round(l[0], 1), round(l[1], 1)) for l in sorted(lines, key=lambda t: -t[1])[:6]]
    if lines and feat.get('detected') and feat.get('obw99_hz'):
        feat['line_top_db'] = max(l[1] for l in lines)
    # ---- OFDM CP test ----
    if feat.get('obw99_hz'):
        b = (feat['obw_lo_hz'] - 0.05 * feat['obw99_hz'], feat['obw_hi_hz'] + 0.05 * feat['obw99_hz'])
        feat['cp_src'] = 'span'
    else:
        b = (-usable_frac * rate / 2, usable_frac * rate / 2)
        feat['cp_src'] = 'usable'
    cp = cp_test(sp, b)
    if cp:
        feat.update(cp)
    cp2 = cp_test(sp, (-usable_frac * rate / 2, usable_frac * rate / 2))
    if cp2:
        feat.update({k + '_full': v for k, v in cp2.items() if k in ('cp_z', 'cp_A', 'cp_lag', 'cp_comb')})
    # ---- hopping ----
    if hop and 'S' in sp:
        h = hop_test(sp, n0, usable_frac)
        if h:
            feat.update(h)
    return feat
