"""Feature extraction for the symbol-rate / modulation-family hint.

analyze(source, rate) -> dict of features (all JSON-serialisable floats/ints/bools/short lists).
No classification decisions are made here (see classify.py); the only gates are "is there a band at
all" and "is the band narrow enough to be analysed".
"""
from __future__ import annotations

import math
import time

import numpy as np

import frontend as fe
from common import movmean, movsum_complex, runs, otsu, parabolic_peak

MAX_SEG = 400            # cap on Welch segments per spectral feature (bounds runtime on long captures)
MAX_ANALYSIS = 1 << 25   # cap on decimated samples that are kept in memory
MAX_FEAT = 1 << 21       # cap on samples used by the per-feature analyses


# ----------------------------------------------------------------------------------------------
# activity mask on the channelised record
# ----------------------------------------------------------------------------------------------
def frame_mask(p, fs, neb, n_ch, K=24, nsig=5.0, min_len=2):
    """Boolean 'signal present' mask from a K-independent-sample moving average of |y|^2.

    Short active runs (< min_len windows) are dropped and short gaps (< min_len windows) are filled.
    Returns (mask, Lp)."""
    Lp = int(math.ceil(K * fs / neb))
    a = movmean(p, Lp)
    # Schmitt trigger: start above nsig sigma, stay on above 2 sigma -> a weak but steady signal is not chopped up
    m = schmitt(a, n_ch * (1.0 + 2.0 / math.sqrt(K)), n_ch * (1.0 + nsig / math.sqrt(K)))
    for val in (True, False):
        st, ln, v = runs(m)
        for s, l, x in zip(st, ln, v):
            if x == val and l < min_len * Lp:
                if val is True or (s > 0 and s + l < len(m)):
                    m[s:s + l] = not val
    return m, Lp


def erode(mask, k):
    """Erode a boolean mask by k samples on each side (run-length based, O(runs))."""
    out = np.zeros_like(mask)
    st, ln, v = runs(mask)
    for s, l, x in zip(st, ln, v):
        if x and l > 2 * k:
            out[s + k:s + l - k] = True
    return out


# ----------------------------------------------------------------------------------------------
# spectral line helpers
# ----------------------------------------------------------------------------------------------
def seg_spectrum(u, fs, nfft, mask=None, max_seg=MAX_SEG, zpad=2):
    """Averaged periodogram of u over 50 %-overlapped Hann segments lying fully inside mask.

    Segments are taken evenly across the usable ones when there are more than max_seg."""
    n = len(u)
    if n < nfft:
        return None
    hop = nfft // 2
    idx0 = np.arange((n - nfft) // hop + 1) * hop
    if mask is not None:
        c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        idx0 = idx0[(c[idx0 + nfft] - c[idx0]) == nfft]
    if len(idx0) == 0:
        return None
    if len(idx0) > max_seg:
        idx0 = idx0[np.linspace(0, len(idx0) - 1, max_seg).astype(int)]
    w = np.hanning(nfft).astype(np.float32)
    cplx = np.iscomplexobj(u)
    acc = None
    step = max(1, (1 << 20) // nfft)
    for s in range(0, len(idx0), step):
        ii = idx0[s:s + step]
        seg = u[ii[:, None] + np.arange(nfft)[None, :]]
        seg = (seg - seg.mean(axis=1, keepdims=True)) * w
        F = np.fft.fft(seg, n=nfft * zpad, axis=1) if cplx else np.fft.rfft(seg, n=nfft * zpad, axis=1)
        P = (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        acc = P if acc is None else acc + P
    S = acc / len(idx0)
    f = np.fft.fftfreq(nfft * zpad, 1.0 / fs) if cplx else np.fft.rfftfreq(nfft * zpad, 1.0 / fs)
    return dict(f=f, S=S, nseg=len(idx0), df=fs / nfft, zpad=zpad, nfft=nfft)


def _floor(S, hw):
    """Running-median floor (block medians interpolated) over +-hw bins."""
    n = len(S)
    blk = max(hw, 8)
    nb = int(math.ceil(n / blk))
    meds = np.empty(nb)
    for i in range(nb):
        meds[i] = np.median(S[max(0, i * blk - hw):min(n, (i + 1) * blk + hw)])
    cen = (np.arange(nb) + 0.5) * blk
    return np.interp(np.arange(n), cen, meds)


def find_lines(sp, fmin, fmax, nlines=4, hw_bins=80):
    """Strongest narrow lines of a segment spectrum within [fmin, fmax].

    ratio = peak / local floor (floor = running median / (1 - 1/(3 nseg)) -> mean-equivalent).
    z     = (ratio - 1) * sqrt(nseg)   (noise-only bins have ratio ~ 1 +- 1/sqrt(nseg)).
    Returns list of dict(f, ratio, z) sorted by ratio."""
    if sp is None:
        return []
    f, S, nseg, zp = sp['f'], sp['S'], sp['nseg'], sp['zpad']
    fl = _floor(S, hw_bins * zp)
    corr = 0.6931 if nseg == 1 else (1.0 - 1.0 / (3.0 * nseg))
    fl = fl / corr
    ratio = S / np.maximum(fl, 1e-300)
    sel = (f >= fmin) & (f <= fmax)
    out = []
    r = np.where(sel, ratio, 0.0)
    for _ in range(nlines):
        k = int(np.argmax(r))
        if r[k] <= 0:
            break
        kp = parabolic_peak(S, k)
        fpk = float(np.interp(kp, np.arange(len(f)), f))
        out.append(dict(f=fpk, ratio=float(ratio[k]), z=float((ratio[k] - 1.0) * math.sqrt(nseg))))
        lo, hi = max(0, k - 3 * zp), k + 3 * zp + 1
        r[lo:hi] = 0.0
    return out


def _ratio_curve(sp, hw_bins=80):
    f, S, nseg, zp = sp['f'], sp['S'], sp['nseg'], sp['zpad']
    fl = _floor(S, hw_bins * zp)
    corr = 0.6931 if nseg == 1 else (1.0 - 1.0 / (3.0 * nseg))
    return f, S, S / np.maximum(fl / corr, 1e-300), nseg


def find_fundamental(sp, fmin, fmax, nharm=4, min_r=2.0, hw_bins=80):
    """Fundamental of a harmonic family of narrow lines (lowest frequency c such that c, 2c, 3c, ... are lines).

    Candidates are (strong peak)/k for k = 1..4; the score of a candidate is the sum of log line ratios
    at its first harmonics inside [fmin, fmax*1.3]; the fundamental itself must be present (ratio >= min_r).
    Returns dict(f0, ratio, z, score, harm=[ratios], n_harm) or None."""
    if sp is None:
        return None
    f, S, r, nseg = _ratio_curve(sp, hw_bins)
    zp = sp['zpad']
    df = f[1] - f[0]
    sel = (f >= fmin) & (f <= fmax)
    if not sel.any():
        return None
    rr = np.where(sel, r, 0.0)
    peaks = []
    for _ in range(6):
        i = int(np.argmax(rr))
        if rr[i] < 3.0:
            break
        peaks.append(i)
        rr[max(0, i - 4 * zp):i + 4 * zp + 1] = 0.0
    if not peaks:
        return None
    hi_lim = min(fmax * 1.3, f[-1])
    rmax = lambda fq: float(r[max(0, int((fq * 0.985) / df) - 1):int((fq * 1.015) / df) + 2].max()) if fq < hi_lim else 0.0

    def local_peak(fq):
        a, b = max(0, int((fq * 0.985) / df) - 1), int((fq * 1.015) / df) + 2
        k0 = a + int(np.argmax(S[a:b]))
        return float(np.interp(parabolic_peak(S, k0), np.arange(len(f)), f))

    # candidates: the strongest line itself and its sub-multiples f/2, f/3, f/4 (a family that does not
    # contain the strongest line cannot be the clock of this signal)
    cands = {}
    i = peaks[0]
    for k in range(1, 5):
        c = f[i] / k
        if c < fmin:
            continue
        key = int(round(math.log(c) / 0.01))
        cands[key] = c
    best = None
    r_top = float(r[peaks[0]])
    for c in cands.values():
        M = min(nharm, int(hi_lim / c))
        if M < 1:
            continue
        rs = [rmax(m * c) for m in range(1, M + 1)]
        # the fundamental must be present, and a sub-multiple of the strongest line must itself be a clear line
        if rs[0] < min_r or (c < f[peaks[0]] * 0.99 and rs[0] < max(3.0, 0.35 * r_top)):
            continue
        score = sum(math.log(max(x, 1.0)) for x in rs)
        # a candidate whose m=2 (and m=1) harmonics are present but explains less than the true fundamental is penalised
        if best is None or score > best[0] + 1e-9:
            best = (score, c, rs)
    if best is None:
        return None
    score, c, rs = best
    # refine: weighted LS of harmonic peak frequencies
    num = den = 0.0
    for m, x in enumerate(rs, start=1):
        if x >= 3.0:
            fm = local_peak(m * c)
            w = x
            num += w * m * fm
            den += w * m * m
    f0 = num / den if den > 0 else c
    return dict(f0=float(f0), ratio=float(rs[0]), z=float((rs[0] - 1.0) * math.sqrt(nseg)), score=float(score),
                harm=[float(x) for x in rs], n_harm=int(sum(1 for x in rs if x >= 3.0)), df=float(df * 1.0))


# ----------------------------------------------------------------------------------------------
# edge-interval (run-length) analysis of a two-level series
# ----------------------------------------------------------------------------------------------
def schmitt(a, th_lo, th_hi):
    st = np.full(len(a), -1, np.int8)
    st[a > th_hi] = 1
    st[a < th_lo] = 0
    idx = np.where(st >= 0, np.arange(len(a)), 0)
    idx = np.maximum.accumulate(idx)
    s = st[idx]
    s[s < 0] = 0
    return s > 0


def estimate_unit(d, tmin):
    """Smallest-cluster / integer-grid estimate of the unit interval T from run durations d [s].

    Returns dict(T, cluster, rms, support, n_used, n_all) or None."""
    d = d[d >= tmin]
    if len(d) < 20:
        return None
    lg = np.log10(d)
    edges = np.arange(lg.min() - 0.05, lg.max() + 0.05, 1 / 40.0)
    if len(edges) < 5:
        return None
    h, e = np.histogram(lg, bins=edges)
    hs = np.convolve(h, np.array([1, 2, 3, 2, 1]) / 9.0, mode='same')
    c = None
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] > 0.04 * hs.max():
            c = 10 ** ((e[i] + e[i + 1]) / 2)
            break
    if c is None:
        return None
    T = c
    for _ in range(6):
        sel = (d > 0.6 * T) & (d < 12 * T)
        if sel.sum() < 10:
            return None
        n = np.maximum(np.round(d[sel] / T), 1)
        T = float((n * d[sel]).sum() / (n * n).sum())
    sel = (d > 0.6 * T) & (d < 12 * T)
    n = np.maximum(np.round(d[sel] / T), 1)
    res = (d[sel] - n * T) / T
    # share of unit-length runs (n == 1): a true 'minimum pulse' should be common
    return dict(T=T, cluster=c, rms=float(np.sqrt((res ** 2).mean())), support=float((np.abs(res) < 0.2).mean()),
                n_used=int(sel.sum()), n_all=int(len(d)), frac_n1=float((n == 1).mean()))


def subsample_edges(a, edges, rising, th_mid, window):
    """Fractional-sample times of the mid-level crossings belonging to the Schmitt edges.

    For every Schmitt edge (sample index, polarity) take the last crossing of th_mid with the same polarity
    in [edge - window, edge] and linearly interpolate it; edges without a crossing keep their sample index."""
    x = a - th_mid
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    dn = np.flatnonzero((x[:-1] >= 0) & (x[1:] < 0))
    t = edges.astype(np.float64).copy()
    for pol, cr in ((True, up), (False, dn)):
        sel = np.flatnonzero(rising == pol)
        if len(sel) == 0 or len(cr) == 0:
            continue
        e = edges[sel]
        pos = np.searchsorted(cr, e, side='right') - 1          # last crossing at or before the edge
        ok = pos >= 0
        pos = np.where(ok, pos, 0)
        j = cr[pos]
        ok &= (e - j) <= window
        j = np.where(ok, j, 0)
        den = a[j + 1] - a[j]
        frac = np.where(np.abs(den) > 0, (th_mid - a[j]) / np.where(den == 0, 1, den), 0.0)
        tt = j + np.clip(frac, 0.0, 1.0)
        t[sel] = np.where(ok, tt, t[sel])
    return t


def refine_unit_samepol(edges, rising_flag, fs, valid, T0, seg_id=None):
    """Refine the unit interval with intervals between edges of the SAME polarity only.

    Rising-to-rising and falling-to-falling intervals are exact multiples of T whatever the threshold
    asymmetry, so the high/low-run bias of the combined estimate cancels.  Returns (T, n_intervals) or None."""
    ds = []
    for pol in (True, False):
        e = edges[(rising_flag == pol)]
        v = valid[(rising_flag == pol)]
        if len(e) < 3:
            continue
        dd = np.diff(e) / fs
        ok = v[:-1] & v[1:]
        if seg_id is not None:
            sg = seg_id[(rising_flag == pol)]
            ok &= sg[:-1] == sg[1:]
        dd = dd[ok]
        dd = dd[(dd > 1.4 * T0) & (dd < 14 * T0)]
        ds.append(dd)
    if not ds:
        return None
    d = np.concatenate(ds)
    if len(d) < 10:
        return None
    T = T0
    for _ in range(6):
        n = np.maximum(np.round(d / T), 2)
        T = float((n * d).sum() / (n * n).sum())
    return T, int(len(d))


def ook_edge_analysis(p, fs, neb, n_ch, fmask, Lp, brk=None, Kc=6.0):
    """Edge-interval analysis of the envelope inside the frames.  p = |y|^2 (float64)."""
    Lc = max(1, int(round(Kc * fs / neb)))
    a = movmean(p, Lc)
    fmd = movmean(fmask.astype(np.float64), 2 * Lp) > 0
    ax = a[fmd]
    if len(ax) < 20 * Lc:
        return None
    sub = ax[::max(1, len(ax) // 200000)]
    thr_o, sep = otsu(sub, 128)
    hi_v = ax[ax >= thr_o]
    lo_v = ax[ax < thr_o]
    if len(hi_v) == 0 or len(lo_v) == 0:
        return None
    P_hi = float(np.median(hi_v))
    P_lo = float(np.percentile(ax, 10))
    if P_hi < 1.5 * n_ch:
        return None
    th_h = n_ch + 0.6 * (P_hi - n_ch)
    th_l = n_ch + 0.3 * (P_hi - n_ch)
    s = schmitt(a, th_l, th_h)
    st, ln, val = runs(s)
    edges = st[1:]
    th_mid = n_ch + 0.45 * (P_hi - n_ch)
    t_edges = subsample_edges(a, edges, val[1:], th_mid, 3 * Lc) if len(edges) else edges.astype(np.float64)
    out = dict(P_hi=P_hi, P_lo=P_lo, floor_ratio=float(P_lo / n_ch), sep=sep, duty_otsu=float(len(hi_v) / len(ax)),
               duty_hi=float(s[fmd].mean()), n_edges=int(len(edges)),
               lo_frac=float(len(lo_v) / len(ax)), lo_level=float(lo_v.mean() / n_ch), hi_level=float(hi_v.mean() / n_ch))
    if len(edges) < 8:
        return out
    seg_id = np.searchsorted(brk, edges, side='right') if brk is not None and len(brk) else np.zeros(len(edges), np.int64)
    d = np.diff(t_edges) / fs
    ins = fmd[edges[:-1]] & fmd[edges[1:]] & (seg_id[:-1] == seg_id[1:])
    d = d[ins]
    out['n_dur'] = int(len(d))
    est = estimate_unit(d, tmin=Kc / neb)
    if est:
        out.update(est)
        out['T_comb'] = est['T']
        # same-polarity refinement (edges are st[1:] with polarity val[1:] = new state)
        rf = refine_unit_samepol(t_edges, val[1:], fs, fmd[edges], est['T'], seg_id)
        if rf:
            out['T'] = rf[0]
            out['n_same'] = rf[1]
        # share of runs that are 1, 2, 3, >=4 unit intervals long (coding hint: PWM / Manchester vs NRZ)
        n = np.round(d / est['T'])
        n = n[(n >= 1) & (n <= 12)]
        if len(n):
            out['run_hist'] = [float((n == k).mean()) for k in (1, 2, 3)] + [float((n >= 4).mean())]
    # autocorrelation of the (smoothed) squared envelope inside frames: width of the main lobe
    try:
        ac = envelope_acf(a, fmd, fs)
        if ac:
            out['acf'] = ac
    except Exception:
        pass
    return out


def envelope_acf(a, fmd, fs, nfft=1 << 10):
    """Normalised autocorrelation of the envelope power over frame segments (FFT, averaged).

    Returns dict(width_s: lag where the lobe falls to 0.1, first_min_s, first_peak_s)."""
    spx = seg_spectrum(a.astype(np.float32), fs, nfft, mask=fmd, max_seg=200, zpad=1)
    if spx is None:
        return None
    S = spx['S']
    r = np.fft.irfft(S)[:nfft // 2]
    r = r / max(r[0], 1e-30)
    below = np.flatnonzero(r < 0.1)
    width = float(below[0] / fs) if len(below) else None
    # first local minimum then next local maximum
    dr = np.diff(r)
    mins = np.flatnonzero((dr[:-1] < 0) & (dr[1:] >= 0)) + 1
    out = dict(width_s=width)
    if len(mins):
        m0 = int(mins[0])
        out['first_min_s'] = float(m0 / fs)
        maxs = np.flatnonzero((dr[:-1] > 0) & (dr[1:] <= 0)) + 1
        maxs = maxs[maxs > m0]
        if len(maxs):
            out['first_peak_s'] = float(maxs[0] / fs)
            out['first_peak_val'] = float(r[maxs[0]])
    return out


def compact(rec, fmask, Lp, maxn):
    """Concatenate the active frames (each with 2*Lp quiet samples either side) until maxn samples."""
    st, ln, v = runs(fmask)
    iv = []
    tot = 0
    for s0, l, x in zip(st, ln, v):
        if not x:
            continue
        a = max(0, s0 - 2 * Lp)
        b = min(len(rec), s0 + l + 2 * Lp)
        if iv and a <= iv[-1][1]:
            tot += b - iv[-1][1]
            iv[-1] = (iv[-1][0], b)
        else:
            iv.append((a, b))
            tot += b - a
        if tot >= maxn:
            break
    if not iv:
        return rec[:maxn], fmask[:maxn], np.zeros(0, np.int64)
    r = np.concatenate([rec[a:b] for a, b in iv])[:maxn]
    m = np.concatenate([fmask[a:b] for a, b in iv])[:maxn]
    brk = np.cumsum([b - a for a, b in iv])[:-1]       # indices where a new, non-adjacent piece starts
    return r, m, brk[brk < len(r)]


# ----------------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------------
def analyze(source, rate, timings=None, search_hz=None):
    t0 = time.time()
    T = {} if timings is None else timings
    res = dict(rate=float(rate), n=int(source.n), seconds=source.n / rate)
    # ---- pass 1: PSD and band
    sc = fe.scan_psd(source)
    fb = fe.find_bands(sc, rate, search_hz=search_hz)
    T['pass1'] = time.time() - t0
    res['dc'] = [float(sc['dc'].real), float(sc['dc'].imag)]
    res['n0'] = fb['n0']
    res['n_bands'] = len(fb['bands'])
    if not fb['bands']:
        res['stage'] = 'nosignal'
        return res
    b = fb['bands'][0]
    res['band'] = dict(lo=b['lo'], hi=b['hi'], centroid=b['centroid'], excess=b['excess'],
                       peak_ratio=b['peak_ratio'], mean_ratio=b['mean_ratio'])
    res['band_bw'] = b['hi'] - b['lo']
    res['rel_bw'] = res['band_bw'] / rate
    if len(fb['bands']) > 1:
        res['band2_rel'] = fb['bands'][1]['excess'] / b['excess']
    else:
        res['band2_rel'] = 0.0
    n0 = fb['n0']
    # carrier concentration and spectral shape from the PSD
    f, psd, df = sc['freqs'], sc['psd'], sc['df']
    inb = (f >= b['lo']) & (f <= b['hi'])
    exc = np.maximum(psd[inb] - n0, 0.0)
    fi = f[inb]
    if exc.sum() > 0:
        k = int(np.argmax(exc))
        res['peak3_frac'] = float(exc[max(0, k - 1):k + 2].sum() / exc.sum())
        cen = float((fi * exc).sum() / exc.sum())
        res['sigma_f'] = float(np.sqrt(((fi - cen) ** 2 * exc).sum() / exc.sum()))
        res['pk_off'] = float(fi[k] - cen)
        # bandwidth holding 90 % of the excess power (centroid-symmetric)
        order = np.argsort(np.abs(fi - cen))
        cum = np.cumsum(exc[order]) / exc.sum()
        res['bw90'] = float(2 * abs(fi[order][min(np.searchsorted(cum, 0.9), len(order) - 1)] - cen) + df)
        res['bw99'] = float(2 * abs(fi[order][min(np.searchsorted(cum, 0.99), len(order) - 1)] - cen) + df)
    else:
        res['peak3_frac'] = 0.0
        res['sigma_f'] = 0.0
        res['bw90'] = res['bw99'] = res['band_bw']
    res['snr_psd_db'] = 10 * math.log10(max(b['excess'], 1e-30) / (n0 * res['band_bw']))
    if res['bw99'] / rate > 0.55:
        res['stage'] = 'wideband'
        return res
    # ---- pass 2: channelise
    t1 = time.time()
    bw_ch = 1.3 * res['band_bw'] + 2e3
    bw_ch = min(bw_ch, 0.9 * rate)
    D = fe.choose_decimation(rate, bw_ch)
    rec, fs, info = fe.channelize_select(source, b['centroid'], bw_ch, D, n0, dc=sc['dc'], max_keep=MAX_FEAT)
    T['pass2'] = time.time() - t1
    res['blocks_seen'], res['blocks_active'] = info['blocks_seen'], info['blocks_active']
    res['early_stop'] = bool(info['truncated'])
    res['blk_duty'] = info['blocks_active'] / max(info['blocks_seen'], 1)
    res['D'], res['fs'], res['n_rec'], res['neb'] = D, fs, int(len(rec)), info['neb']
    res['bw_ch'] = bw_ch
    res['fc'] = info['fc']
    res['truncated'] = bool(info['truncated'])
    neb = info['neb']
    n_ch = n0 * neb
    t2 = time.time()
    p = (rec.real.astype(np.float32) ** 2 + rec.imag.astype(np.float32) ** 2)
    fmask, Lp = frame_mask(p, fs, neb, n_ch)
    act = float(fmask.mean())
    res['act'] = act
    st, ln, v = runs(fmask)
    fl = ln[v]
    # A continuous signal whose channel SNR sits near the detection threshold produces a PATCHY mask (many
    # frames only a few windows long).  Real frames are much longer than the K-sample window, so treat a
    # mask with short median frames (and not nearly full) as 'no usable burst structure' = weak/continuous.
    gaps = np.array([ln[i] for i in range(1, len(v) - 1) if not v[i]])
    patchy = bool(len(fl) > 0 and act < 0.99 and ((np.median(fl) < 10 * Lp and act < 0.9) or
                                                     (len(gaps) >= 3 and np.median(gaps) < 8 * Lp)))
    weak = act < 0.02 or patchy
    res['patchy'] = patchy
    if weak:
        fmask = np.ones(len(p), bool)
        st, ln, v = runs(fmask)
        fl = ln[v]
    res['weak'] = bool(weak)
    res['n_frames'] = int(len(fl))
    res['frame_med_s'] = float(np.median(fl) / fs) if len(fl) else 0.0
    # compact record: the frames (plus quiet margins), capped in length, used by all later features
    rec, fmask, brk = compact(rec, fmask, Lp, MAX_FEAT)
    p = (rec.real.astype(np.float32) ** 2 + rec.imag.astype(np.float32) ** 2)
    core = erode(fmask, Lp // 2) if not weak else fmask
    if core.sum() < 8 * Lp:
        core = fmask
    res['n_feat'] = int(len(rec))
    # on-state power / snr
    P_on = float(p[core].mean())
    snr_on = max(P_on / n_ch - 1.0, 1e-3)
    res['snr_on_db'] = 10 * math.log10(snr_on)
    res['n_ch'] = n_ch
    # envelope excess variance relative to a constant-envelope signal at the same SNR
    pc = p[core].astype(np.float64)
    q = np.partition(pc, int(len(pc) * 0.98))[int(len(pc) * 0.98):]
    res['top2_frac'] = float(q.sum() / max(pc.sum(), 1e-30))
    V = float(pc.var() / pc.mean() ** 2)
    Vc = (1 + 2 * snr_on) / (1 + snr_on) ** 2
    res['env_V'] = V
    res['env_exc'] = V - Vc
    # ---- spectral lines in |y|^2 (cyclic), PSK / OOK / AM
    ref = (np.median(fl) if len(fl) else len(p))
    seg = 1 << int(math.log2(max(512.0, min(0.8 * ref, 16384.0))))
    seg = max(512, min(seg, 1 << int(math.log2(max(512, len(p) // 8)))))
    res['seg'] = seg
    pn = (p / max(P_on, 1e-30)).astype(np.float32)
    sp_env = seg_spectrum(pn, fs, seg, mask=core)
    fmin = max(8.0 * fs / seg, 50.0)
    fmax = min(0.47 * fs, 1.2 * res['bw_ch'])
    lines_env = find_lines(sp_env, fmin, fmax)
    res['env_lines'] = lines_env
    res['env_fund'] = find_fundamental(sp_env, fmin, fmax)
    res['env_df'] = fs / seg
    T['features_env'] = time.time() - t2
    # ---- discriminator, delay-multiply lines (FSK), x^2 / x^4
    t3 = time.time()
    sig_f = max(res['sigma_f'], 0.15 * res['bw90'] / 2, 200.0)
    res['sig_f_used'] = sig_f
    z1 = rec[1:] * np.conj(rec[:-1])
    Ls = max(1, int(round(fs / (3.0 * sig_f))))
    zs = movsum_complex(z1, Ls)
    sfq = np.angle(zs) * (fs / (2 * np.pi))
    mk = core[1:]
    sv = sfq[mk]
    if len(sv) > 100:
        cen = float(np.median(sv))
        sv = sv - cen
        res['disc_Ls'] = Ls
        res['disc_sd'] = float(sv.std())
        res['disc_mabs'] = float(np.mean(np.abs(sv)))
        q = np.percentile(np.abs(sv), [10, 50, 90])
        res['disc_q'] = [float(x) for x in q]
        # bimodality: share of samples near the centre vs flanks, relative to the mean |f|
        mabs = res['disc_mabs']
        res['disc_centre_frac'] = float((np.abs(sv) < 0.35 * mabs).mean())
        # kurtosis
        m2 = float((sv ** 2).mean())
        res['disc_ku'] = float((sv ** 4).mean() / max(m2 * m2, 1e-30))
        # chirp indicator: a linear frequency sweep (LoRa up-chirps) has a one-signed slope almost all the time,
        # random FM / FSK data have a symmetric slope distribution
        m_ = max(2 * Ls, 4)
        if len(sfq) > 4 * m_:
            dd = (sfq[m_:] - sfq[:-m_])[core[1:][m_:] & core[1:][:-m_]]
            if len(dd) > 100:
                res['chirp_bias'] = float(abs((dd > 0).mean() - 0.5) * 2.0)
        # bimodality coefficient (sarle)
        m3 = float((sv ** 3).mean())
        sk = m3 / max(m2, 1e-30) ** 1.5
        res['disc_bc'] = (sk * sk + 1.0) / max(res['disc_ku'], 1e-9)
    # delay-and-multiply real-part series at tau ~ 1/(4 dev), 2/(4 dev)
    dm = []
    dmf = []
    for mult in (1.0, 2.0):
        tau = max(1, int(round(mult * fs / (4.0 * sig_f))))
        if tau >= len(rec) // 8:
            continue
        c = rec[tau:] * np.conj(rec[:-tau])
        u = np.real(c).astype(np.float32)
        sp = seg_spectrum(u, fs, seg, mask=core[tau:])
        lines = find_lines(sp, fmin, fmax)
        if lines:
            dm.append(dict(tau=tau, mult=mult, **lines[0]))
        fu = find_fundamental(sp, fmin, fmax)
        if fu:
            dmf.append(dict(tau=tau, mult=mult, **fu))
    res['dm_lines'] = dm
    res['dm_fund'] = dmf
    # discriminator-based refinement around the best delay-multiply fundamental
    if dmf and len(sv) > 100:
        best = max(dmf, key=lambda x: x['z'])
        f0 = best['f0']
        sps = fs / f0
        res['dm_sps'] = float(sps)
        # periodicity of the discriminator at lag T: random data -> ~0 ; tones / chirps -> large
        lag = int(round(sps))
        if 2 <= lag < len(sfq) // 4:
            a_, b_ = sfq[:-lag] - cen, sfq[lag:] - cen
            ok_ = core[1:][:-lag] & core[1 + lag:]
            if ok_.sum() > 100:
                res['disc_acf_T'] = float((a_[ok_] * b_[ok_]).sum() / max((a_[ok_] ** 2).sum(), 1e-30))
            lag2 = max(1, int(round(sps / 2)))
            a_, b_ = sfq[:-lag2] - cen, sfq[lag2:] - cen
            ok_ = core[1:][:-lag2] & core[1 + lag2:]
            if ok_.sum() > 100:
                res['disc_acf_T2'] = float((a_[ok_] * b_[ok_]).sum() / max((a_[ok_] ** 2).sum(), 1e-30))
        if sps >= 8:
            Lq = max(1, int(round(sps / 12)))
            zq = movsum_complex(z1, Lq)
            sq = np.angle(zq) * (fs / (2 * np.pi)) - cen
            h = 0.35 * float(np.mean(np.abs(sq[core[1:]])))
            bs = schmitt(sq, -h, h)
            st_, ln_, v_ = runs(bs)
            edges_ = st_[1:]
            if len(edges_) > 8:
                d_ = np.diff(edges_) / fs
                okd = core[1:][edges_[:-1]] & core[1:][edges_[1:]]
                est = estimate_unit(d_[okd], tmin=2.0 / fs * Lq)
                res['fsk_runs'] = dict(n_dur=int(okd.sum()), **(est or {}))
    T['features_fsk'] = time.time() - t3
    # x^2, x^4 delta lines (PSK order)
    t4 = time.time()
    seg2 = min(1 << 14, seg)
    for q in (2, 4):
        xq = rec ** q if q == 2 else (rec * rec) ** 2
        sp = seg_spectrum(xq.astype(np.complex64), fs, seg2, mask=core, max_seg=60, zpad=1)
        ln_ = find_lines(sp, -0.48 * fs, 0.48 * fs, nlines=1, hw_bins=60)
        if ln_:
            # sharpness: peak vs mean of ring bins [4, 40) away
            S = sp['S']
            k = int(np.argmax(S))
            ring = np.concatenate([S[max(0, k - 40):max(0, k - 4)], S[k + 5:k + 40]])
            pk = S[max(0, k - 1):k + 2].max()
            res['x%d_line' % q] = dict(ln_[0], delta=float(pk / max(ring.mean(), 1e-30)))
    T['features_x'] = time.time() - t4
    # ---- OOK edge-interval analysis
    t5 = time.time()
    if not weak or True:
        try:
            res['ook'] = ook_edge_analysis(p.astype(np.float64), fs, neb, n_ch, fmask, Lp, brk)
        except Exception as e:  # keep the harness alive; recorded for debugging
            res['ook_err'] = repr(e)
    T['features_ook'] = time.time() - t5
    res['stage'] = 'ok'
    T['total'] = time.time() - t0
    return res
