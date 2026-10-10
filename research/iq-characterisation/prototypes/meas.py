"""Independent measurement helpers (use only res['iq'] and res['truth']; never generator internals)."""
import math

import numpy as np


def welch(x, fs, nfft):
    """Hann-window Welch PSD in power/Hz (sum(psd)*df == mean |x|^2).  Returns (freqs, psd) fftshifted."""
    nfft = int(min(nfft, len(x) // 4))
    w = np.hanning(nfft)
    hop = nfft // 2
    nseg = (len(x) - nfft) // hop + 1
    acc = np.zeros(nfft)
    B = max(1, (1 << 21) // nfft)
    ar = np.arange(nfft)[None, :]
    for s in range(0, nseg, B):
        e = min(nseg, s + B)
        seg = x[(np.arange(s, e)[:, None] * hop) + ar] * w.astype(np.float32)
        F = np.fft.fft(seg, axis=1)
        acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
    psd = acc / (nseg * fs * float((w ** 2).sum()))
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs)), np.fft.fftshift(psd), nseg


def pick_nfft(rate, bw, n):
    target = rate / max(bw / 24.0, 1.0)
    nf = 1 << int(round(math.log2(max(target, 512))))
    return int(min(max(nf, 512), 1 << 16, max(512, n // 8)))


def _inband_mask(f, bands, widen=0.0):
    m = np.zeros(len(f), bool)
    for lo, hi in bands:
        m |= (f >= lo - widen) & (f <= hi + widen)
    return m


def snr_estimate(x, truth, bands=None, bw=None, duty=None, remove_dc=False, all_bands=None):
    """Measured in-band, on-state SNR [dB] and spectral centroid, from the samples only.

    noise PSD: mean PSD in the out-of-band region (outside every band + a guard)
    signal power in band = sum(PSD) over band bins - N0 * band width;  on-state => / duty."""
    rate = truth['rate']
    bands = bands if bands is not None else truth['bands_hz']
    bw = bw if bw is not None else truth['occupied_bw_hz']
    allb = all_bands if all_bands is not None else bands
    duty = duty if duty is not None else truth['on_fraction']
    x = np.asarray(x)
    if remove_dc:
        x = x - x.mean()
    nfft = pick_nfft(rate, bw, len(x))
    f, P, nseg = welch(x, rate, nfft)
    df = f[1] - f[0]
    lo_min, hi_max = min(b[0] for b in allb), max(b[1] for b in allb)
    margin = min(lo_min + rate / 2, rate / 2 - hi_max)
    guard = max(3 * df, min(0.5 * bw, 0.5 * margin))
    oob = ~_inband_mask(f, allb, guard)
    if remove_dc:
        oob &= np.abs(f) > 4 * df
    if oob.sum() < 30:
        raise RuntimeError('too few out-of-band bins (%d) to estimate the noise floor' % oob.sum())
    n0 = float(P[oob].mean())
    inb = _inband_mask(f, bands)
    p_in = float(P[inb].sum() * df)
    p_noise = n0 * inb.sum() * df
    p_sig = p_in - p_noise
    snr = 10 * math.log10(max(p_sig, 1e-12) / duty / (n0 * bw))
    # centroid of (P - n0) over the bands widened by half the occupied bandwidth
    reg = _inband_mask(f, bands, 0.5 * bw)
    wgt = (P - n0)[reg]
    cen = float((f[reg] * wgt).sum() / wgt.sum())
    return dict(snr_db=snr, centroid_hz=cen, n0=n0, df=df, nseg=nseg, p_sig=p_sig, duty=duty)


def containment(x, truth, remove_dc=False):
    """Fraction of the (noise-subtracted) signal power inside the nominal occupied bandwidth, and the
    smallest symmetric (about the centroid) bandwidth holding 99 % / 99.9 % of it."""
    rate = truth['rate']
    bands = truth['bands_hz']
    bw = truth['occupied_bw_hz']
    nfft = pick_nfft(rate, bw, len(x))
    f, P, _ = welch(x, rate, nfft)
    df = f[1] - f[0]
    lo_min, hi_max = min(b[0] for b in bands), max(b[1] for b in bands)
    margin = min(lo_min + rate / 2, rate / 2 - hi_max)
    guard = max(3 * df, min(0.5 * bw, 0.5 * margin))
    oob = ~_inband_mask(f, bands, guard)
    n0 = float(P[oob].mean())
    Ps = np.maximum(P - n0, 0.0)
    tot = Ps.sum()
    frac_in = float(Ps[_inband_mask(f, bands)].sum() / tot)
    c = float((f * Ps).sum() / tot)
    order = np.argsort(np.abs(f - c))
    cum = np.cumsum(Ps[order]) / tot
    bw99 = 2 * abs(f[order][np.searchsorted(cum, 0.99)] - c) + df
    bw999 = 2 * abs(f[order][min(np.searchsorted(cum, 0.999), len(order) - 1)] - c) + df
    return dict(frac_in=frac_in, bw99=float(bw99), bw999=float(bw999), df=df)


def _otsu(v, nb=128):
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    mt = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (mt * w - mu) ** 2 / (w * (1 - w))
    s[~np.isfinite(s)] = 0
    return float(e[int(np.argmax(s)) + 1])


def _erlang_upper(m, p):
    """x such that P(mean of m unit exponentials > x) = p (bisection on the Erlang tail)."""
    def tail(x):
        mx = m * x
        k = np.arange(m)
        logt = -mx + k * np.log(mx) - np.array([math.lgamma(i + 1) for i in k])
        return float(np.exp(logt).sum())
    lo, hi = 1.0, 2.0
    while tail(hi) > p:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tail(mid) > p:
            lo = mid
        else:
            hi = mid
    return hi


def envelope_events(x, truth, bands=None, on_intervals=None, n0=None, p_on=None):
    """Channel filter (mix to band centre, smooth brick-wall low-pass via FFT, decimate) -> |.|^2 ->
    moving average over ~feature/3 -> threshold derived from the *measured* noise PSD n0 (Welch, out of band)
    and measured on-state power p_on -> runs.  Compared with truth['on_intervals'] by event matching."""
    rate = truth['rate']
    bands = bands if bands is not None else truth['bands_hz']
    on_iv = on_intervals if on_intervals is not None else truth['on_intervals']
    lo, hi = min(b[0] for b in bands), max(b[1] for b in bands)
    fc, span = 0.5 * (lo + hi), hi - lo
    n = len(x)
    tm = np.zeros(n, bool)
    for a, b in on_iv:
        tm[int(round(a * rate)):int(round(b * rate))] = True
    st, en = _runs(tm)
    if len(st) == 0:
        return None
    gaps = (st[1:] - en[:-1]) / rate if len(st) > 1 else np.array([1.0])
    feat = float(min(((en - st) / rate).min(), gaps.min() if len(gaps) else 1.0))
    cutoff = min(0.55 * span, rate / 2 / 1.3)
    t = np.arange(n, dtype=np.float64)
    y = x * np.exp(-2j * np.pi * np.remainder(t * (fc / rate), 1.0)).astype(np.complex64)
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(n, 1.0 / rate)
    H = 0.5 - 0.5 * np.cos(np.pi * np.clip((1.3 * cutoff - np.abs(f)) / (0.3 * cutoff), 0, 1))
    yf = np.fft.ifft(Y * H.astype(np.float32))
    D = max(1, int(rate / (2.6 * cutoff)))
    yd = yf[::D]
    rd = rate / D
    noise = float(n0 * (rate / n) * np.sum(H.astype(np.float64) ** 2))     # E|noise|^2 per output sample
    p = (yd.real.astype(np.float64) ** 2 + yd.imag.astype(np.float64) ** 2)
    nd = len(yd)
    snr_eff = p_on / noise                       # on-state power / noise power inside the channel filter
    pfa = 1.0 / (2.0 * nd)                       # about one false alarm per capture
    cap = 1.0 + 0.5 * snr_eff                    # never above half-way to the on-state level
    mmax = int(max(1, feat * rd / 2.0))
    m = 1
    while m < mmax and _erlang_upper(m, pfa) > cap:
        m += 1
    ps = np.convolve(p, np.ones(m) / m, mode='same') if m > 1 else p
    thr = noise * min(_erlang_upper(m, pfa), cap)
    det = ps > thr
    ds, de = _runs(det)
    keep = (de - ds) >= max(1, m // 2)
    ds, de = ds[keep], de[keep]
    tr = np.zeros(nd, bool)
    idx = (np.arange(nd) * D)
    tr = tm[np.minimum(idx + D // 2, n - 1)]
    ts, te = _runs(tr)

    def overlap(a0, a1, bs, be):
        return bool(np.any((bs < a1) & (be > a0)))
    rec = np.mean([overlap(a, b, ds, de) for a, b in zip(ts, te)]) if len(ts) else 1.0
    pre = np.mean([overlap(a, b, ts, te) for a, b in zip(ds, de)]) if len(ds) else 0.0
    dm = np.zeros(nd, bool)
    for a, b in zip(ds, de):
        dm[a:b] = True
    iou = float((dm & tr).sum() / max(1, (dm | tr).sum()))
    errs = []
    for a, b in zip(ts, te):
        k = np.flatnonzero((ds < b) & (de > a))
        if len(k):
            errs.append(abs(ds[k[0]] - a) / rd)
    return dict(recall=float(rec), precision=float(pre), iou=iou, n_truth=int(len(ts)), n_det=int(len(ds)),
                edge_err_s=float(np.median(errs)) if errs else float('nan'), res_s=1.0 / rd,
                smooth_s=m / rd, feat_s=feat, thr_over_noise=float(thr / noise))


def _runs(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
