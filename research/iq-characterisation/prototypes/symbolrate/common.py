"""Small numpy helpers shared by the feature extractors."""
from __future__ import annotations

import math

import numpy as np


def movmean(x, L):
    """Centered moving average with edge handling by shrinking the window (cumsum, float64)."""
    L = int(max(1, L))
    if L == 1:
        return x.astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - L // 2)
    hi = np.minimum(n, lo + L)
    lo = np.maximum(0, hi - L)
    return (c[hi] - c[lo]) / (hi - lo)


def movsum_complex(z, L):
    L = int(max(1, L))
    if L == 1:
        return z.astype(np.complex128)
    cr = np.concatenate([[0.0], np.cumsum(z.real, dtype=np.float64)])
    ci = np.concatenate([[0.0], np.cumsum(z.imag, dtype=np.float64)])
    n = len(z)
    i = np.arange(n)
    lo = np.maximum(0, i - L // 2)
    hi = np.minimum(n, lo + L)
    lo = np.maximum(0, hi - L)
    return (cr[hi] - cr[lo]) + 1j * (ci[hi] - ci[lo])


def runs(mask):
    """Run-length encode a boolean array -> (starts, lengths, values)."""
    m = np.asarray(mask, bool)
    if len(m) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    d = np.flatnonzero(m[1:] != m[:-1]) + 1
    starts = np.concatenate([[0], d])
    ends = np.concatenate([d, [len(m)]])
    return starts, ends - starts, m[starts]


def otsu(v, nb=128, lo=None, hi=None):
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi <= lo:
        return lo, 0.0
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    c = (e[:-1] + e[1:]) / 2
    mu = np.cumsum(p * c)
    mt = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (mt * w - mu) ** 2 / (w * (1 - w))
    s[~np.isfinite(s)] = 0
    k = int(np.argmax(s))
    tot = float((p * (c - mt) ** 2).sum())
    return float(e[k + 1]), float(s[k] / tot) if tot > 0 else 0.0


def parabolic_peak(y, k):
    """Sub-bin position of the maximum near index k using a log-parabola on 3 points."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = np.log(max(y[k - 1], 1e-300)), np.log(max(y[k], 1e-300)), np.log(max(y[k + 1], 1e-300))
    den = a - 2 * b + c
    if den >= 0:
        return float(k)
    return float(k + 0.5 * (a - c) / den)


def welch_real(u, fs, nfft, mask=None, window='hann', zpad=1):
    """Averaged periodogram of a real/complex series using only nfft-segments that are fully inside
    `mask` (bool, same length) if given.  50 % overlap.  Returns (f, S, nseg)."""
    n = len(u)
    if n < nfft:
        return None, None, 0
    hop = nfft // 2
    nseg = (n - nfft) // hop + 1
    w = np.hanning(nfft) if window == 'hann' else np.ones(nfft)
    idx0 = np.arange(nseg) * hop
    if mask is not None:
        c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        full = (c[idx0 + nfft] - c[idx0]) == nfft
        idx0 = idx0[full]
    if len(idx0) == 0:
        return None, None, 0
    acc = None
    cplx = np.iscomplexobj(u)
    step = max(1, (1 << 21) // nfft)
    for s in range(0, len(idx0), step):
        ii = idx0[s:s + step]
        seg = u[ii[:, None] + np.arange(nfft)[None, :]]
        seg = seg - seg.mean(axis=1, keepdims=True)
        seg = seg * w
        if cplx:
            F = np.fft.fft(seg, n=nfft * zpad, axis=1)
        else:
            F = np.fft.rfft(seg, n=nfft * zpad, axis=1)
        P = (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        acc = P if acc is None else acc + P
    S = acc / len(idx0) / float((w ** 2).sum())
    if cplx:
        f = np.fft.fftfreq(nfft * zpad, 1.0 / fs)
    else:
        f = np.fft.rfftfreq(nfft * zpad, 1.0 / fs)
    return f, S, len(idx0)


def line_stat(f, S, nseg, fmin, fmax, excl=2, med_hw=40, zpad=1):
    """Find the strongest narrow line of an averaged periodogram within [fmin, fmax].

    ratio = S[k] / local_median(S) (median over +-med_hw bins excluding +-excl bins).
    A noise-only bin of an average of nseg periodograms is ~ Gamma(nseg)/nseg, median/mean = ln2-ish
    so we quote the ratio to the local MEAN-equivalent (median/0.69 for large dof; exact for dof 2).
    Returns dict(f, ratio, z, k) with z = (ratio-1)*sqrt(nseg)."""
    if f is None:
        return None
    sel = np.flatnonzero((f >= fmin) & (f <= fmax))
    if len(sel) < 3:
        return None
    # local floor: running median over a wide window (computed on a decimated copy for speed)
    n = len(S)
    hw = med_hw * zpad
    # median filter by sorting windows of strided samples
    floor = np.empty(n)
    # use block medians (non-overlapping, hw wide) interpolated
    blk = max(2 * hw, 8)
    nb = int(math.ceil(n / blk))
    meds = np.array([np.median(S[max(0, i * blk - hw):min(n, (i + 1) * blk + hw)]) for i in range(nb)])
    cen = (np.arange(nb) + 0.5) * blk
    floor = np.interp(np.arange(n), cen, meds)
    # correct median -> mean for averaged periodograms (gamma with nseg dof*2 ~ chi2); factor ln2 for nseg=1
    k_dof = max(nseg, 1)
    # median of Gamma(k,1/k) approx 1 - 1/(3k)  (k>=1 reasonable); for k=1 it is ln 2 = 0.693
    corr = 0.6931 if k_dof == 1 else (1.0 - 1.0 / (3.0 * k_dof) + 0.0)
    floor = floor / corr
    ratio = S / np.maximum(floor, 1e-300)
    k = sel[int(np.argmax(ratio[sel]))]
    kp = parabolic_peak(S, k)
    fpk = float(np.interp(kp, np.arange(n), f)) if n > 1 else float(f[k])
    return dict(f=fpk, ratio=float(ratio[k]), z=float((ratio[k] - 1.0) * math.sqrt(k_dof)), k=int(k),
                df=float(f[1] - f[0]) / zpad * zpad)
