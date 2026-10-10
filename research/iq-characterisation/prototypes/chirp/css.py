"""Chirp-spread-spectrum (LoRa-like) detection on a channelised stream.

Input is y: complex64 at the critical rate BW (the sub-band [fc-BW/2, fc+BW/2) of the capture, see
frontend.Channeliser).  Two detectors, both blind to symbol timing, to the exact carrier frequency
and to the data (cyclic shifts of the base chirp):

  LAG-LINE   z[n] = y[n+1] * conj(y[n]).  For a linear chirp phi[n] = pi n^2 / M (M = 2^SF samples per
             symbol at the critical rate) z is a pure tone at +1/M cycles/sample (up-chirp; -1/M for a
             down-chirp) whatever the cyclic shift or CFO, and it is phase-continuous across the frequency
             wrap (a jump of exactly BW*(1/BW) = 1 cycle).  So one narrow spectral line, at BW/2^SF Hz, is
             coherent over the WHOLE packet, data symbols included.  Noise and every non-chirp signal give a
             flat or symmetric z spectrum.  z is box-car decimated by q = M/8 (the line then sits at 1/8
             cycles/sample), cut in L-point blocks (L = 64 -> 8 symbols), Hann-windowed, block mean removed.
             R = (power in the 3 bins around the line) / (3 x mean power in the background bins).

  DECHIRP    window of M samples (hop M/4), multiplied by conj(up-chirp) (and by the up-chirp for a
             down-chirp), rectangular FFT: a dominant tone per symbol.  R = (peak + 2 neighbours) / (3 x mean
             of the other bins).  A window that straddles two data symbols splits its energy over two
             tones (worst case 3 dB), which is why the hop is M/4.

The null (white Gaussian noise) of both is known: the dechirp bins are exactly i.i.d. exponential, so its tail
is closed-form; the Hann-windowed lag-line statistic is calibrated by Monte Carlo (nulls.py).
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------------------------
# lag-line
# ----------------------------------------------------------------------------------------------
def lag_z(y):
    """z[n] = y[n+1] * conj(y[n])  (computed once per channel)."""
    return y[1:] * np.conj(y[:-1])


def bg_bins(L, q_div=8, near=3, hi_mult=2.0):
    """Background bins around a line at k0 = L/q_div: upper side k0+near .. hi_mult*k0 + near-ish, lower side
    2 .. k0-near (never bin 0 or the DC leakage bin 1).  A LOCAL background: data-modulated signals (slow FSK/OOK/AM/
    NFM) have a coloured z spectrum that falls off with frequency, and a far-away background then makes their
    ordinary spectrum look like a line."""
    k0 = L // q_div
    hi = list(range(k0 + near, min(L // 2, int(round(hi_mult * k0)) + near * 3) + 1))
    lo = [k for k in range(2, k0 - near + 1)]
    return np.array(lo + hi)


def block_stats(seg, L=64, q_div=8, trim=1, near=3, hi_mult=2.0):
    """seg: (nb, L) complex decimated-z blocks.  Returns (R_up, R_dn): power in the 3 bins around the +-line
    (bin L/q_div) over 3 x trimmed-mean power of the LOCAL background bins (largest `trim` dropped)."""
    k0 = L // q_div
    w = np.hanning(L).astype(np.float32)
    seg = seg - seg.mean(1, keepdims=True)
    F = np.fft.fft(seg * w[None, :], axis=1)
    P = F.real * F.real + F.imag * F.imag
    idx_u = bg_bins(L, q_div, near, hi_mult)
    idx_d = (L - idx_u) % L
    res = []
    for k_line, idx in ((k0, idx_u), (L - k0, idx_d)):
        bgv = P[:, idx]
        nbg = bgv.shape[1]
        if trim > 0:
            bg = np.partition(bgv, nbg - trim - 1, axis=1)[:, :nbg - trim].mean(1)
        else:
            bg = bgv.mean(1)
        res.append((P[:, k_line - 1:k_line + 2].sum(1) / (3.0 * bg + 1e-30)).astype(np.float32))
    return res[0], res[1]


def lag_blocks(y, sf, L=64, hop_div=2, q_div=8, z=None):
    """Per-block line statistics for spreading factor `sf` (M = 2**sf samples per symbol at rate BW).

    returns dict(up=R_up[nb], dn=R_dn[nb], start=first y-sample index of each block, span=y samples per block)
    or None if y is too short."""
    M = 1 << sf
    q = max(1, M // q_div)
    if z is None:
        z = lag_z(y)
    m = len(z) // q
    if m < L:
        return None
    zd = z[:m * q].reshape(m, q).sum(1) if q > 1 else z[:m]
    hop = L // hop_div
    nb = (m - L) // hop + 1
    win = np.lib.stride_tricks.sliding_window_view(zd, L)[::hop][:nb]
    up, dn = block_stats(win, L, q_div)
    return dict(up=up, dn=dn, start=np.arange(nb) * hop * q, span=L * q, q=q, L=L, sf=sf)


# ----------------------------------------------------------------------------------------------
# dechirp
# ----------------------------------------------------------------------------------------------
def base_upchirp(M):
    n = np.arange(M, dtype=np.float64)
    return np.exp(1j * (np.pi * n * n / M - np.pi * n)).astype(np.complex64)


def dechirp_blocks(y, sf, hop_div=4):
    """Per-window dechirp statistics.  Returns dict(up, dn, bin_up, bin_dn, start, span)."""
    M = 1 << sf
    hop = max(1, M // hop_div)
    nw = (len(y) - M) // hop + 1
    if nw < 1:
        return None
    ref_u = np.conj(base_upchirp(M))            # up-chirp data * conj(up) -> tone
    ref_d = base_upchirp(M)                      # down-chirp data * up -> tone
    up = np.empty(nw, np.float32)
    dn = np.empty(nw, np.float32)
    bu = np.empty(nw, np.int32)
    bd = np.empty(nw, np.int32)
    B = max(1, (1 << 20) // M)
    ar = np.arange(M)[None, :]
    for s in range(0, nw, B):
        e = min(nw, s + B)
        seg = y[(np.arange(s, e)[:, None] * hop) + ar]
        for ref, out, bn in ((ref_u, up, bu), (ref_d, dn, bd)):
            F = np.fft.fft(seg * ref[None, :], axis=1)
            P = F.real.astype(np.float32) ** 2 + F.imag.astype(np.float32) ** 2
            k = P.argmax(1)
            tri = P + np.roll(P, 1, 1) + np.roll(P, -1, 1)          # circular triple sums
            kt = tri.argmax(1)
            tot = P.sum(1)
            tsel = tri[np.arange(e - s), kt]
            out[s:e] = tsel / (np.maximum(tot - tsel, 1e-30) / (M - 3) * 3.0)
            bn[s:e] = kt
    return dict(up=up, dn=dn, bin_up=bu, bin_dn=bd, start=np.arange(nw) * hop, span=M, M=M, sf=sf)


def dechirp_null_tail(M, t):
    """Union-bound P(R > t) per window for white noise (closed form; rectangular FFT bins are i.i.d. Exp(1)).
    R = triple-sum / (3 * mean of the other M-3 bins)."""
    a = M - 3
    th = 1.0 / a
    s = 3.0 * t
    tot = 0.0
    for k in range(3):
        g = math.exp(math.lgamma(a + k) - math.lgamma(a) + k * math.log(th) - (a + k) * math.log1p(s * th))
        tot += (s ** k) / math.factorial(k) * g
    return min(1.0, M * tot)


def dechirp_threshold(M, p):
    """Smallest t with dechirp_null_tail(M, t) <= p (bisection)."""
    lo, hi = 1.0, 2.0
    while dechirp_null_tail(M, hi) > p:
        hi *= 2.0
        if hi > 1e9:
            return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dechirp_null_tail(M, mid) > p:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------------------
# packet structure evidence from the dechirp tone track
# ----------------------------------------------------------------------------------------------
def preamble_run(dc, thr, M, tol=1):
    """Longest run of >= 2 *consecutive symbols* (windows spaced by M samples, i.e. every 4th window of the
    hop-M/4 sequence) whose dechirp tone is strong and sits in the same bin (+-tol).  A preamble is 8 identical
    up-chirps; random payload does not repeat.  Returns the best run length over the 4 interleaved phases."""
    best = 0
    up, bu = dc['up'], dc['bin_up']
    for ph in range(4):
        r = up[ph::4]
        b = bu[ph::4]
        ok = r > thr
        run = 0
        prev = None
        for i in range(len(r)):
            if ok[i] and (prev is not None and min((b[i] - prev) % M, (prev - b[i]) % M) <= tol):
                run += 1
            elif ok[i]:
                run = 1
            else:
                run = 0
            prev = b[i] if ok[i] else None
            best = max(best, run)
    return best
