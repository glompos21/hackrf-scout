"""Burst-gated spectrum: the PSD of ONLY the time segments where the chirp test fired.

A time-averaged PSD of a 10 % duty signal is 10 dB shallower than the signal's own spectrum; the lag-line / dechirp
test already tells us WHEN the signal is on, so the spectrum over exactly those segments gives the occupied bandwidth
and the centre frequency with the full on-state SNR.  Used to (a) reject narrow-band look-alikes (AM / NFM tones, CW
with modulation, slow keying) that fool the lag-line statistic, (b) resolve the BW ambiguity between candidate
channels with the same chirp rate (125 kHz / SF9 ~ 250 kHz / SF11), (c) give the centre frequency.
"""
from __future__ import annotations

import numpy as np


def gated_psd(src, mean, segments, rate, nfft=1024, max_samples=6_000_000):
    """segments: list of (start, length) in input samples.  Returns (freqs, psd) (fftshifted, power/Hz, mean
    removed) or None if no complete segment could be read."""
    w = np.hanning(nfft).astype(np.float32)
    wn = float((w.astype(np.float64) ** 2).sum())
    acc = np.zeros(nfft, np.float64)
    nseg = 0
    used = 0
    for a, n in segments:
        if used >= max_samples:
            break
        x = src.read(int(a), int(n))
        if len(x) < nfft:
            continue
        x = x - np.complex64(mean)
        m = (len(x) - nfft) // (nfft // 2) + 1
        idx = np.arange(m)[:, None] * (nfft // 2) + np.arange(nfft)[None, :]
        F = np.fft.fft(x[idx] * w[None, :], axis=1)
        acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        nseg += m
        used += len(x)
    if nseg == 0:
        return None
    psd = acc / (nseg * rate * wn)
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate)), np.fft.fftshift(psd), nseg


def occupied(freqs, psd, nseg, floor, fc, win_hz=370e3, smooth_hz=6e3):
    """Occupied band of the strongest spectral component within +-win_hz of fc, from the excess PSD (psd - floor).
    The component is the connected set of smoothed bins above max(4 sigma, 25 % of the median excess of the bins
    above 4 sigma) that contains the maximum (gaps of <= 3 bins are bridged, so spectral horns / ripple do not split a
    flat-topped signal).  Returns dict(measurable, bw, centre, lo, hi, plateau_db, n_other) - `n_other` counts the
    other separate components in the window."""
    df = freqs[1] - freqs[0]
    sel = np.abs(freqs - fc) <= win_hz
    f, ex = freqs[sel], (psd - floor)[sel]
    s = max(1, int(round(smooth_hz / df)))
    exs = np.convolve(ex, np.ones(s) / s, mode='same')
    sd = floor / np.sqrt(max(nseg / 1.8, 1.0) * s)
    big = exs > 4.0 * sd
    out = dict(measurable=False)
    if big.sum() < 3:
        return out
    wide = np.convolve(exs, np.ones(9) / 9.0, mode='same')
    # plateau level: start from the strongest smoothed bin and iterate (region above 25 % of the plateau -> median of the
    # region = new plateau); the iteration ignores skirts / sidelobes at high SNR and spectral horns on a flat top
    plateau = float(wide.max())
    for _ in range(3):
        thr = max(4.0 * sd, 0.25 * plateau)
        mask = exs > thr
        dil = np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 1             # closing: dilate ...
        mask = mask | (np.convolve(dil.astype(np.int8), np.ones(7, np.int8), mode='same') >= 7)    # ... then erode
        if mask.sum() < 3:
            return out
        plateau = float(np.median(exs[mask]))
    med = plateau
    k = int(np.argmax(np.where(mask, wide, -1e30)))
    if not mask[k]:
        return out
    lo = k
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    st = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])) == 1)
    out.update(measurable=True, bw=float((hi - lo + 1) * df), centre=float(0.5 * (f[lo] + f[hi])), lo=float(f[lo]),
               hi=float(f[hi]), plateau_db=float(10 * np.log10(1 + med / floor)), n_other=int(len(st) - 1))
    return out
